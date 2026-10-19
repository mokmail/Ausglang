"""The default agent driver: inbox, turns, steps, streaming and tool calls.

A *step* is one model request plus the tools it calls; a *turn* is zero or more
steps. The loop claims input, assembles the prompt, runs `agent/pre-step`,
`step/start`, `agent/request`, commits accepted facts to the session log, derives
and freezes the request from that log, streams the bound call, runs the tool
pipeline, and closes the turn once nothing is owed. Live `agent/*` events carry
what the UI needs; durable `session/event` facts carry everything replay needs.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from typing import Any, AsyncIterator, Callable

from .llm import ContextOverflow, LLMError
from .tools import ExecContext

__all__ = ["Agent", "AgentLoop", "AgentLoopPlugin", "InboxProjection", "extract_tool_calls"]


def _now() -> float:
    return time.time()


def extract_tool_calls(text: str, known: set[str]) -> list[dict[str, Any]]:
    """Read the JSON tool protocol a local model can actually emit.

    Accepts fenced blocks, several calls in one reply, and the `{"tool": ...}`
    shape the prompt documents. Unknown tool names are dropped before dispatch.
    """
    blobs = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text or "", re.S)
    if not blobs:
        blobs = re.findall(r'(\{[^{}]*"tool"\s*:.*?\})', text or "", re.S)
    calls: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for blob in blobs:
        try:
            obj = json.loads(blob)
        except json.JSONDecodeError:
            continue
        name = obj.get("tool") or obj.get("name")
        if not isinstance(name, str) or name not in known:
            continue
        args = obj.get("arguments") or obj.get("args") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"input": args}
        if not isinstance(args, dict):
            args = {}
        key = (name, json.dumps(args, sort_keys=True))
        if key in seen:
            continue
        seen.add(key)
        calls.append({"tool": name, "arguments": args})
    return calls


class InboxProjection:
    """Durable inbox view: pending input is readable without a live Agent."""

    def __init__(self) -> None:
        self.state: dict[str, Any] = {"pending": 0, "claimed": 0, "last": None}

    def fold(self, event: dict[str, Any]) -> None:
        t = event.get("type")
        if t == "agent/inbox/inserted":
            self.state["pending"] = int(self.state.get("pending") or 0) + 1
            self.state["last"] = event.get("text")
        elif t == "agent/inbox/claimed":
            self.state["pending"] = max(0, int(self.state.get("pending") or 0) - 1)
            self.state["claimed"] = int(self.state.get("claimed") or 0) + 1


class Agent:
    """One agent: one session, one inbox, one turn machine."""

    def __init__(self, ctx: Any, session: Any, config: dict[str, Any] | None = None,
                 *, scope: str = "agent", child_factory: Callable[..., Any] | None = None,
                 max_tool_rounds: int = 3) -> None:
        self.ctx = ctx
        self.session = session
        self.config = dict(config or {})
        self.scope = scope
        self.child_factory = child_factory
        self.max_tool_rounds = int(max_tool_rounds)
        self.exec_meta = dict(self.config.get("exec_meta") or {})
        self.inbox: list[dict[str, Any]] = []
        self.injected: list[dict[str, Any]] = []
        self.turns = 0
        self.cancelled = asyncio.Event()
        self.cancel_keep_inbox = False
        self.last_result: dict[str, Any] = {}
        self.session.projections.register("inbox", InboxProjection())

    # -- input ------------------------------------------------------------- #
    def followup(self, text: str, *, source: str = "prompt") -> dict[str, Any]:
        message = {"role": "user", "content": text, "source": source, "id": _now()}
        self.inbox.append(message)
        self.session.append("agent/inbox/inserted", text=text, source=source)
        self.ctx.emit("agent/inbox/inserted", {"agent": self, "message": message})
        return message

    def inject(self, text: str, *, source: str = "context") -> dict[str, Any]:
        """Add model-facing context that lands in the next admitted request."""
        message = {"role": "user", "content": text, "source": source, "id": _now()}
        self.injected.append(message)
        self.ctx.emit("agent/inbox/inserted", {"agent": self, "message": message})
        return message

    def cancel(self, *, keep_inbox: bool = False) -> None:
        self.cancel_keep_inbox = keep_inbox
        self.cancelled.set()
        self.ctx.emit("agent/status", {"agent": self, "state": "cancelling"})

    def _claim_prompt(self) -> list[dict[str, Any]]:
        claimed: list[dict[str, Any]] = []
        if self.inbox:
            claimed.append(self.inbox.pop(0))
        claimed.extend(self.injected)
        self.injected = []
        for message in claimed:
            self.session.append("agent/inbox/claimed", text=message.get("content"),
                                source=message.get("source"))
            self.ctx.emit("agent/inbox/claimed", {"agent": self, "message": message})
        return claimed

    def _claim_next(self) -> list[dict[str, Any]]:
        if not self.inbox:
            return []
        message = self.inbox.pop(0)
        self.session.append("agent/inbox/claimed", text=message.get("content"), source="steer")
        self.ctx.emit("agent/inbox/claimed", {"agent": self, "message": message})
        return [message]

    # -- run --------------------------------------------------------------- #
    async def run(self, text: str | None = None, *,
                  attachments: list[dict[str, Any]] | None = None) -> AsyncIterator[dict[str, Any]]:
        if text is not None:
            self.followup(text)
        for att in (attachments or []):
            self.inject(f"[attached file: {att.get('name') or 'attachment'}]\n```\n"
                        f"{str(att.get('text') or '')[:100_000]}\n```", source="attachment")
        if not self.inbox and not self.injected:
            yield {"type": "error", "message": "nothing to run"}
            return

        prompt = self.ctx.require("systemPrompt")
        tools = self.ctx.require("tools")
        llm = self.ctx.require("llm")

        self.turns += 1
        turn = len(self.session.of_type("turn/start")) + 1
        self.turns = turn
        started = _now()
        self.session.append("turn/start", turn=turn)
        self.ctx.emit("agent/status", {"agent": self, "state": "running", "turn": turn})
        yield {"type": "turn/start", "turn": turn}

        final_text = ""
        reasoning = ""
        usage_total = {"prompt_tokens": 0, "completion_tokens": 0}
        tool_events: list[dict[str, Any]] = []
        rounds = 0
        prompt_report: dict[str, Any] = {}
        step = 0

        while True:
            step += 1
            claimed = self._claim_prompt() if step == 1 else self._claim_next()

            assembly = await prompt.assemble_async(scope=self.scope,
                                                   tools=tools.schemas(self.scope))
            system_text = prompt.render(assembly)
            prompt_report = assembly.report

            pre = await self.ctx.waterfall("agent/pre-step", {
                "agent": self, "session": self.session, "messages": list(claimed),
                "system": system_text, "tools": assembly.tools, "config": self.config,
                "decision": "enter", "step": step,
            })
            if not isinstance(pre, dict):
                pre = {"decision": "enter"}
            if pre.get("decision") == "reject":
                if step == 1:
                    self.session.append("turn/end", turn=turn, rejected=True,
                                        reason=str(pre.get("reason") or ""))
                    self.ctx.emit("agent/status", {"agent": self, "state": "idle", "turn": turn})
                    yield {"type": "rejected", "reason": pre.get("reason") or ""}
                    return
                break
            entered = pre.get("messages")
            if not isinstance(entered, list):
                entered = claimed
            if step == 1 and not entered:
                # a first enter rewritten empty closes the turn without a step
                break

            self.session.append("step/start", step=step)
            yield {"type": "step/start", "step": step}

            request = await self.ctx.waterfall("agent/request", {
                "agent": self, "config": dict(self.config), "system": system_text,
                "tools": assembly.tools,
            })
            if isinstance(request, dict) and isinstance(request.get("config"), dict):
                request = request["config"]
            if not isinstance(request, dict):
                request = dict(self.config)
            prepared = llm.prepare_call(request)

            # commit the prompt and the accepted users before deriving history
            last_system = next((e for e in reversed(self.session.of_type("system/message"))
                                if e.get("text")), None)
            if system_text.strip() and (last_system is None or last_system.get("text") != system_text):
                self.session.append("system/message", text=system_text,
                                    sections=len(assembly.sections))
            for message in entered:
                self.session.append("user/message", text=message.get("content") or "",
                                    source=message.get("source") or "prompt")

            messages = self.session.derive_messages()
            frozen = [dict(m) for m in messages]
            self.ctx.emit("agent/request", {"agent": self, "model": prepared.model,
                                            "messages": len(frozen)})

            attempt = 0
            out = ""
            reasoning_delta = ""
            step_usage = {"prompt_tokens": 0, "completion_tokens": 0}
            error: Exception | None = None
            while True:
                attempt += 1
                result = {"text": "", "reasoning": "", "usage": step_usage, "error": None}
                async for frame in self._stream(llm, prepared, frozen, assembly.tools, result):
                    yield frame
                out, reasoning_delta = result["text"], result["reasoning"]
                step_usage = result["usage"]
                error = result["error"]
                if error is None:
                    break
                self.session.append("assistant/attempt", text=out,
                                    error=str(error), finish="error")
                handled = await self.ctx.waterfall("agent/request-error", {
                    "agent": self, "error": error, "config": request,
                    "messages": frozen, "attempt": attempt,
                })
                if isinstance(handled, dict) and handled.get("action") == "retry" and attempt < 3:
                    fresh = self.session.derive_messages()
                    frozen = [dict(m) for m in fresh]
                    self.session.append("step/retry", attempt=attempt,
                                        reason=str(handled.get("reason") or ""))
                    continue
                if isinstance(error, ContextOverflow):
                    raise error
                raise LLMError(str(error))

            reasoning += reasoning_delta
            usage_total["completion_tokens"] += step_usage.get("completion_tokens", 0)
            usage_total["prompt_tokens"] = max(usage_total["prompt_tokens"],
                                               step_usage.get("prompt_tokens", 0))

            if self.cancelled.is_set():
                self.session.append("assistant/message", text=out, interrupted=True,
                                    usage=step_usage, finish="cancelled")
                final_text += out
                self.session.append("step/end", step=step, outcome="cancelled")
                break

            self.session.append("assistant/message", text=out, usage=step_usage,
                                finish="stop", elapsed=round(_now() - started, 2))
            yield {"type": "message", "text": out, "usage": step_usage}

            calls = extract_tool_calls(out, set(tools.names(self.scope)))
            if not calls:
                # only an answer contributes to the user-visible result; a
                # tool-call turn stays in history as the call it was
                final_text = out
            if calls and rounds < self.max_tool_rounds and not self.cancelled.is_set():
                rounds += 1
                exec_ctx = ExecContext(session=self.session, agent=self,
                                       signal=self.cancelled,
                                       meta={**self.exec_meta, "round": rounds})
                for call in calls:
                    self.session.append("tool/call", tool=call["tool"],
                                        arguments=call["arguments"], round=rounds)
                results = await tools.run_many(calls, exec_ctx, scope=self.scope)
                for call, result in zip(calls, results):
                    self.session.append("tool/result", tool=result.tool,
                                        status=result.status, content=result.content[:6000])
                    tool_events.append({**result.as_event(), "arguments": call["arguments"]})
                    yield {"type": "tool", "tool": result.tool, "server": result.server,
                           "state": "done" if result.status == "ok" else "error",
                           "result": result.content[:4000], "status": result.status,
                           "arguments": call["arguments"]}
                for art in (exec_ctx.meta.get("artifacts") or []):
                    if art:
                        yield {"type": "artifact", "artifact": art}
                exec_ctx.meta["artifacts"] = []
                self.session.append("step/end", step=step, outcome="tools")
                continue

            self.session.append("step/end", step=step, outcome="answer")
            if self.inbox:
                continue
            break

        stopping = await self.ctx.waterfall("agent/turn-stopping", {
            "agent": self, "session": self.session, "text": final_text,
        })
        if isinstance(stopping, dict) and stopping.get("text") is not None:
            final_text = str(stopping["text"])

        elapsed = round(_now() - started, 2)
        self.session.append("turn/end", turn=turn, text=final_text,
                            usage=usage_total, elapsed=elapsed)
        self.ctx.emit("agent/status", {"agent": self, "state": "idle", "turn": turn})
        self.last_result = {
            "text": final_text.strip(),
            "reasoning": reasoning,
            "usage": usage_total,
            "tool_events": tool_events,
            "rounds": rounds,
            "elapsed_s": elapsed,
            "prompt": prompt_report,
            "session_id": self.session.id,
            "compactions": (self.session.projections.state_of("usage") or {}).get("compactions", 0),
        }
        yield {"type": "done", **self.last_result}

    async def _stream(self, llm: Any, prepared: Any, messages: list[dict[str, Any]],
                      tools: list[dict[str, Any]], result: dict[str, Any],
                      ) -> AsyncIterator[dict[str, Any]]:
        """Stream one attempt; frames are yielded, settlements land in `result`."""
        parts: list[str] = []
        reasoning_parts: list[str] = []
        usage = {"prompt_tokens": 0, "completion_tokens": 0}
        error: Exception | None = None
        self.ctx.emit("agent/assistant-stream/start", {"agent": self, "model": prepared.model})
        try:
            async for chunk in llm.stream(prepared, messages, tools, self.cancelled):
                if chunk.kind == "text":
                    parts.append(chunk.text)
                    yield {"type": "delta", "text": chunk.text}
                elif chunk.kind == "reasoning":
                    reasoning_parts.append(chunk.text)
                    yield {"type": "reasoning", "text": chunk.text}
                elif chunk.kind == "usage":
                    usage["prompt_tokens"] = max(usage["prompt_tokens"],
                                                 chunk.usage.get("prompt_tokens", 0))
                    usage["completion_tokens"] += chunk.usage.get("completion_tokens", 0)
        except asyncio.CancelledError:
            self.cancelled.set()
        except Exception as exc:
            error = exc
        result.update(text="".join(parts), reasoning="".join(reasoning_parts),
                      usage=usage, error=error)
        self.ctx.emit("agent/assistant-stream/end",
                      {"agent": self, "chars": len(result["text"])})


class AgentLoop:
    """ctx.agentLoop — creates agents and owns their teardown."""

    def __init__(self, ctx: Any, config: dict[str, Any] | None = None) -> None:
        self.ctx = ctx
        self.config = dict(config or {})
        self._agents: dict[str, Agent] = {}

    def create(self, session: Any, config: dict[str, Any] | None = None,
               *, scope: str = "agent", child_factory: Callable[..., Any] | None = None,
               max_tool_rounds: int | None = None) -> Agent:
        agent = Agent(self.ctx, session, config or {},
                      scope=scope, child_factory=child_factory,
                      max_tool_rounds=int(max_tool_rounds or self.config.get("maxToolRounds") or 3))
        self._agents[session.id] = agent
        self.ctx.emit("agent/created", {"agent": agent, "session": session.id})
        return agent

    def get(self, session_id: str) -> Agent | None:
        return self._agents.get(session_id)

    def dispose(self, session_id: str) -> None:
        agent = self._agents.pop(session_id, None)
        if agent is not None:
            agent.cancel()
            self.ctx.emit("agent/disposed", {"agent": agent})


class AgentLoopPlugin:
    name = "agent-loop"
    inject = ("agentSessionStore", "systemPrompt", "tools", "llm")

    def __init__(self, ctx: Any, config: dict[str, Any] | None = None) -> None:
        self.ctx = ctx
        self.config = dict(config or {})
        self.loop = AgentLoop(ctx, self.config)
        ctx.provide("agentLoop", self.loop)

    def apply(self) -> None:
        pass
