"""compaction-basic: pressure-triggered condensation with overflow recovery.

Mirrors the harness contract:

* trigger at `floor(min(W × thresholdRatio, W − O − headroom))`
* keep the newest `retainRatio` of `W − O` verbatim
* prune oversized tool results first (no model call), summarize only if needed
* replace the oldest balanced span with one durable summary marker
* recover from a confirmed context overflow by condensing and retrying, once
"""

from __future__ import annotations

from typing import Any

from .llm import ContextOverflow, estimate_tokens

__all__ = ["CompactionBasic", "CompactionPlugin"]

SUMMARY_PROMPT = (
    "You condense an agent conversation. Keep every fact, decision, identifier, "
    "file name, command, result and open question that later turns may need. "
    "Drop pleasantries and repetition. Write compact prose or bullets, no preamble."
)


class CompactionBasic:
    """The service behind the plugin; also callable on demand (`/compact`)."""

    def __init__(self, ctx: Any, config: dict[str, Any] | None = None) -> None:
        self.ctx = ctx
        self.config = dict(config or {})
        self.threshold_ratio = float(self.config.get("thresholdRatio") or 0.8)
        self.retain_ratio = float(self.config.get("retainRatio") or 0.16)
        self.retain_tokens = self.config.get("retainTokens")
        self.headroom = int(self.config.get("headroomTokens") or 4096)
        self.auto = bool(self.config.get("auto", True))
        self.max_overflow_retries = int(self.config.get("maxOverflowRetries") or 1)
        self.prune_chars = int(self.config.get("pruneToolResultChars") or 1800)
        self.overflow_used = 0
        self.running = False
        self.last: dict[str, Any] | None = None

    # -- pressure ---------------------------------------------------------- #
    def window(self) -> int:
        return int(self.config.get("contextWindow") or 12288)

    def output_cap(self) -> int:
        return int(self.config.get("maxTokens") or 512)

    def threshold(self) -> int:
        w, o = self.window(), self.output_cap()
        return int(min(w * self.threshold_ratio, w - o - self.headroom))

    def retain(self) -> int:
        if self.retain_tokens:
            return int(self.retain_tokens)
        return max(512, int((self.window() - self.output_cap()) * self.retain_ratio))

    def pressure(self, session: Any) -> int:
        return estimate_tokens(session.derive_messages())

    def over(self, session: Any) -> bool:
        return self.pressure(session) >= self.threshold()

    # -- condensation ------------------------------------------------------ #
    async def condense(self, session: Any, *, config: dict[str, Any] | None = None,
                       force: bool = False) -> dict[str, Any]:
        if self.running:
            return {"condensed": False, "reason": "already running"}
        self.running = True
        try:
            before = self.pressure(session)
            pruned = self.prune_tool_results(session)
            after_prune = self.pressure(session)
            if not force and after_prune < self.threshold():
                self.last = {"items": 0, "pruned": pruned, "freed": before - after_prune,
                             "summary": False}
                return {"condensed": pruned > 0, **self.last}
            summary, items = await self.summarize(session, config or {})
            if not summary or items <= 0:
                self.last = {"items": 0, "pruned": pruned, "freed": before - after_prune,
                             "summary": False}
                return {"condensed": pruned > 0, **self.last}
            after = self.pressure(session)
            self.last = {"items": items, "pruned": pruned, "freed": before - after,
                         "summary": True}
            return {"condensed": True, **self.last}
        finally:
            self.running = False

    def prune_tool_results(self, session: Any) -> int:
        """Trim oversized tool observations; no model call, no summary."""
        cutoff = int((session.last_compaction() or {}).get("upto") or 0)
        targets: dict[int, int] = {}
        for event in session.events:
            if event["seq"] <= cutoff or event.get("type") != "tool/result":
                continue
            content = str(event.get("content") or "")
            if len(content) > self.prune_chars:
                targets[event["seq"]] = self.prune_chars
        if not targets:
            return 0
        session.append("compaction/prune", targets=targets,
                       note="oversized tool results trimmed")
        return len(targets)

    async def summarize(self, session: Any,
                        config: dict[str, Any]) -> tuple[str, int]:
        """Replace the oldest span with one summary, keeping the recent tail."""
        events = [e for e in session.events if e["type"] in
                  ("user/message", "assistant/message", "tool/result", "system/message")]
        cutoff = int((session.last_compaction() or {}).get("upto") or 0)
        events = [e for e in events if e["seq"] > cutoff]
        if len(events) < 4:
            return "", 0

        budget = self.retain()
        keep: list[dict[str, Any]] = []
        spent = 0
        for event in reversed(events):
            text = str(event.get("text") or event.get("content") or "")
            cost = (len(text) + 3) // 4
            if spent + cost > budget and keep:
                break
            keep.append(event)
            spent += cost
        keep.sort(key=lambda e: e["seq"])
        if not keep:
            return "", 0
        split = keep[0]["seq"]
        span = [e for e in events if e["seq"] < split]
        if not span:
            return "", 0

        transcript = []
        for event in span:
            kind = event["type"]
            text = str(event.get("text") or event.get("content") or "")
            if not text.strip():
                continue
            label = {"user/message": "USER", "assistant/message": "ASSISTANT",
                     "tool/result": f"TOOL:{event.get('tool') or 'tool'}",
                     "system/message": "SYSTEM"}.get(kind, kind)
            transcript.append(f"{label}: {text[:1500]}")
        if not transcript:
            return "", 0

        summary = await self._ask(transcript, config)
        if not summary:
            return "", 0
        tokens_before = self.pressure(session)
        session.append("compaction/replace", upto=split, summary=summary,
                       items=len(span), tokens_before=tokens_before)
        return summary, len(span)

    async def _ask(self, transcript: list[str], config: dict[str, Any]) -> str:
        llm = self.ctx.get("llm")
        if llm is None:
            return ""
        messages = [
            {"role": "system", "content": SUMMARY_PROMPT},
            {"role": "user", "content": "Conversation to condense:\n\n"
                                        + "\n\n".join(transcript)[-24000:]},
        ]
        model = config.get("model") or self.config.get("model")
        if not model:
            return ""
        try:
            prepared = llm.prepare_call({
                "model": model,
                "max_tokens": int(self.config.get("maxTokens") or 600),
                "temperature": 0.2,
                "enable_thinking": False,
            })
            parts: list[str] = []
            async for chunk in llm.stream(prepared, messages, []):
                if chunk.kind == "text":
                    parts.append(chunk.text)
        except Exception:
            return ""
        return "".join(parts).strip()


class CompactionPlugin:
    name = "compaction-basic"
    inject = ("agentSessionStore",)

    def __init__(self, ctx: Any, config: dict[str, Any] | None = None) -> None:
        self.ctx = ctx
        self.config = dict(config or {})
        self.service = CompactionBasic(ctx, self.config)
        ctx.provide("compaction", self.service)
        ctx.on("agent/pre-step", self._pre_step, kind="waterfall", order=-50)
        ctx.on("agent/request-error", self._request_error, kind="waterfall", order=-50)

    def apply(self) -> None:
        pass

    async def _pre_step(self, payload: dict[str, Any], next_fn: Any) -> Any:
        service, session = self.service, payload.get("session")
        if session is None or not service.auto:
            return await next_fn(payload)
        if payload.get("step") == 1:
            service.overflow_used = 0
        if service.overflow_used == 0 and service.over(session):
            result = await service.condense(session, config=payload.get("config") or {})
            payload["compaction"] = result
        return await next_fn(payload)

    async def _request_error(self, payload: dict[str, Any], next_fn: Any) -> Any:
        service = self.service
        agent = payload.get("agent")
        session = getattr(agent, "session", None)
        error = payload.get("error")
        if (session is not None and isinstance(error, ContextOverflow)
                and service.overflow_used < service.max_overflow_retries):
            service.overflow_used += 1
            result = await service.condense(session, config=payload.get("config") or {},
                                            force=True)
            if result.get("condensed"):
                return {"action": "retry", "reason": "context overflow condensed",
                        "compaction": result}
        return await next_fn(payload)
