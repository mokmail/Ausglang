"""Harness components mounted as plugins.

Each editable component of the workbench harness becomes a plugin row, so the
whole harness is a composition: c1 role, c2 guardrails, c3 memory, c4 tool
guidance, c5 subagents, c6 skills, c7 scripts, c8 behavioural hooks — beside the
infrastructure rows (session, prompt, tools, llm, loop, compaction).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from .kernel import Plugin
from .session import Session, SessionStore
from .tools import ExecContext, Tool

__all__ = [
    "TOOL_PROTOCOL", "SessionPlugin", "ChatSurfacePlugin", "RolePlugin",
    "GuardrailsPlugin", "MemoryPlugin", "LessonsPlugin", "ToolGuidancePlugin",
    "SkillsPlugin", "ScriptsPlugin", "HooksPlugin", "SubagentsPlugin",
    "ArtifactsPlugin", "McpToolsPlugin", "RuntimeContextPlugin", "HARNESS_PLUGINS",
]

TOOL_PROTOCOL = """\
You can use tools. Available tools are listed below as JSON schemas.

To call a tool, reply with ONLY a fenced json block in this exact shape:
```json
{"tool": "<tool_name>", "arguments": { ... }}
```
You may emit several such blocks to call several tools at once. Wait for the
observation before writing the final answer. Never invent observations.
Do not wrap the JSON in prose. If no tool is needed, answer normally."""


# --------------------------------------------------------------------------- #
# infrastructure rows
# --------------------------------------------------------------------------- #


class SessionPlugin(Plugin):
    """`session` row: the append-only log and the store new sessions come from."""

    name = "session"

    def apply(self) -> None:
        root = self.config.get("root")
        persist = bool(self.config.get("persist", True))
        store = SessionStore(Path(root) if root else None)
        self.provide("agentSessionStore", store)
        self.ctx.effect(store.sessions.clear)

    @staticmethod
    def for_chat(store: SessionStore, chat_id: str, *, ctx: Any = None) -> Session:
        return store.create(chat_id=chat_id or None, persist=True, ctx=ctx)


class ChatSurfacePlugin(Plugin):
    """`chat-surface` row: what transports this composition exposes."""

    name = "chat-surface"

    def apply(self) -> None:
        self.provide("chatSurface", {
            "transport": self.config.get("transport") or "sse",
            "streaming": True,
            "artifacts": True,
        })


class RuntimeContextPlugin(Plugin):
    """Dynamic runtime facts as a late prompt section (never cached as history)."""

    name = "runtime-context"
    inject = ("systemPrompt",)

    def apply(self) -> None:
        prompt = self.ctx.require("systemPrompt")
        prompt.variable("model", lambda: self.config.get("model") or "")
        prompt.variable("provider", lambda: self.config.get("provider") or "sglang")

        def render() -> str:
            import time

            bits = [f"Current time: {time.strftime('%Y-%m-%d %H:%M %Z')}."]
            if self.config.get("model"):
                bits.append(f"Model: {self.config['model']}.")
            if self.config.get("workspace"):
                bits.append(f"Workspace: {self.config['workspace']}.")
            return " ".join(bits)

        prompt.section("runtime:context", 9000, render)


# --------------------------------------------------------------------------- #
# harness components c1..c8
# --------------------------------------------------------------------------- #


class RolePlugin(Plugin):
    """c1 — the base instruction set."""

    name = "harness-role"
    inject = ("systemPrompt",)

    def apply(self) -> None:
        text = (self.config.get("system_prompt") or "").strip()
        if text:
            self.ctx.require("systemPrompt").section("harness:role", 10, text)


class GuardrailsPlugin(Plugin):
    """c2 — rules in the prompt, machine checks on the way in."""

    name = "guardrails"
    inject = ("systemPrompt",)

    def apply(self) -> None:
        import re

        prompt = self.ctx.require("systemPrompt")
        text = (self.config.get("guardrails") or "").strip()
        rules = self.config.get("rules") or {}
        if text:
            prompt.section("harness:guardrails", 20,
                           "Guardrails (must hold):\n" + text)
        blocked = [p for p in (rules.get("blocked_input") or []) if p]

        async def pre_step(payload: dict[str, Any], next_fn: Any) -> Any:
            if not blocked:
                return await next_fn(payload)
            blob = "\n".join(str(m.get("content") or "") for m in payload.get("messages") or [])
            for pattern in blocked:
                try:
                    if re.search(pattern, blob, re.I):
                        return {"decision": "reject",
                                "reason": f"input matches blocked pattern /{pattern}/"}
                except re.error:
                    continue
            return await next_fn(payload)

        self.ctx.on("agent/pre-step", pre_step, kind="waterfall", order=-100)


class MemoryPlugin(Plugin):
    """c3 — durable context: preamble, stored facts, injected notes."""

    name = "memory"
    inject = ("systemPrompt",)

    def apply(self) -> None:
        prompt = self.ctx.require("systemPrompt")
        preamble = (self.config.get("memory") or "").strip()
        facts = (self.config.get("memory_text") or "").strip()
        parts = []
        if preamble:
            parts.append(preamble)
        if facts:
            parts.append("Known facts about the user (use when relevant):\n" + facts)
        if self.config.get("json_mode"):
            parts.append("Answer with a single valid JSON object and nothing else.")
        if parts:
            prompt.section("context:memory", 30, "\n\n".join(parts))


class LessonsPlugin(Plugin):
    """c3 — the experience library, injected when it applies."""

    name = "lessons"
    inject = ("systemPrompt",)

    def apply(self) -> None:
        rows = [r for r in (self.config.get("rows") or []) if r.get("lesson")]
        if not rows:
            return
        body = "\n".join(f"- ({r.get('domain') or 'general'}) {r['lesson']}" for r in rows)
        self.ctx.require("systemPrompt").section(
            "context:lessons", 40, "Lessons from earlier sessions in this domain:\n" + body)


class ToolGuidancePlugin(Plugin):
    """c4 — per-tool usage notes folded into the schemas the model sees."""

    name = "harness-tool-guidance"
    inject = ("systemPrompt",)

    def apply(self) -> None:
        guidance = {str(k): str(v) for k, v in (self.config.get("guidance") or {}).items()}
        if not guidance:
            return

        async def fold(assembly: Any, next_fn: Any) -> Any:
            for tool in getattr(assembly, "tools", []):
                note = guidance.get(str(tool.get("name") or ""))
                if note and note.strip():
                    base = (tool.get("description") or "").strip()
                    tool["description"] = f"{base}\nNote: {note.strip()}".strip()
            return await next_fn(assembly)

        self.ctx.on("system-prompt/assemble", fold, kind="waterfall", order=50)


class SkillsPlugin(Plugin):
    """c6 — a catalog in the prompt and a loader tool for the full body."""

    name = "skills"
    inject = ("systemPrompt", "tools")

    def apply(self) -> None:
        skills = [s for s in (self.config.get("skills") or []) if s.get("name")]
        if not skills:
            return
        prompt = self.ctx.require("systemPrompt")
        tools = self.ctx.require("tools")
        bodies = {str(s["name"]): str(s.get("body") or "") for s in skills}
        lines = [f"- `{s['name']}` — {(s.get('description') or s.get('body') or '')[:120]}"
                 for s in skills]
        prompt.section("harness:skills", 50,
                       "Skills you can load with the `skill` tool:\n" + "\n".join(lines))

        async def load(args: dict[str, Any], exec_ctx: ExecContext) -> str:
            name = str(args.get("name") or "")
            body = bodies.get(name)
            if body is None:
                return f"no skill named {name!r}; available: {', '.join(sorted(bodies))}"
            return body

        tools.register(Tool(
            name="skill",
            description="Load the full instructions of one skill from the catalog.",
            parameters={"type": "object",
                        "properties": {"name": {"type": "string",
                                                "enum": sorted(bodies)}},
                        "required": ["name"]},
            execute=load,
            server="harness",
        ))


class ScriptsPlugin(Plugin):
    """c7 — named procedures that are always available."""

    name = "harness-scripts"
    inject = ("systemPrompt",)

    def apply(self) -> None:
        scripts = [s for s in (self.config.get("scripts") or [])
                   if (s.get("body") or "").strip()]
        if not scripts:
            return
        body = "\n\n".join(f"### script: {s.get('name', 'script')}\n{s['body'].strip()}"
                           for s in scripts)
        self.ctx.require("systemPrompt").section(
            "harness:scripts", 60, "Scripts (named procedures you can follow):\n" + body)


class HooksPlugin(Plugin):
    """c8 — behavioural hooks applied to every turn."""

    name = "harness-plugins"
    inject = ("systemPrompt",)

    def apply(self) -> None:
        hooks = [p for p in (self.config.get("plugins") or [])
                 if (p.get("body") or "").strip()]
        if not hooks:
            return
        body = "\n".join(f"- {p.get('name', 'hook')}: {p['body'].strip()}" for p in hooks)
        self.ctx.require("systemPrompt").section(
            "harness:hooks", 80, "Behavioural hooks applied to every turn:\n" + body)


class SubagentsPlugin(Plugin):
    """c5 — named delegates, each callable as a tool, plus a generic `subagent`."""

    name = "subagents"
    inject = ("tools", "agentLoop", "systemPrompt")

    def apply(self) -> None:
        subs = [s for s in (self.config.get("subagents") or []) if s.get("name")]
        if not subs:
            return
        tools = self.ctx.require("tools")
        prompt = self.ctx.require("systemPrompt")
        names = [str(s["name"]) for s in subs]
        prompt.section(
            "harness:subagents", 70,
            "Subagents you can delegate to:\n"
            + "\n".join(f"- `{s['name']}` — {s.get('description') or 'delegate a subtask'}"
                        for s in subs)
            + "\n\nCall one by name with {\"task\": ...}, or use the `subagent` tool "
              "with {\"name\": ..., \"task\": ...}.",
        )

        runner: Callable[..., Any] | None = self.config.get("runner")
        max_depth = int(self.config.get("maxDepth") or 1)

        async def run_one(name: str, task: str, exec_ctx: ExecContext) -> str:
            depth = int((exec_ctx.meta or {}).get("depth") or 0)
            if depth >= max_depth:
                return f"error: delegation depth {depth} reached the limit ({max_depth})"
            if runner is None:
                return "error: no subagent runner configured in this composition"
            parent = exec_ctx.agent
            return await runner(
                name=name, task=task, parent=parent,
                system=self.config.get("child_system") or "",
                meta={**(exec_ctx.meta or {}), "depth": depth + 1},
            )

        for sub in subs:
            async def call(args: dict[str, Any], exec_ctx: ExecContext,
                           _name=str(sub["name"])) -> str:
                return await run_one(_name, str(args.get("task") or ""), exec_ctx)

            tools.register(Tool(
                name=str(sub["name"]),
                description=sub.get("description") or f"Delegate a subtask to {sub['name']}.",
                parameters={"type": "object",
                            "properties": {"task": {"type": "string"}},
                            "required": ["task"]},
                execute=call,
                server="harness",
            ))

        async def generic(args: dict[str, Any], exec_ctx: ExecContext) -> str:
            return await run_one(str(args.get("name") or ""), str(args.get("task") or ""),
                                 exec_ctx)

        tools.register(Tool(
            name="subagent",
            description="Delegate a subtask to a named subagent and return its answer.",
            parameters={"type": "object",
                        "properties": {"name": {"type": "string", "enum": names},
                                       "task": {"type": "string"}},
                        "required": ["name", "task"]},
            execute=generic,
            server="harness",
        ))


class ArtifactsPlugin(Plugin):
    """Native file tools over the chat workspace (write/read/list)."""

    name = "artifacts"
    inject = ("tools",)

    def apply(self) -> None:
        tools = self.ctx.require("tools")
        write = self.config.get("write")
        read = self.config.get("read")
        list_files = self.config.get("list")
        if not (write and read and list_files):
            return

        async def do_write(args: dict[str, Any], exec_ctx: ExecContext) -> Any:
            res = write(str(args.get("name") or ""), str(args.get("content") or ""))
            if isinstance(res, dict) and res.get("artifact"):
                exec_ctx.meta.setdefault("artifacts", []).append(res["artifact"])
            return res.get("name") if isinstance(res, dict) else str(res)

        async def do_read(args: dict[str, Any], exec_ctx: ExecContext) -> str:
            return str(read(str(args.get("name") or "")))

        async def do_list(args: dict[str, Any], exec_ctx: ExecContext) -> str:
            return json.dumps(list_files(), ensure_ascii=False)

        tools.register(Tool(
            name="write_file",
            description="Create or overwrite a file in this chat's workspace. "
                        "Use for code, notes, data, any deliverable.",
            parameters={"type": "object",
                        "properties": {"name": {"type": "string"},
                                       "content": {"type": "string"}},
                        "required": ["name", "content"]},
            execute=do_write, server="artifacts"))
        tools.register(Tool(
            name="read_file",
            description="Read back a file from this chat's workspace.",
            parameters={"type": "object",
                        "properties": {"name": {"type": "string"}},
                        "required": ["name"]},
            execute=do_read, server="artifacts"))
        tools.register(Tool(
            name="list_files",
            description="List the files already written in this chat.",
            parameters={"type": "object", "properties": {}},
            execute=do_list, server="artifacts"))


class McpToolsPlugin(Plugin):
    """Proxy tools from stdio MCP servers into the registry."""

    name = "mcp-tools"
    inject = ("tools",)

    def apply(self) -> None:
        tools = self.ctx.require("tools")
        call = self.config.get("call")
        schemas = self.config.get("schemas") or []
        if not call:
            return
        for schema in schemas:
            name = str(schema.get("name") or "")
            if not name:
                continue
            server = str(schema.get("server") or "")

            async def invoke(args: dict[str, Any], exec_ctx: ExecContext,
                             _name=name, _server=server) -> Any:
                return await call(_server, _name, args)

            tools.register(Tool(
                name=name,
                description=schema.get("description") or f"{name} ({server})",
                parameters=schema.get("input_schema") or {"type": "object", "properties": {}},
                execute=invoke,
                server=server or "mcp",
                exclusive=bool(schema.get("exclusive")),
            ))


class ToolProtocolPlugin(Plugin):
    """The JSON tool protocol and the visible schemas, as one prompt section."""

    name = "harness-tool-protocol"
    inject = ("systemPrompt", "tools")

    def apply(self) -> None:
        prompt = self.ctx.require("systemPrompt")
        tools = self.ctx.require("tools")
        scope = str(self.config.get("scope") or "agent")

        def render() -> str:
            schemas = tools.schemas(scope)
            if not schemas:
                return ""
            return TOOL_PROTOCOL + "\n\nTools:\n" + json.dumps(schemas, ensure_ascii=False)

        prompt.section("tools:protocol", 200, render)


HARNESS_PLUGINS = [
    SessionPlugin, ChatSurfacePlugin, RuntimeContextPlugin, ToolProtocolPlugin,
    RolePlugin, GuardrailsPlugin, MemoryPlugin, LessonsPlugin,
    ToolGuidancePlugin, SkillsPlugin, ScriptsPlugin, HooksPlugin,
    SubagentsPlugin, ArtifactsPlugin, McpToolsPlugin,
]
