"""The scoped tool registry and the guarded execution pipeline.

Every call runs the same fixed pipeline as the harness:

    tools/pre-execute  -> registered guards -> tools/execute
    -> tools/post-execute -> tools/result

`pre-execute` is a waterfall that may deny with a reason; guards are monotonic,
so a denial can never be turned back into permission; `execute` wraps dispatch
for retry/timeout; `post-execute` inspects or replaces the result. Exclusive
tools are ordering barriers, parallel-safe tools share a bounded pool.
"""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, field
from typing import Any, Callable

__all__ = ["Tool", "ToolRuntime", "ToolResult", "ExecContext", "ToolsPlugin",
           "validate_arguments"]


def _maybe_await(value: Any) -> Any:
    return value


async def _await_if_needed(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    execute: Callable[[dict[str, Any], "ExecContext"], Any]
    server: str = "local"
    exclusive: bool = False
    output: Any = None
    render: Callable[[dict[str, Any], Any], str] | None = None
    scope: str = "global"

    def schema(self) -> dict[str, Any]:
        return {"name": self.name, "server": self.server,
                "description": (self.description or "")[:600],
                "input_schema": self.parameters or {"type": "object", "properties": {}}}


@dataclass
class ToolResult:
    tool: str
    status: str                      # ok | error | denied | aborted
    content: str = ""
    value: Any = None
    error: str = ""
    server: str = "local"

    def as_event(self) -> dict[str, Any]:
        return {"tool": self.tool, "server": self.server, "state":
                "done" if self.status == "ok" else "error",
                "result": self.content, "error": self.error}


@dataclass
class ExecContext:
    """Everything a tool body may use, plus the caller-owned cancellation signal."""

    session: Any = None
    agent: Any = None
    signal: asyncio.Event = field(default_factory=asyncio.Event)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def cancelled(self) -> bool:
        return self.signal.is_set()


class ToolRuntime:
    """ctx.tools — the one registry tools register into and the loop dispatches through."""

    def __init__(self, ctx: Any, config: dict[str, Any] | None = None) -> None:
        self.ctx = ctx
        self.config = dict(config or {})
        self._tools: dict[str, Tool] = {}
        self._scoped: dict[str, dict[str, Tool]] = {}
        self._restrictions: dict[str, dict[str, Any]] = {}
        self._guards: list[Callable[[Tool, dict[str, Any]], Any]] = []

    # -- registration ------------------------------------------------------ #
    def register(self, tool: Tool, *, scope: str = "global") -> Callable[[], None]:
        tool.scope = scope
        target = self._tools if scope == "global" else self._scoped.setdefault(scope, {})
        if tool.name in target:
            raise ValueError(f"tool {tool.name!r} already registered in this scope")
        target[tool.name] = tool
        self.ctx.emit("tools/change", {"tool": tool.name, "action": "register"})

        def dispose() -> None:
            if target.get(tool.name) is tool:
                target.pop(tool.name, None)

        return dispose

    def restrict(self, *, allow: list[str] | None = None,
                 deny: list[str] | None = None, scope: str = "global") -> Callable[[], None]:
        rule = {"allow": [a for a in (allow or []) if a], "deny": [d for d in (deny or []) if d]}
        self._restrictions[scope] = rule

        def dispose() -> None:
            if self._restrictions.get(scope) is rule:
                self._restrictions.pop(scope, None)

        return dispose

    def guard(self, fn: Callable[[Tool, dict[str, Any]], Any]) -> Callable[[], None]:
        """A monotonic synchronous guard: a reason denies, and stays denied."""
        self._guards.append(fn)

        def dispose() -> None:
            if fn in self._guards:
                self._guards.remove(fn)

        return dispose

    # -- reads -------------------------------------------------------------- #
    def _all(self, scope: str = "global") -> dict[str, Tool]:
        merged = dict(self._tools)
        merged.update(self._scoped.get(scope) or {})
        rule = self._restrictions.get(scope) or self._restrictions.get("global") or {}
        allow, deny = rule.get("allow") or [], rule.get("deny") or []
        out = {}
        for name, tool in merged.items():
            if allow and name not in allow and tool.server not in allow:
                continue
            if name in deny or tool.server in deny:
                continue
            out[name] = tool
        return out

    def get(self, name: str, scope: str = "global") -> Tool | None:
        return self._all(scope).get(name)

    def schemas(self, scope: str = "global") -> list[dict[str, Any]]:
        return [t.schema() for t in self._all(scope).values()]

    def names(self, scope: str = "global") -> list[str]:
        return list(self._all(scope))

    # -- execution ---------------------------------------------------------- #
    async def run(self, name: str, arguments: dict[str, Any], exec_ctx: ExecContext,
                  scope: str = "global") -> ToolResult:
        tool = self.get(name, scope)
        if tool is None:
            return ToolResult(tool=name, status="error", error="UNKNOWN_TOOL",
                              content=f"error: unknown tool {name!r}")

        problem = validate_arguments(tool.parameters, arguments)
        if problem:
            result = ToolResult(tool=name, status="error", error="INVALID_ARGUMENTS",
                                content=f"error: {problem}", server=tool.server)
            self.ctx.emit("tools/result", {"tool": name, "status": result.status})
            return result

        # 1. extensible pre-execute decision
        decision = await self.ctx.waterfall(
            "tools/pre-execute",
            {"tool": tool, "name": name, "arguments": arguments, "exec": exec_ctx,
             "decision": "allow", "reason": ""},
        )
        if not isinstance(decision, dict):
            decision = {"decision": "allow"}
        if decision.get("decision") == "deny":
            result = ToolResult(tool=name, status="denied",
                                error=decision.get("reason") or "DENIED",
                                content=f"error: {decision.get('reason') or 'denied'}",
                                server=tool.server)
            self.ctx.emit("tools/result", {"tool": name, "status": result.status})
            return result

        # 2. monotonic guards: no later listener can undo a denial
        for guard in list(self._guards):
            try:
                reason = await _await_if_needed(guard(tool, arguments))
            except Exception as exc:
                reason = f"{type(exc).__name__}: {exc}"
            if reason:
                result = ToolResult(tool=name, status="denied", error=str(reason),
                                    content=f"error: {reason}", server=tool.server)
                self.ctx.emit("tools/result", {"tool": name, "status": result.status})
                return result

        # 3. execute, wrapped by the extensible `tools/execute` waterfall
        async def dispatch(payload: Any = None) -> ToolResult:
            if exec_ctx.cancelled:
                return ToolResult(tool=name, status="aborted", error="ABORTED",
                                  content="error: aborted", server=tool.server)
            try:
                value = await _await_if_needed(tool.execute(arguments, exec_ctx))
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                return ToolResult(tool=name, status="error", server=tool.server,
                                  error=f"{type(exc).__name__}: {exc}",
                                  content=f"error: {type(exc).__name__}: {exc}")
            content = tool.render(arguments, value) if tool.render else _stringify(value)
            return ToolResult(tool=name, status="ok", value=value,
                              content=content, server=tool.server)

        executed = await self.ctx.waterfall(
            "tools/execute",
            {"tool": tool, "name": name, "arguments": arguments, "run": dispatch},
        )
        result = executed if isinstance(executed, ToolResult) else await dispatch()

        # 4. post-execute may inspect or replace the result
        posted = await self.ctx.waterfall("tools/post-execute", result)
        if isinstance(posted, ToolResult):
            result = posted

        # 5. observe only
        self.ctx.emit("tools/result", {"tool": name, "status": result.status,
                                       "content": result.content[:500]})
        return result

    async def run_many(self, calls: list[dict[str, Any]], exec_ctx: ExecContext,
                       *, max_parallel: int | None = None,
                       scope: str = "global") -> list[ToolResult]:
        """Exclusive calls form barriers; parallel-safe calls share a bounded pool."""
        limit = max(1, int(max_parallel or self.config.get("maxParallelToolCalls") or 4))
        sem = asyncio.Semaphore(limit)
        results: list[ToolResult | None] = [None] * len(calls)

        async def one(index: int, call: dict[str, Any]) -> None:
            async with sem:
                results[index] = await self.run(
                    str(call.get("tool") or ""), call.get("arguments") or {},
                    exec_ctx, scope,
                )

        pending: list[asyncio.Task] = []
        for index, call in enumerate(calls):
            tool = self.get(str(call.get("tool") or ""), scope)
            exclusive = tool.exclusive if tool else True
            if exclusive:
                if pending:
                    await asyncio.gather(*pending)
                    pending = []
                results[index] = await self.run(
                    str(call.get("tool") or ""), call.get("arguments") or {}, exec_ctx, scope)
            else:
                pending.append(asyncio.create_task(one(index, call)))
        if pending:
            await asyncio.gather(*pending)
        return [r for r in results if r is not None]


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    import json
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)


def validate_arguments(schema: dict[str, Any], args: dict[str, Any]) -> str | None:
    """The JSON-Schema subset the harness enforces before dispatch."""
    if not isinstance(schema, dict):
        return None
    if not isinstance(args, dict):
        return "arguments must be an object"
    properties = schema.get("properties") or {}
    for key in schema.get("required") or []:
        if key not in args:
            return f"missing required argument {key!r}"
    types = {"string": str, "number": (int, float), "integer": int,
             "boolean": bool, "object": dict, "array": list}
    for key, value in args.items():
        spec = properties.get(key)
        if not isinstance(spec, dict):
            continue
        want = spec.get("type")
        if want in types:
            expected = types[want]
            if want in ("number", "integer") and isinstance(value, bool):
                return f"argument {key!r} must be {want}"
            if not isinstance(value, expected):
                return f"argument {key!r} must be {want}"
        enum = spec.get("enum")
        if enum and value not in enum:
            return f"argument {key!r} must be one of {enum}"
    return None


class ToolsPlugin:
    name = "tools"
    inject: tuple[str, ...] = ()

    def __init__(self, ctx: Any, config: dict[str, Any] | None = None) -> None:
        self.ctx = ctx
        self.config = dict(config or {})
        self.tools = ToolRuntime(ctx, self.config)
        ctx.provide("tools", self.tools)

    def apply(self) -> None:
        rule = self.config.get("restrict") or {}
        if rule.get("allow") or rule.get("deny"):
            self.tools.restrict(allow=rule.get("allow") or [], deny=rule.get("deny") or [])
