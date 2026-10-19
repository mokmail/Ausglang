"""DeepSeek Harness — plugin kernel port.

Cordis' contract in miniature: plugins contribute services, typed events and
reversible effects to a shared context. Nothing is privileged, everything a
plugin registers unwinds when its mount is disposed, and a composition is just
an ordered list of rows.

Port notes
----------
* `Context.provide` is Cordis' service registration; `Context.effect` is a
  reversible effect. Both return disposers and both are undone LIFO on
  `dispose()`, like the JS runtime.
* Events come in two kinds, matching the harness: *serial* listeners observe a
  payload in order, *waterfall* listeners must call `next(value)` to delegate
  and may end the chain by returning without delegating (reject).
* `PluginHost` mounts rows and honours `Plugin.inject`: a plugin whose services
  are missing stays pending and activates when the service appears, exactly the
  deferred-injection behaviour the agent loop relies on at boot.
"""

from __future__ import annotations

import asyncio
import inspect
from collections import defaultdict
from typing import Any, Awaitable, Callable

__all__ = ["Context", "Plugin", "PluginHost", "Row", "ServiceMissing"]


class ServiceMissing(RuntimeError):
    """Raised when a plugin or consumer asks for a service that is not mounted."""


async def _await_if_needed(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


class Context:
    """A scoped plugin context: services, listeners and effects."""

    def __init__(self, parent: "Context | None" = None, name: str = "root") -> None:
        self.parent = parent
        self.name = name
        self.disposed = False
        self._services: dict[str, Any] = {}
        self._listeners: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._effects: list[tuple[str, Callable[[], Any]]] = []
        self._pending: list[dict[str, Any]] = []
        self._children: list["Context"] = []
        if parent is not None:
            parent._children.append(self)

    # -- services ---------------------------------------------------------- #
    def get(self, key: str, default: Any = None) -> Any:
        ctx: Context | None = self
        while ctx is not None:
            if key in ctx._services:
                return ctx._services[key]
            ctx = ctx.parent
        return default

    def require(self, key: str) -> Any:
        value = self.get(key)
        if value is None:
            raise ServiceMissing(f"service {key!r} is not mounted in this composition")
        return value

    def has(self, *keys: str) -> bool:
        return all(self.get(k) is not None for k in keys)

    def root(self) -> "Context":
        ctx: Context = self
        while ctx.parent is not None:
            ctx = ctx.parent
        return ctx

    def provide(self, key: str, value: Any, *, local: bool = False) -> Callable[[], None]:
        """Register a service. Services are host-wide by default (visible to
        every mounted plugin), matching how the harness treats `ctx.<service>`;
        `local=True` keeps one in this scope for a shadowing registration."""
        target = self if local else self.root()
        if key in target._services:
            raise RuntimeError(f"service {key!r} already provided in this composition")
        target._services[key] = value

        def dispose() -> None:
            if target._services.get(key) is value:
                target._services.pop(key, None)

        target._effects.append((key, dispose))
        target._flush_tree()
        return dispose

    def _flush_tree(self) -> None:
        self._flush_pending()
        for child in list(self._children):
            child._flush_tree()

    # -- effects ----------------------------------------------------------- #
    def effect(self, dispose: Callable[[], Any]) -> Callable[[], None]:
        """Register a reversible effect; it unwinds when the scope disposes."""
        self._effects.append(("effect", dispose))

        def undo() -> None:
            _run_quietly(dispose)

        return undo

    # -- events ------------------------------------------------------------ #
    def on(
        self,
        event: str,
        listener: Callable[..., Any],
        *,
        kind: str = "serial",
        order: float = 0,
        name: str = "",
        local: bool = False,
    ) -> Callable[[], None]:
        """Subscribe to an event. Listeners are composition-wide by default so a
        plugin mounted in one scope can serve every other; `local=True` keeps
        the subscription in this scope for a shadowing override."""
        if kind not in ("serial", "waterfall"):
            raise ValueError(f"unknown listener kind {kind!r}")
        entry = {
            "fn": listener,
            "kind": kind,
            "order": order,
            "name": name or getattr(listener, "__name__", "listener"),
            "scope": self.name,
        }
        target = self if local else self.root()
        target._listeners[event].append(entry)

        def dispose() -> None:
            lst = target._listeners.get(event) or []
            if entry in lst:
                lst.remove(entry)

        self._effects.append((f"on:{event}", dispose))
        return dispose

    def _visible_listeners(self, event: str) -> list[dict[str, Any]]:
        """Listeners of this scope plus ancestors, shadowed by (name) in child scopes."""
        chain: list[Context] = []
        ctx: Context | None = self
        while ctx is not None:
            chain.append(ctx)
            ctx = ctx.parent
        seen: set[tuple[str, str]] = set()
        picked: list[dict[str, Any]] = []
        for scope in chain:  # nearest scope first
            local = sorted(scope._listeners.get(event, []), key=lambda e: (e["order"], e["name"]))
            for entry in local:
                key = (entry["kind"], entry["name"])
                if key in seen:
                    continue  # a scoped listener shadows the global one of the same name
                seen.add(key)
                picked.append(entry)
        picked.sort(key=lambda e: (e["order"], e["name"]))
        return picked

    def emit(self, event: str, payload: Any) -> list[Any]:
        """Serial notification; async listeners are scheduled, not awaited."""
        out = []
        for entry in self._visible_listeners(event):
            if entry["kind"] != "serial":
                continue
            result = entry["fn"](payload)
            if inspect.isawaitable(result):
                result = _fire(result)
            out.append(result)
        return out

    async def emit_async(self, event: str, payload: Any) -> list[Any]:
        out = []
        for entry in self._visible_listeners(event):
            if entry["kind"] != "serial":
                continue
            out.append(await _await_if_needed(entry["fn"](payload)))
        return out

    async def waterfall(self, event: str, payload: Any) -> Any:
        """Cooperative chain. Listeners delegate with `next(value)`.

        A listener that returns without delegating ends the chain and its return
        value becomes the result (this is how `agent/pre-step` rejects a step).
        A listener that ignores the payload entirely delegates unchanged.
        """
        listeners = self._visible_listeners(event)

        async def step(index: int, value: Any) -> Any:
            if index >= len(listeners):
                return value
            entry = listeners[index]
            state: dict[str, Any] = {"called": False, "next_result": None}

            async def nxt(next_value: Any = None) -> Any:
                state["called"] = True
                state["next_result"] = await step(
                    index + 1, value if next_value is None else next_value
                )
                return state["next_result"]

            result = await _await_if_needed(entry["fn"](value, nxt))
            if state["called"]:
                return state["next_result"] if result is None else result
            return result if result is not None else value

        return await step(0, payload)

    # -- scopes ------------------------------------------------------------ #
    def scope(self, name: str) -> "Context":
        return Context(parent=self, name=name)

    def dispose(self) -> None:
        if self.disposed:
            return
        self.disposed = True
        for child in reversed(list(self._children)):
            child.dispose()
        self._children.clear()
        while self._effects:
            _, fn = self._effects.pop()
            _run_quietly(fn)
        self._services.clear()
        self._listeners.clear()
        self._pending.clear()
        if self.parent is not None and self in self.parent._children:
            self.parent._children.remove(self)

    # -- deferred injection ------------------------------------------------ #
    def inject(self, services: tuple[str, ...] | list[str], callback: Callable[[], Any],
               *, name: str = "") -> Callable[[], None]:
        """Run `callback` once every service is available (Cordis `ctx.inject`)."""
        services = tuple(s for s in services if s)
        entry = {"services": services, "callback": callback, "name": name or "inject"}
        if self.has(*services):
            return _run_quietly(callback)
        self._pending.append(entry)
        self._flush_pending()

        def cancel() -> None:
            if entry in self._pending:
                self._pending.remove(entry)

        return cancel

    def _flush_pending(self) -> None:
        if not self._pending:
            return
        ready = [e for e in self._pending if self.has(*e["services"])]
        for entry in ready:
            self._pending.remove(entry)
            try:
                entry["callback"]()
            except Exception:  # a failed activation must not poison the others
                continue


def _run_quietly(fn: Callable[[], Any]) -> Any:
    try:
        return fn()
    except Exception:
        return None


def _fire(awaitable: Awaitable[Any]) -> Any:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return None  # no loop: drop the notification rather than deadlock
    return loop.create_task(_await_if_needed(awaitable))


class Plugin:
    """Base class for every harness plugin."""

    name: str = "plugin"
    #: services this plugin needs before it may apply
    inject: tuple[str, ...] = ()

    def __init__(self, ctx: Context, config: dict[str, Any] | None = None) -> None:
        self.ctx = ctx
        self.config = dict(config or {})

    # -- lifecycle --------------------------------------------------------- #
    def apply(self) -> None:  # pragma: no cover - overridden by subclasses
        raise NotImplementedError

    def dispose(self) -> None:
        """Optional extra teardown; ctx effects unwind automatically."""

    # -- helpers ----------------------------------------------------------- #
    def provide(self, key: str, value: Any) -> None:
        self.ctx.provide(key, value)

    def on(self, event: str, fn: Callable[..., Any], **kw: Any) -> None:
        self.ctx.on(event, fn, **kw)

    def section(self, name: str, order: float, text: Any) -> None:
        self.ctx.require("systemPrompt").section(name, order, text)


Row = dict[str, Any]


class PluginHost:
    """Mounts composition rows into one root context."""

    def __init__(self, registry: dict[str, type[Plugin]] | None = None,
                 *, name: str = "harness") -> None:
        self.registry: dict[str, type[Plugin]] = dict(registry or {})
        self.root = Context(name=name)
        self.rows: list[Row] = []
        self.mounted: dict[str, dict[str, Any]] = {}
        self.failed: list[dict[str, str]] = []
        self.warnings: list[str] = []

    def register(self, cls: type[Plugin]) -> None:
        self.registry[cls.name] = cls

    def mount(self, rows: list[Row]) -> "PluginHost":
        for row in rows:
            self.mount_row(row)
        return self

    def mount_row(self, row: Row) -> None:
        rid = str(row.get("id") or row.get("name"))
        name = str(row.get("name") or "")
        if row.get("enabled", True) is False:
            self.rows.append({**row, "id": rid, "state": "disabled"})
            return
        cls = self.registry.get(name)
        if cls is None:
            self.failed.append({"id": rid, "name": name, "error": "plugin not registered"})
            self.rows.append({**row, "id": rid, "state": "missing"})
            return
        ctx = self.root.scope(f"{rid}:{name}")
        plugin = cls(ctx, row.get("config") or {})
        entry = {"id": rid, "name": name, "ctx": ctx, "plugin": plugin, "state": "mounted"}
        self.rows.append({**row, "id": rid, "state": "mounted"})
        self.mounted[rid] = entry

        def apply_now() -> None:
            if ctx.disposed:
                return
            try:
                plugin.apply()
            except Exception as exc:  # keep the rest of the tree alive, report loudly
                entry["state"] = "failed"
                self.failed.append({"id": rid, "name": name,
                                    "error": f"{type(exc).__name__}: {exc}"})
                ctx.dispose()
            else:
                entry["state"] = "mounted"

        if cls.inject and not self.root.has(*cls.inject):
            ctx.inject(cls.inject, apply_now, name=f"{rid}:apply")
            entry["state"] = "pending"
        else:
            apply_now()

    def unmount(self, rid: str) -> bool:
        entry = self.mounted.pop(rid, None)
        if not entry:
            return False
        entry["ctx"].dispose()
        for row in self.rows:
            if row.get("id") == rid:
                row["state"] = "disabled"
        return True

    def dispose(self) -> None:
        self.root.dispose()
        self.mounted.clear()

    # -- introspection ----------------------------------------------------- #
    def services(self) -> list[str]:
        found: set[str] = set()

        def walk(ctx: Context) -> None:
            found.update(ctx._services)
            for child in ctx._children:
                walk(child)

        walk(self.root)
        return sorted(found)

    def listeners(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []

        def walk(ctx: Context) -> None:
            for event, entries in ctx._listeners.items():
                for e in entries:
                    out.append({"event": event, "kind": e["kind"], "name": e["name"],
                                "order": e["order"], "scope": ctx.name})
            for child in ctx._children:
                walk(child)

        walk(self.root)
        return out

    def dump(self) -> dict[str, Any]:
        return {
            "rows": [
                {k: v for k, v in row.items() if k in ("id", "name", "config", "state")}
                for row in self.rows
            ],
            "services": self.services(),
            "listeners": self.listeners(),
            "failed": self.failed,
        }
