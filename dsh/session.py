"""The append-only session log and the projections derived from it.

Model-visible means logged: every request is reconstructable from the events
below, and `derive_messages()` is the one projection that turns those facts into
model history. Tool calls and their observations are durable events, not a side
channel; compaction is a durable replacement marker, so replay produces exactly
the condensed history the model saw.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

__all__ = ["Session", "SessionStore", "Projections", "UsageProjection",
           "TurnBoundaryProjection"]


def _now() -> float:
    return time.time()


class Projections:
    """Registered units that fold committed events incrementally."""

    def __init__(self) -> None:
        self._units: dict[str, Any] = {}

    def register(self, name: str, unit: Any) -> None:
        self._units[name] = unit

    def state_of(self, name: str) -> Any:
        unit = self._units.get(name)
        return getattr(unit, "state", None)

    def fold(self, event: dict[str, Any]) -> None:
        for unit in self._units.values():
            try:
                unit.fold(event)
            except Exception:
                continue

    def snapshot(self) -> dict[str, Any]:
        return {name: getattr(u, "state", None) for name, u in self._units.items()}


class UsageProjection:
    def __init__(self) -> None:
        self.state = {"steps": 0, "turns": 0, "tool_calls": 0, "prompt_tokens": 0,
                      "completion_tokens": 0, "compactions": 0, "errors": 0}

    def fold(self, event: dict[str, Any]) -> None:
        t = event.get("type")
        if t == "turn/start":
            self.state["turns"] += 1
        elif t == "step/start":
            self.state["steps"] += 1
        elif t == "tool/call":
            self.state["tool_calls"] += 1
        elif t == "assistant/message":
            usage = event.get("usage") or {}
            self.state["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
            self.state["completion_tokens"] += int(usage.get("completion_tokens") or 0)
        elif t == "compaction/replace":
            self.state["compactions"] += 1
        elif t in ("agent/error", "assistant/attempt"):
            if t == "agent/error":
                self.state["errors"] += 1


class TurnBoundaryProjection:
    def __init__(self) -> None:
        self.state = {"state": "idle", "turn": 0, "step": 0, "steps_in_turn": 0}

    def fold(self, event: dict[str, Any]) -> None:
        t = event.get("type")
        if t == "turn/start":
            self.state.update(state="running", turn=event.get("turn") or 0, steps_in_turn=0)
        elif t == "step/start":
            self.state["step"] = event.get("step") or 0
            self.state["steps_in_turn"] = int(self.state.get("steps_in_turn") or 0) + 1
        elif t == "turn/end":
            self.state["state"] = "idle"


class Session:
    """One agent's durable log."""

    def __init__(self, id: str | None = None, *, ctx: Any = None,
                 chat_id: str | None = None, persist: bool = False,
                 path: Path | None = None) -> None:
        self.id = id or f"session-{uuid.uuid4().hex[:12]}"
        self.chat_id = chat_id
        self.ctx = ctx
        self.persist = persist
        self.path = path
        self._events: list[dict[str, Any]] = []
        self.seq = 0
        self.projections = Projections()
        self.projections.register("usage", UsageProjection())
        self.projections.register("turn", TurnBoundaryProjection())

    # -- append ------------------------------------------------------------ #
    def append(self, type: str, **fields: Any) -> dict[str, Any]:
        self.seq += 1
        event = {"seq": self.seq, "ts": _now(), "type": type, "session": self.id, **fields}
        self._events.append(event)
        self.projections.fold(event)
        if self.persist and self.path:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(event, ensure_ascii=False) + "\n")
            except OSError:
                self.persist = False
        if self.ctx is not None:
            self.ctx.emit("session/event", event)
        return event

    # -- reads ------------------------------------------------------------- #
    @property
    def events(self) -> list[dict[str, Any]]:
        return list(self._events)

    def of_type(self, *types: str) -> list[dict[str, Any]]:
        wanted = set(types)
        return [e for e in self._events if e.get("type") in wanted]

    def tail(self, n: int = 100) -> list[dict[str, Any]]:
        return self._events[-n:]

    def last_compaction(self) -> dict[str, Any] | None:
        for event in reversed(self._events):
            if event.get("type") == "compaction/replace":
                return event
        return None

    def prune_map(self) -> dict[int, int]:
        """Durable tool-result trims: event seq -> retained characters."""
        out: dict[int, int] = {}
        for event in self._events:
            if event.get("type") == "compaction/prune":
                for key, value in (event.get("targets") or {}).items():
                    try:
                        out[int(key)] = int(value)
                    except (TypeError, ValueError):
                        continue
        return out

    # -- model history ----------------------------------------------------- #
    def derive_messages(self) -> list[dict[str, Any]]:
        """Project the log onto the exact message array the model sees."""
        compaction = self.last_compaction()
        cutoff = int(compaction.get("upto") or 0) if compaction else 0
        events = [e for e in self._events if e["seq"] > cutoff]
        prunes = self.prune_map()

        system: str = ""
        messages: list[dict[str, Any]] = []
        if compaction and (compaction.get("summary") or "").strip():
            messages.append({
                "role": "user",
                "content": "[compact] Earlier conversation, condensed:\n"
                           + str(compaction["summary"]).strip(),
            })

        for event in events:
            t = event.get("type")
            if t == "system/message":
                text = event.get("text") or ""
                system = text  # the latest non-empty system node is the effective prompt
            elif t == "user/message":
                text = event.get("text") or ""
                if text.strip():
                    messages.append({"role": "user", "content": text})
            elif t == "assistant/message":
                text = event.get("text") or ""
                if text.strip():
                    messages.append({"role": "assistant", "content": text})
            elif t == "assistant/attempt":
                text = event.get("text") or ""
                if event.get("interrupted") and text.strip():
                    messages.append({"role": "assistant", "content": text})
            elif t == "tool/result":
                text = event.get("content") or ""
                limit = prunes.get(event["seq"])
                if limit is not None and len(text) > limit:
                    text = text[:limit].rstrip() + "\n… (trimmed)"
                name = event.get("tool") or "tool"
                messages.append({
                    "role": "user",
                    "content": f"Observation from {name}:\n{text}\n\nContinue. "
                               "Use another tool call if needed, otherwise answer.",
                })
        if system.strip():
            messages.insert(0, {"role": "system", "content": system})
        return messages

    def exported(self) -> dict[str, Any]:
        return {"id": self.id, "chat_id": self.chat_id, "seq": self.seq,
                "events": self.events, "projections": self.projections.snapshot()}


class SessionStore:
    """Session acquisition: memory-only by default, JSONL when persistent."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root else None
        self.sessions: dict[str, Session] = {}

    def create(self, *, id: str | None = None, chat_id: str | None = None,
               persist: bool | None = None, ctx: Any = None) -> Session:
        persist = self.root is not None if persist is None else persist
        path = None
        if persist and self.root is not None:
            key = chat_id or id or uuid.uuid4().hex[:12]
            path = self.root / f"{key}.jsonl"
        session = Session(id=id, ctx=ctx, chat_id=chat_id, persist=persist, path=path)
        self.sessions[session.id] = session
        return session

    def open(self, id: str) -> Session | None:
        return self.sessions.get(id)

    def restore(self, path: Path, *, ctx: Any = None,
                chat_id: str | None = None) -> Session:
        """Read a JSONL log back into a Session (replay-safe)."""
        events: list[dict[str, Any]] = []
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        session = Session(id=path.stem, ctx=ctx, chat_id=chat_id or path.stem)
        for event in events:
            session._events.append(event)
            session.seq = max(session.seq, int(event.get("seq") or 0))
            session.projections.fold(event)
        self.sessions[session.id] = session
        return session

    def list(self, root: Path | None = None) -> list[dict[str, Any]]:
        base = root or self.root
        if base is None or not Path(base).exists():
            return []
        directory = Path(base)
        out = []
        for path in sorted(directory.glob("*.jsonl")):
            try:
                last = None
                for line in path.read_text(encoding="utf-8").splitlines()[-1:]:
                    last = json.loads(line) if line.strip() else None
                out.append({"id": path.stem, "events": _count_lines(path),
                            "updated": path.stat().st_mtime,
                            "last": (last or {}).get("type")})
            except (OSError, json.JSONDecodeError):
                continue
        return out


def _count_lines(path: Path) -> int:
    try:
        with path.open("rb") as fh:
            return sum(1 for _ in fh)
    except OSError:
        return 0
