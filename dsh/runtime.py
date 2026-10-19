"""Composition bridge: a workbench harness becomes a mounted plugin tree.

`build_rows` turns the eight-component harness plus the app's runtime facts into
patch layers over the `workbench` profile; `build_host` mounts them; `run_turn`
seeds the session with the persisted chat history and drives one turn.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, AsyncIterator, Callable

from . import compaction as compaction_mod
from . import harness as harness_mod
from . import llm as llm_mod
from . import loop as loop_mod
from . import prompt as prompt_mod
from . import tools as tools_mod
from .kernel import PluginHost
from .profiles import PROFILES, compose_profile, dump_config
from .session import SessionStore

__all__ = ["REGISTRY", "build_rows", "build_host", "composition_report",
           "run_turn", "preview_system", "PROFILE_NAME"]

PROFILE_NAME = "workbench"

REGISTRY = {
    # infrastructure
    "session": harness_mod.SessionPlugin,
    "system-prompt": prompt_mod.SystemPromptPlugin,
    "tools": tools_mod.ToolsPlugin,
    "llm": llm_mod.LLMPlugin,
    "agent-loop": loop_mod.AgentLoopPlugin,
    "compaction-basic": compaction_mod.CompactionPlugin,
    "chat-surface": harness_mod.ChatSurfacePlugin,
    "runtime-context": harness_mod.RuntimeContextPlugin,
    "harness-tool-protocol": harness_mod.ToolProtocolPlugin,
    # harness components c1..c8
    "harness-role": harness_mod.RolePlugin,
    "guardrails": harness_mod.GuardrailsPlugin,
    "memory": harness_mod.MemoryPlugin,
    "lessons": harness_mod.LessonsPlugin,
    "harness-tool-guidance": harness_mod.ToolGuidancePlugin,
    "skills": harness_mod.SkillsPlugin,
    "harness-scripts": harness_mod.ScriptsPlugin,
    "harness-plugins": harness_mod.HooksPlugin,
    "subagents": harness_mod.SubagentsPlugin,
    # tools
    "artifacts": harness_mod.ArtifactsPlugin,
    "mcp-tools": harness_mod.McpToolsPlugin,
}


def build_rows(
    harness: dict[str, Any],
    params: dict[str, Any],
    *,
    chat_id: str = "",
    session_root: str | Path | None = None,
    memory_text: str = "",
    lessons: list[dict[str, Any]] | None = None,
    tool_schemas: list[dict[str, Any]] | None = None,
    tool_call: Callable[..., Any] | None = None,
    artifact_write: Callable[..., Any] | None = None,
    artifact_read: Callable[..., Any] | None = None,
    artifact_list: Callable[..., Any] | None = None,
    subagent_runner: Callable[..., Any] | None = None,
    model: str = "",
    provider: str = "sglang",
    workspace: str = "",
) -> list[dict[str, Any]]:
    """Compose the workbench profile and patch the harness into it."""
    harness = harness or {}
    rules = harness.get("rules") or {}
    profile = PROFILES[PROFILE_NAME]
    base = compose_profile(profile)

    patch: list[dict[str, Any]] = [
        {"id": "session", "config": {
            "root": str(session_root) if session_root else "",
            "persist": bool(session_root),
        }},
        {"id": "system-prompt", "config": {
            "includeHarnessIdentity": True,
            "includeRuntimeContext": True,
            "personaPrefix": "",
            "personaSuffix": "",
        }},
        {"id": "tools", "config": {
            "mode": "native",
            "maxParallelToolCalls": int(params.get("max_parallel_tool_calls") or 4),
            "restrict": {"allow": [a for a in (rules.get("tool_allowlist") or []) if a],
                         "deny": []},
        }},
        {"id": "llm", "config": {"adapter": "openai-compatible"}},
        {"id": "agent-loop", "config": {
            "maxToolRounds": int(params.get("max_tool_rounds") or 3),
        }},
        {"id": "compaction-basic", "config": {
            "contextWindow": int(params.get("context_window") or params.get("ctx") or 12288),
            "maxTokens": int(params.get("max_tokens") or 512),
            "thresholdRatio": float(params.get("compact_threshold_ratio") or 0.8),
            "retainRatio": float(params.get("compact_retain_ratio") or 0.16),
            "headroomTokens": int(params.get("compact_headroom_tokens") or 4096),
            "auto": bool(params.get("auto_compaction", True)),
            "maxOverflowRetries": int(params.get("max_overflow_retries") or 1),
            "pruneToolResultChars": int(params.get("prune_tool_result_chars") or 1800),
        }},
        {"id": "guardrails", "config": {
            "guardrails": harness.get("guardrails") or "",
            "rules": rules,
        }},
        {"id": "memory", "config": {
            "memory": harness.get("memory") or "",
            "memory_text": memory_text,
            "json_mode": bool(params.get("json_mode")),
        }},
        {"id": "lessons", "config": {"rows": lessons or []}},
        {"id": "skills", "config": {"skills": harness.get("skills") or []}},
        {"id": "subagents", "config": {
            "subagents": harness.get("subagents") or [],
            "runner": subagent_runner,
            "maxDepth": int(params.get("max_subagent_depth") or 1),
        }},
        {"id": "artifacts", "config": {
            "write": artifact_write, "read": artifact_read, "list": artifact_list,
        }},
        {"id": "mcp-tools", "config": {
            "schemas": tool_schemas or [],
            "call": tool_call,
        }},
        # components that only contribute prompt sections or tool decoration
        {"id": "harness-role", "name": "harness-role",
         "config": {"system_prompt": harness.get("system_prompt") or ""}},
        {"id": "harness-tool-guidance", "name": "harness-tool-guidance",
         "config": {"guidance": harness.get("tool_guidance") or {}},
         "insert_after": "mcp-tools"},
        {"id": "harness-scripts", "name": "harness-scripts",
         "config": {"scripts": harness.get("scripts") or []}},
        {"id": "harness-plugins", "name": "harness-plugins",
         "config": {"plugins": harness.get("plugins") or []}},
        {"id": "runtime-context", "name": "runtime-context",
         "config": {"model": model, "provider": provider, "workspace": workspace}},
    ]

    from .profiles import apply_patch

    rows = base
    for entry in patch:
        rows = apply_patch(rows, [entry])  # unknown ids append, known ids replace
    return rows


def build_host(rows: list[dict[str, Any]], *, adapter: Any,
               base_url: str = "", api_key: str = "", name: str = "workbench") -> PluginHost:
    host = PluginHost(REGISTRY, name=name)
    host.mount(rows)
    llm = host.root.get("llm")
    if llm is not None and adapter is not None:
        llm.register_adapter("openai-compatible", adapter)
    return host


def _jsonable(value: Any) -> Any:
    """Configs carry callbacks; the report must stay serializable."""
    if callable(value):
        return "<callback>"
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def composition_report(rows: list[dict[str, Any]], host: PluginHost) -> dict[str, Any]:
    dumped = host.dump()
    services_by_row = {}
    for rid, entry in host.mounted.items():
        ctx = entry["ctx"]
        services_by_row[rid] = sorted(ctx._services)
    for item in dumped["rows"]:
        item["provides"] = services_by_row.get(item["id"], [])
        item["config"] = _jsonable(item.get("config") or {})
    return {
        "profile": PROFILE_NAME,
        "bundles": PROFILES[PROFILE_NAME].bundles,
        "rows": dumped["rows"],
        "services": dumped["services"],
        "listeners": dumped["listeners"],
        "failed": dumped["failed"],
        "config": _jsonable(dump_config(rows)),
    }


async def preview_system(rows: list[dict[str, Any]], *, adapter: Any = None) -> dict[str, Any]:
    host = build_host(rows, adapter=adapter)
    prompt = host.root.get("systemPrompt")
    tools = host.root.get("tools")
    if prompt is None:
        return {"system": "", "report": {}, "services": host.services()}
    assembly = await prompt.assemble_async("agent", tools.schemas("agent") if tools else [])
    text = prompt.render(assembly)
    report = assembly.report
    report["services"] = host.services()
    report["plugins"] = [r["name"] for r in host.dump()["rows"]]
    return {"system": text, "report": report, "assembly": assembly}


# --------------------------------------------------------------------------- #
# turn driver
# --------------------------------------------------------------------------- #


def seed_history(session: Any, history: list[dict[str, Any]]) -> None:
    """Replay the workspace's persisted turns into the session log.

    Restored facts are marked so a reader can tell replay from live work, but
    they derive into model history exactly like live events.
    """
    for message in history:
        role = message.get("role")
        text = message.get("content") or ""
        if not text:
            continue
        if role == "user":
            session.append("user/message", text=text, source="prompt", restored=True)
        elif role == "assistant":
            session.append("assistant/message", text=text, restored=True, finish="stop")


def _model_history_len(session: Any) -> int:
    """How many user/assistant turns the log already covers."""
    count = 0
    for message in session.derive_messages():
        content = str(message.get("content") or "")
        if message.get("role") not in ("user", "assistant"):
            continue
        if content.startswith(("Observation from ", "[compact]")):
            continue
        count += 1
    return count


async def run_turn(
    host: PluginHost,
    *,
    chat_id: str,
    text: str,
    history: list[dict[str, Any]] | None = None,
    attachments: list[dict[str, Any]] | None = None,
    model: str = "",
    params: dict[str, Any] | None = None,
    holder: dict[str, Any] | None = None,
) -> AsyncIterator[dict[str, Any]]:
    store: SessionStore = host.root.require("agentSessionStore")
    loop = host.root.require("agentLoop")

    # One session per chat: resume the durable log and only top up the turns the
    # workspace recorded after it (a fresh chat seeds the whole history once).
    path = (store.root / f"{chat_id}.jsonl") if (store.root and chat_id) else None
    if path is not None and path.exists():
        session = store.restore(path, ctx=host.root, chat_id=chat_id)
        session.persist, session.path = True, path
        logged = _model_history_len(session)
        if logged < len(history or []):
            seed_history(session, (history or [])[logged:])
    else:
        session = store.create(chat_id=chat_id or None, persist=path is not None, ctx=host.root)
        seed_history(session, history or [])

    agent = loop.create(session, {
        "model": model,
        **(params or {}),
    }, scope="agent")
    if holder is not None:
        holder["agent"] = agent
        holder["session"] = session
    async for frame in agent.run(text, attachments=attachments):
        yield frame


async def run_subagent(host: PluginHost, *, name: str, task: str, parent: Any = None,
                       system: str = "", meta: dict[str, Any] | None = None,
                       params: dict[str, Any] | None = None,
                       max_tokens: int = 512) -> str:
    """spawn-in-process provider: a fresh child agent in the same composition."""
    loop = host.root.require("agentLoop")
    store: SessionStore = host.root.require("agentSessionStore")
    prompt = host.root.get("systemPrompt")
    scope = f"subagent:{name}"
    session = store.create(persist=False, ctx=host.root)
    dispose = prompt.section(scope, 5, system) if (prompt and system.strip()) else None
    depth = int((meta or {}).get("depth") or 1)
    try:
        child = loop.create(session, {
            "model": (params or {}).get("model") or "",
            **{k: v for k, v in (params or {}).items()
               if k in ("temperature", "top_p", "max_tokens", "enable_thinking")},
            "max_tokens": int(max_tokens),
            "exec_meta": {"depth": depth},
        }, scope=scope, max_tool_rounds=1)
        text = ""
        async for frame in child.run(task):
            if frame.get("type") == "done":
                text = frame.get("text") or ""
        return text or "(the subagent returned nothing)"
    finally:
        if dispose:
            dispose()
