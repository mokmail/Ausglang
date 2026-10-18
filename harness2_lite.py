#!/usr/bin/env python3
"""Harness²-lite — the eight editable harness components, adapted from
google-research/harness2 for a single-user local workbench.

harness2 treats a harness as eight editable components around the model:

    c1 system prompt     the base instruction set
    c2 guardrails        rules + machine checks (input/output, budgets, tool allowlist)
    c3 memory            durable context: preamble, stored facts, domain lessons
    c4 tool guidance     per-tool usage notes folded into the tool schema descriptions
    c5 subagents         named delegates the model may call for focused subtasks
    c6 skills            reusable instruction files (SKILL.md style)
    c7 scripts           named prompt recipes that are always available to the agent
    c8 plugins           behavioural hooks appended to the turn (lint, tone, format)

The full research harness also does Propose–Probe–Compose search over candidate
harnesses with an LLM judge and a per-domain experience library. This module
implements the parts that make a chat app more usable — the component model, the
guardrails, the experience library as "lessons", and a manual probe — and leaves
the recursive search to the research code.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from typing import Any

COMPONENTS = [
    ("system_prompt", "System prompt"),
    ("guardrails", "Guardrails"),
    ("memory", "Memory"),
    ("tool_guidance", "Tool guidance"),
    ("subagents", "Subagents"),
    ("skills", "Skills"),
    ("scripts", "Scripts"),
    ("plugins", "Plugins"),
]

DEFAULT_HARNESS: dict[str, Any] = {
    "name": "Default",
    "system_prompt": "You are a helpful, precise assistant running locally on the user's Mac.",
    "guardrails": "Stay factual. Say when you are unsure. Never invent tool output.",
    "memory": "",
    "tool_guidance": {},
    "subagents": [],
    "skills": [],
    "scripts": [],
    "plugins": [],
    "rules": {
        "blocked_input": [],
        "forbid_output": [],
        "require_json": False,
        "max_prompt_tokens": 0,
        "tool_allowlist": [],
        "include_lessons": True,
    },
}


def blank_harness(name: str = "New harness") -> dict[str, Any]:
    h = json.loads(json.dumps(DEFAULT_HARNESS))
    h["name"] = name
    h["system_prompt"] = ""
    h["guardrails"] = ""
    return h


def normalize(raw: dict[str, Any] | None) -> dict[str, Any]:
    """Coerce anything stored in the database into a complete harness."""
    h = json.loads(json.dumps(DEFAULT_HARNESS))
    if not raw:
        return h
    for k, v in raw.items():
        if k in ("skills", "scripts", "subagents", "plugins"):
            h[k] = v if isinstance(v, list) else []
        elif k == "tool_guidance":
            h[k] = v if isinstance(v, dict) else {}
        elif k == "rules":
            h[k] = {**DEFAULT_HARNESS["rules"], **(v if isinstance(v, dict) else {})}
        elif isinstance(v, str):
            h[k] = v
    return h


# --------------------------------------------------------------------------- #
# prompt composition
# --------------------------------------------------------------------------- #


def _block(title: str, body: str) -> str:
    body = (body or "").strip()
    return f"## {title}\n{body}" if body else ""


def compose_system(
    harness: dict[str, Any],
    *,
    lessons: list[dict[str, Any]] | None = None,
    memory_text: str = "",
    json_mode: bool = False,
) -> tuple[str, dict[str, Any]]:
    """Build the system message and a per-section size report."""
    h = normalize(harness)
    sections: list[tuple[str, str]] = []

    sections.append(("system_prompt", _block("Role", h["system_prompt"])))
    sections.append(("guardrails", _block("Guardrails (must hold)", h["guardrails"])))

    mem_parts = []
    if h["memory"].strip():
        mem_parts.append(h["memory"].strip())
    if memory_text.strip():
        mem_parts.append("Known facts about the user (use when relevant):\n" + memory_text.strip())
    if h["rules"].get("include_lessons", True) and lessons:
        rows = [f"- ({l.get('domain') or 'general'}) {l.get('lesson')}" for l in lessons]
        if rows:
            mem_parts.append("Lessons from earlier sessions in this domain:\n" + "\n".join(rows))
    if json_mode:
        mem_parts.append("Answer with a single valid JSON object and nothing else.")
    sections.append(("memory", _block("Context", "\n\n".join(mem_parts))))

    if h["skills"]:
        body = "\n\n".join(
            f"### skill: {s.get('name', 'skill')}\n{(s.get('body') or '').strip()}"
            for s in h["skills"] if (s.get("body") or "").strip()
        )
        sections.append(("skills", _block("Skills (apply when relevant)", body)))

    if h["scripts"]:
        body = "\n\n".join(
            f"### script: {s.get('name', 'script')}\n{(s.get('body') or '').strip()}"
            for s in h["scripts"] if (s.get("body") or "").strip()
        )
        sections.append(("scripts", _block("Scripts (named procedures you can follow)", body)))

    if h["subagents"]:
        body = "\n\n".join(
            f"- `{s.get('name')}` — {s.get('description') or 'delegate a subtask'}"
            for s in h["subagents"] if s.get("name")
        )
        if body:
            body += "\n\nCall one with the `delegate` tool: {\"tool\": \"delegate\", \"arguments\": {\"name\": ..., \"task\": ...}}"
            sections.append(("subagents", _block("Subagents you can delegate to", body)))

    if h["plugins"]:
        body = "\n\n".join(
            f"- {p.get('name', 'hook')}: {(p.get('body') or '').strip()}"
            for p in h["plugins"] if (p.get("body") or "").strip()
        )
        sections.append(("plugins", _block("Behavioural hooks applied to every turn", body)))

    text = "\n\n".join(s for _, s in sections if s.strip())
    report = {
        name: {"chars": len(body), "tokens": (len(body) + 3) // 4}
        for name, body in sections
        if body.strip()
    }
    report["total"] = {"chars": len(text), "tokens": (len(text) + 3) // 4}
    return text, report


def apply_tool_guidance(tools: list[dict[str, Any]], guidance: dict[str, str]) -> list[dict[str, Any]]:
    """Fold per-tool notes into the schemas the model sees (c4)."""
    if not guidance:
        return tools
    out = []
    for t in tools:
        note = guidance.get(str(t.get("name") or ""))
        if note:
            t = {**t, "description": f"{t.get('description', '').strip()}\nNote: {note.strip()}".strip()}
        out.append(t)
    return out


def filter_tools(tools: list[dict[str, Any]], allowlist: list[str]) -> list[dict[str, Any]]:
    allow = [a for a in (allowlist or []) if a]
    if not allow:
        return tools
    return [t for t in tools if t.get("name") in allow or t.get("server") in allow]


# --------------------------------------------------------------------------- #
# guardrails
# --------------------------------------------------------------------------- #


def check_input(harness: dict[str, Any], text: str) -> list[str]:
    rules = normalize(harness)["rules"]
    hits = []
    for pat in rules.get("blocked_input") or []:
        try:
            if re.search(pat, text, re.I):
                hits.append(f"input matches blocked pattern /{pat}/")
        except re.error:
            continue
    return hits


def check_output(harness: dict[str, Any], text: str) -> list[str]:
    rules = normalize(harness)["rules"]
    problems = []
    for pat in rules.get("forbid_output") or []:
        try:
            if re.search(pat, text, re.I):
                problems.append(f"output matches forbidden pattern /{pat}/")
        except re.error:
            continue
    if rules.get("require_json"):
        try:
            json.loads(text.strip())
        except Exception:
            problems.append("output is not valid JSON")
    return problems


def budget_trim(messages: list[dict[str, Any]], max_tokens: int) -> tuple[list[dict[str, Any]], int]:
    """Drop the oldest non-system turns until the prompt fits the budget."""
    if not max_tokens or max_tokens <= 0:
        return messages, 0
    def size(msgs: list[dict[str, Any]]) -> int:
        return sum((len(m.get("content") or "") + 3) // 4 for m in msgs)

    trimmed = 0
    while size(messages) > max_tokens and len(messages) > 2:
        for i, m in enumerate(messages):
            if m["role"] != "system":
                del messages[i]
                trimmed += 1
                break
        else:
            break
    return messages, trimmed


# --------------------------------------------------------------------------- #
# lessons (the experience library, per domain)
# --------------------------------------------------------------------------- #


def lesson_context(rows: list[dict[str, Any]], query: str, limit: int = 5) -> list[dict[str, Any]]:
    """Keyword-ranked retrieval; the research code retrieves by domain, we rank by overlap."""
    if not rows:
        return []
    words = {w for w in re.findall(r"[a-z0-9]{3,}", (query or "").lower())}
    scored = []
    for r in rows:
        blob = f"{r.get('cue', '')} {r.get('lesson', '')} {r.get('evidence', '')} {r.get('domain', '')}"
        tw = {w for w in re.findall(r"[a-z0-9]{3,}", blob.lower())}
        scored.append((len(words & tw), r))
    scored.sort(key=lambda x: -x[0])
    picked = [r for s, r in scored if s > 0][:limit]
    return picked or [r for _, r in scored[:limit]]


def new_id() -> str:
    return uuid.uuid4().hex[:12]


def now() -> float:
    return time.time()
