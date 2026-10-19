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

The full research harness runs Propose–Probe–Compose search over benchmark rollouts
with an LLM judge and a per-domain experience library. This module keeps the same
eight-component model and adds a local, contrast-based PPC loop for a single-user
workbench: prompt builders for propose/judge/compose/reflect, tolerant JSON parsing,
safe edit application, and deterministic fallbacks so a small local model can still
produce a validated refinement.

The app supplies the model calls and persistence; everything here is pure and
unit-testable.
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
        elif k in DEFAULT_HARNESS and isinstance(v, str):
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
            body += (
                "\n\nCall one either by its own name as a tool "
                "({\"tool\": \"<name>\", \"arguments\": {\"task\": ...}}) or with "
                "{\"tool\": \"delegate\", \"arguments\": {\"name\": ..., \"task\": ...}}."
            )
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


# --------------------------------------------------------------------------- #
# recursive refinement: Propose–Probe–Compose
# --------------------------------------------------------------------------- #
# The research harness runs this loop over benchmark rollouts with an LLM judge.
# Here the task is the chat's current request and the judge compares behaviour
# without ground truth. Everything is expressed as small, safe edits so a local
# model can participate without rewriting the whole harness on every step.

RECURSION_MODES = ("sequential", "parallel")

EDITABLE_TEXT = ("system_prompt", "guardrails", "memory")
EDITABLE_LISTS = ("skills", "scripts", "subagents", "plugins")
MAX_SECTION = 5000
MAX_ITEMS = 12


def parse_json(text: str) -> Any:
    """Tolerant JSON extraction: models like to wrap objects in prose or fences."""
    raw = (text or "").strip()
    if not raw:
        return None
    fenced = re.search(r"```(?:json)?\s*(.*?)```", raw, re.S | re.I)
    if fenced:
        raw = fenced.group(1).strip()
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = raw.find(opener), raw.rfind(closer)
        if start < 0 or end <= start:
            continue
        blob = raw[start:end + 1]
        for candidate in (blob, re.sub(r",\s*([}\]])", r"\1", blob)):
            try:
                return json.loads(candidate)
            except Exception:
                continue
    return None


def harness_brief(harness: dict[str, Any], limit: int = 2400) -> str:
    """Compact, prompt-friendly view of every component (c1..c8)."""
    h = normalize(harness)
    lines = [f"name: {h.get('name') or 'harness'}"]
    for key in EDITABLE_TEXT:
        body = (h.get(key) or "").strip()
        if body:
            lines.append(f"{key}: {body[:420]}")
    for key in EDITABLE_LISTS:
        items = [x for x in (h.get(key) or []) if isinstance(x, dict)]
        if items:
            names = ", ".join(str(x.get("name") or "?") for x in items[:8])
            lines.append(f"{key} ({len(items)}): {names}")
            for x in items[:3]:
                body = str(x.get("body") or x.get("description") or "").strip()
                if body:
                    lines.append(f"  - {x.get('name')}: {body[:180]}")
    if h.get("tool_guidance"):
        lines.append("tool_guidance: " + ", ".join(list(h["tool_guidance"])[:10]))
    rules = h.get("rules") or {}
    active = [k for k in ("blocked_input", "forbid_output", "require_json",
                          "max_prompt_tokens", "tool_allowlist") if rules.get(k)]
    if active:
        lines.append("rules: " + ", ".join(active))
    return "\n".join(lines)[:limit]


def edits_from_payload(raw: Any) -> list[dict[str, Any]]:
    """Accept either the edit schema or a full harness-looking object."""
    if isinstance(raw, dict) and isinstance(raw.get("edits"), list):
        raw_edits = [e for e in raw["edits"] if isinstance(e, dict)]
    elif isinstance(raw, list):
        raw_edits = [e for e in raw if isinstance(e, dict)]
    elif isinstance(raw, dict):
        raw_edits = []
        for key in EDITABLE_TEXT:
            if isinstance(raw.get(key), str) and raw[key].strip():
                raw_edits.append({"component": key, "op": "replace", "text": raw[key]})
        for key in EDITABLE_LISTS:
            for item in raw.get(key) or []:
                if isinstance(item, dict):
                    raw_edits.append({"component": key, "op": "add", "item": item})
        if isinstance(raw.get("tool_guidance"), dict):
            for key, note in raw["tool_guidance"].items():
                raw_edits.append({"component": "tool_guidance", "op": "set",
                                  "key": str(key), "text": str(note)})
    else:
        raw_edits = []
    return [e for e in (normalize_edit(x) for x in raw_edits) if e]


COMPONENT_ALIASES = {
    "system": "system_prompt", "role": "system_prompt", "prompt": "system_prompt",
    "system prompt": "system_prompt", "system_prompt": "system_prompt",
    "guardrail": "guardrails", "guardrails": "guardrails", "rules": "guardrails",
    "memory": "memory", "context": "memory",
    "skill": "skills", "skills": "skills", "script": "scripts", "scripts": "scripts",
    "subagent": "subagents", "subagents": "subagents",
    "plugin": "plugins", "plugins": "plugins",
    "tool_guidance": "tool_guidance", "tool guidance": "tool_guidance",
    "tool": "tool_guidance",
}


def normalize_edit(edit: dict[str, Any]) -> dict[str, Any] | None:
    """Map the many shapes a local model invents onto the edit schema."""
    if not isinstance(edit, dict):
        return None
    raw_comp = str(edit.get("component") or edit.get("target") or edit.get("section") or "")
    comp = COMPONENT_ALIASES.get(raw_comp.strip().lower(), "")
    # A model that edits the answer rather than the harness must not be obeyed.
    if not comp or any(w in raw_comp.lower() for w in ("output", "answer", "response")):
        return None
    op = str(edit.get("op") or edit.get("type") or "append").strip().lower()
    if op not in ("append", "replace", "add", "set"):
        op = "append"
    text = str(edit.get("text") or edit.get("value") or edit.get("body") or "").strip()
    item = edit.get("item") if isinstance(edit.get("item"), dict) else None
    out: dict[str, Any] = {"component": comp, "op": op}
    if text:
        out["text"] = text
    if item:
        out["item"] = item
    if comp == "tool_guidance":
        key = str(edit.get("key") or edit.get("name") or "").strip()
        if key:
            out["key"] = key
    if comp == "rules":
        key = str(edit.get("key") or "").strip()
        if key:
            out["key"] = key
            out["value"] = edit.get("value", text)
    if not (out.get("text") or out.get("item")):
        return None
    if comp in ("skills", "scripts"):
        body = str((out.get("item") or {}).get("body") or out.get("text") or "")
        if re.match(r"\s*(def|class|function|import|from|const|let|var|#include)\b", body):
            return None   # a skill/script is an instruction, not executable code
    return out


def apply_edits(harness: dict[str, Any], edits: list[dict[str, Any]], *,
                report: bool = False) -> Any:
    """Apply proposed edits without ever losing the rest of the harness.

    `report=True` returns (harness, applied, skipped) so callers can detect a
    proposal that parsed as JSON but named no real component.
    """
    h = normalize(harness)
    applied = skipped = 0
    for raw in edits or []:
        e = normalize_edit(raw)
        if not e:
            skipped += 1
            continue
        comp = e["component"]
        op = e["op"]
        text = e.get("text") or ""
        before = json.dumps(h, sort_keys=True)
        if comp in EDITABLE_TEXT:
            if not text:
                skipped += 1
                continue
            if op == "replace":
                h[comp] = text[:MAX_SECTION]
            else:
                merged = (h[comp].rstrip() + "\n\n" + text).strip()
                h[comp] = merged[:MAX_SECTION]
        elif comp in EDITABLE_LISTS:
            item = e.get("item")
            if item is None and text:
                item = {"name": f"{comp[:-1]}-{len(h[comp]) + 1}", "body": text}
            if not item:
                skipped += 1
                continue
            name = str(item.get("name") or f"{comp[:-1]}-{len(h[comp]) + 1}")[:60]
            clean = {**item, "name": name}
            if comp == "plugins":
                clean.setdefault("body", str(item.get("body") or item.get("description") or "")[:1200])
            h[comp] = [x for x in h[comp] if x.get("name") != name][:MAX_ITEMS - 1] + [clean]
        elif comp == "tool_guidance" and text:
            key = str(e.get("key") or "").strip()[:60]
            if not key:
                skipped += 1
                continue
            h["tool_guidance"][key] = text[:1200]
        elif comp == "rules":
            key = str(e.get("key") or "").strip()
            if key not in h["rules"]:
                skipped += 1
                continue
            value = e.get("value", text)
            if isinstance(h["rules"][key], list) and isinstance(value, str):
                value = [x.strip() for x in value.splitlines() if x.strip()]
            h["rules"][key] = value
        else:
            skipped += 1
            continue
        if json.dumps(h, sort_keys=True) != before:
            applied += 1
        else:
            skipped += 1
    if report:
        return h, applied, skipped
    return h


def heuristic_edits(task: str) -> list[dict[str, Any]]:
    """Deterministic fallback when a small model cannot produce valid JSON."""
    t = " ".join((task or "").split())[:260]
    if not t:
        return []
    low = t.lower()
    if any(w in low for w in ("code", "function", "script", "bug", "python", "javascript", "sql")):
        tip = ("For implementation tasks: restate inputs, outputs and constraints in one line; "
               "then write the code in one block and finish with the exact check that proves it works.")
    elif any(w in low for w in ("write", "draft", "email", "report", "essay", "summar")):
        tip = ("For writing tasks: lead with the deliverable, keep one idea per paragraph, "
               "and close with assumptions or open questions only when they matter.")
    elif any(w in low for w in ("analy", "compare", "research", "evaluate", "why")):
        tip = ("For analysis tasks: state the method, give the findings in order of importance, "
               "and separate evidence from inference.")
    else:
        tip = "Restate the deliverable in one line, then complete it without preamble."
    return [{"component": "system_prompt", "op": "append",
             "text": f"Task focus: {t}\n{tip}"}]


# ---- prompt builders ------------------------------------------------------- #


def proposal_messages(harness: dict[str, Any], task: str, *,
                      lessons: list[dict[str, Any]] | None = None,
                      angle: str = "general", step: int = 1,
                      budget: int = 1) -> list[dict[str, str]]:
    lesson_text = "\n".join(f"- {l.get('lesson')}" for l in (lessons or [])[:5]) or "(none)"
    schema = (
        '{"rationale":"one sentence",'
        '"edits":[{"component":"system_prompt|guardrails|memory","op":"append|replace","text":"..."},'
        '{"component":"skills|scripts|subagents|plugins","op":"add",'
        '"item":{"name":"...","body":"...","description":"..."}},'
        '{"component":"tool_guidance","op":"set","key":"tool_name","text":"..."}],'
        '"lesson":{"cue":"when to apply","lesson":"imperative sentence",'
        '"evidence":"what in prior work motivates it","confidence":"low|medium|high"}}'
    )
    system = (
        "You are Harness², an editor of agent harnesses. You never change model weights; "
        "you edit the system around the model. Return JSON only, no prose. "
        "An edit is an instruction for future behaviour, never an answer to the task. "
        "Never describe the topic. Every edit must be worth its token cost."
    )
    user = (
        f"TASK:\n{task[:4000]}\n\n"
        f"CURRENT HARNESS:\n{harness_brief(harness)}\n\n"
        f"LESSONS FROM EARLIER WORK:\n{lesson_text}\n\n"
        f"STEP {step} OF {budget}, ANGLE: {angle}. "
        + ("Propose one invariant improvement that helps on any task of this shape."
           if angle == "general" else
           "Propose one domain-specific improvement that exploits what this particular task needs.")
        + "\n\nRules for a good edit:\n"
        "- Write a rule such as 'Before answering, restate the deliverable and list assumptions.'\n"
        "- Do NOT write the answer, a summary of the answer, or a fact about the topic.\n"
        "- Do NOT restate existing harness text.\n"
        "- Do NOT add tools that do not exist.\n"
        "- One edit only. Text must be an instruction, not a description.\n"
        "Components you may edit: system_prompt, guardrails, memory, tool_guidance, "
        "subagents, skills, scripts, plugins. Never target the model's output or answer.\n"
        f"Return exactly this shape:\n{schema}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def judge_messages(task: str, baseline: dict[str, Any],
                   candidates: list[dict[str, Any]]) -> list[dict[str, str]]:
    blocks = [
        f"BASELINE ({baseline.get('id', 'base')}):\n{(baseline.get('text') or '')[:1600]}"
    ]
    for c in candidates:
        flags = c.get("guardrails") or []
        blocks.append(
            f"CANDIDATE {c.get('id')} ({c.get('name') or ''}):\n"
            f"{(c.get('text') or '')[:1600]}"
            + (f"\nGUARDRAIL FLAGS: {'; '.join(flags)}" if flags else "")
        )
    system = (
        "You compare executions of the same task under different harnesses. There is no ground "
        "truth. Judge observable behaviour only. Ignore length unless it hurts the task. "
        "Use the exact CANDIDATE ids shown below. Return JSON only."
    )
    user = (
        f"TASK:\n{task[:2500]}\n\n" + "\n\n".join(blocks) +
        '\n\nScore the BASELINE (id "baseline") and every candidate 0–10 for how well it '
        'performs the task and how usable the answer is. Guardrail flags lower the score. '
        'Choose the winner among baseline and candidates. Return:\n'
        '{"scores":[{"id":"baseline","score":7,"note":"one sentence"},'
        '{"id":"<candidate id>","score":8,"note":"one sentence"}],'
        '"winner":"<baseline or candidate id>","rationale":"one sentence"}'
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def compose_messages(base: dict[str, Any], task: str, baseline: dict[str, Any],
                     candidates: list[dict[str, Any]], verdict: dict[str, Any]) -> list[dict[str, str]]:
    evidence = []
    for c in candidates:
        score = next((s for s in (verdict.get("scores") or [])
                      if str(s.get("id")) == str(c.get("id"))), {})
        evidence.append(
            f"{c.get('id')} ({c.get('name') or ''}) score={score.get('fit', '?')}/"
            f"{score.get('actionability', '?')}/{score.get('safety', '?')} · {score.get('note', '')}\n"
            f"{(c.get('text') or '')[:1200]}"
        )
    system = (
        "You are Harness² composing the next harness from probe evidence. Keep what worked, "
        "drop what did not, and preserve the rest of the harness. Return JSON only."
    )
    user = (
        f"TASK:\n{task[:3000]}\n\nBASE HARNESS:\n{harness_brief(base)}\n\n"
        f"BASELINE OUTPUT:\n{(baseline.get('text') or '')[:1200]}\n\n"
        "PROBES:\n" + "\n\n".join(evidence) +
        f"\n\nJUDGE: winner={verdict.get('winner')} — {verdict.get('rationale', '')}\n"
        'Return the same edit schema as before: {"rationale":"...","edits":[...]}. '
        'If no edit is justified, return {"rationale":"baseline already best","edits":[]}. '
        'Never include an edit that only repeats existing text. Never target the model output; '
        'edit only the harness components.'
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def reflect_messages(task: str, before: dict[str, Any], after: dict[str, Any],
                     evidence: str) -> list[dict[str, str]]:
    system = (
        "You maintain lessons for a recursive harness improver. Return JSON only, one lesson."
    )
    user = (
        f"TASK:\n{task[:2000]}\n\nBEFORE:\n{harness_brief(before)}\n\n"
        f"AFTER:\n{harness_brief(after)}\n\nEVIDENCE:\n{evidence[:2000]}\n\n"
        'Extract one transferable lesson that would help next time. Return '
        '{"cue":"when to apply it","lesson":"imperative sentence",'
        '"evidence":"what shows it","confidence":"low|medium|high"}'
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
