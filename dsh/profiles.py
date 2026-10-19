"""Profiles, bundles and ordered patch layers.

A running harness is a plugin tree composed at boot from ordered layers, exactly
as in DeepSeek Harness:

    profile -> bundles (ordered) -> profile patch -> home patch -> --patch overlays

A *bundle* is an ordered list of config rows. A *patch* targets a row by id and
replaces its whole config, or inserts new rows around an existing id. Layers
never concatenate configs: the last layer that names a row owns that row.
"""

from __future__ import annotations

import copy
from typing import Any, Iterable

Row = dict[str, Any]

__all__ = [
    "Row", "Bundle", "Profile", "merge_rows", "apply_patch",
    "compose_profile", "dump_config", "BUNDLES", "PROFILES",
]


def row(name: str, config: dict[str, Any] | None = None, *, id: str | None = None,
        enabled: bool = True) -> Row:
    out: Row = {"id": id or name, "name": name, "config": dict(config or {})}
    if not enabled:
        out["enabled"] = False
    return out


class Bundle:
    def __init__(self, id: str, rows: Iterable[Row], *, description: str = "") -> None:
        self.id = id
        self.description = description
        self.rows = [copy.deepcopy(r) for r in rows]

    def dump(self) -> dict[str, Any]:
        return {"id": self.id, "description": self.description,
                "rows": [r["id"] for r in self.rows]}


class Profile:
    def __init__(self, name: str, bundles: Iterable[str], *,
                 patch: Iterable[Row] | None = None, description: str = "") -> None:
        self.name = name
        self.bundles = list(bundles)
        self.patch = [copy.deepcopy(r) for r in (patch or [])]
        self.description = description

    def dump(self) -> dict[str, Any]:
        return {"name": self.name, "bundles": self.bundles,
                "patch": [r["id"] for r in self.patch], "description": self.description}


def merge_rows(target: list[Row], source: list[Row]) -> list[Row]:
    """Append rows whose id is new, replace config of rows already present."""
    index = {r["id"]: i for i, r in enumerate(target)}
    for r in source:
        if r["id"] in index:
            target[index[r["id"]]] = copy.deepcopy(r)
        else:
            target.append(copy.deepcopy(r))
            index[r["id"]] = len(target) - 1
    return target


def apply_patch(rows: list[Row], patch: list[Row]) -> list[Row]:
    """Apply one patch layer.

    Two operations, as in the harness: target a row by id (replace its whole
    config, or disable it with `enabled: false`), or insert a new row, optionally
    anchored with `insert_after`.
    """
    out = copy.deepcopy(rows)
    index = {r["id"]: i for i, r in enumerate(out)}
    for spec in patch or []:
        rid = spec.get("id") or spec.get("name")
        if not rid:
            continue
        anchor = spec.get("insert_after")
        if rid not in index and anchor:
            spec = copy.deepcopy(spec)
            pos = index.get(anchor)
            if pos is None:
                out.append(spec)
            else:
                out.insert(pos + 1, spec)
                index = {r["id"]: i for i, r in enumerate(out)}
            continue
        if rid in index:
            before = copy.deepcopy(out[index[rid]])
            merged = {**before, **copy.deepcopy(spec), "id": rid}
            merged.setdefault("name", before["name"])
            if "config" in spec:
                merged["config"] = copy.deepcopy(spec["config"])
            out[index[rid]] = merged
        else:
            out.append(copy.deepcopy(spec))
            index[rid] = len(out) - 1
    return out


def compose_profile(profile: Profile, *layers: Iterable[Row]) -> list[Row]:
    rows: list[Row] = []
    for bundle_id in profile.bundles:
        bundle = BUNDLES.get(bundle_id)
        if bundle is None:
            raise KeyError(f"profile {profile.name!r} references unknown bundle {bundle_id!r}")
        merge_rows(rows, bundle.rows)
    for layer in (profile.patch, *layers):
        rows = apply_patch(rows, list(layer or []))
    return rows


def dump_config(rows: Iterable[Row]) -> list[dict[str, Any]]:
    """The `dsh --dump-config` view: every row a patch can name."""
    return [
        {"id": r["id"], "name": r["name"],
         "enabled": r.get("enabled", True),
         "config": copy.deepcopy(r.get("config") or {})}
        for r in rows
    ]


# --------------------------------------------------------------------------- #
# shipped composition
# --------------------------------------------------------------------------- #
# The workbench bundle mirrors `dsh-base`: model adapter, prompt assembly,
# session log, tools, the agent loop, compaction and the harness components.
# `headless` is the same tree without the chat surface; `sdk-minimal` is the
# deliberate exception that composes its own complete tree.

BASE_ROWS: list[Row] = [
    row("session", {"persist": True}),
    row("system-prompt", {
        "includeHarnessIdentity": True,
        "includeRuntimeContext": True,
        "personaPrefix": "",
        "personaSuffix": "",
    }),
    row("tools", {"mode": "native", "maxParallelToolCalls": 6}),
    row("llm", {"adapter": "openai-compatible"}),
    row("agent-loop", {"maxParallelToolCalls": 6}),
    row("harness-tool-protocol"),
    row("compaction-basic", {
        "thresholdRatio": 0.8,
        "retainRatio": 0.16,
        "headroomTokens": 4096,
        "auto": True,
        "maxOverflowRetries": 1,
    }),
    row("guardrails"),
    row("memory"),
    row("lessons"),
    row("skills"),
    row("subagents", {"maxDepth": 1, "maxActiveSubagents": 8}),
    row("artifacts"),
    row("mcp-tools"),
]

BUNDLES: dict[str, Bundle] = {
    "dsh-base": Bundle("dsh-base", BASE_ROWS,
                       description="model adapter, tools, persistence and the agent loop"),
    "dsh-web-app": Bundle("dsh-web-app", [
        row("chat-surface", {"transport": "sse"}),
    ], description="browser application surface"),
    "dsh-headless": Bundle("dsh-headless", [
        row("headless-runner", {"transport": "stdout"}),
    ], description="one-shot runner, no server"),
    "dsh-sdk-minimal": Bundle("dsh-sdk-minimal", [
        row("session", {"persist": False}),
        row("system-prompt", {"includeHarnessIdentity": True}),
        row("tools", {"mode": "native"}),
        row("llm", {"adapter": "openai-compatible"}),
        row("agent-loop"),
    ], description="explicit minimal tree; does not stack dsh-base"),
}

PROFILES: dict[str, Profile] = {
    "workbench": Profile(
        "workbench", ["dsh-base", "dsh-web-app"],
        description="the local SGLang workbench (this app)",
    ),
    "headless": Profile(
        "headless", ["dsh-base", "dsh-headless"],
        description="no chat surface; run one turn and print it",
    ),
    "sdk-minimal": Profile(
        "sdk-minimal", ["dsh-sdk-minimal"],
        description="smallest explicit composition",
    ),
}
