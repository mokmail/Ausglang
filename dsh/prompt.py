"""Prompt-section and tool-schema assembly.

Contributions are registered by plugins, scoped per agent, and concatenated in
ascending order with ties broken by section name. Tool schemas take part in the
same assembly because "what the model is told it can do" is one coherent thing.
Rendering is strict: an unknown `{{variable}}` fails loudly instead of shipping a
malformed prompt.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

__all__ = ["SystemPrompt", "SystemPromptPlugin", "Assembly", "PromptError"]

_VAR = re.compile(r"\{\{\s*([A-Za-z0-9_.-]+)\s*\}\}")


class PromptError(RuntimeError):
    pass


@dataclass
class Section:
    name: str
    order: float
    text: Any
    complete: bool = False
    interpolate: bool = True
    scope: str = "global"

    def render_text(self) -> str:
        value = self.text() if callable(self.text) else self.text
        return str(value or "")


@dataclass
class Assembly:
    sections: list[Section]
    tools: list[dict[str, Any]]
    variables: dict[str, str]
    complete: bool = False
    report: dict[str, Any] = field(default_factory=dict)


class SystemPrompt:
    """The registry every prompt contribution lands in (ctx.systemPrompt)."""

    def __init__(self, ctx: Any, config: dict[str, Any] | None = None) -> None:
        self.ctx = ctx
        self.config = dict(config or {})
        self._sections: dict[str, Section] = {}
        self._scoped: dict[str, dict[str, Section]] = {}
        self._variables: dict[str, Callable[[], Any]] = {}
        self._scoped_vars: dict[str, dict[str, Callable[[], Any]]] = {}
        self._tool_providers: list[Callable[[], list[dict[str, Any]]]] = []

        if self.config.get("includeHarnessIdentity", True):
            self.section("harness:identity", -1000,
                         "You are an AI agent powered by DeepSeek Harness.")
        prefix = (self.config.get("personaPrefix") or "").strip()
        if prefix:
            self.section("deployment:persona-prefix", 0, prefix)
        suffix = (self.config.get("personaSuffix") or "").strip()
        if suffix:
            self.section("deployment:persona-suffix", 10200, suffix)

    # -- registration ------------------------------------------------------ #
    def section(self, name: str, order: float, text: Any, *,
                complete: bool = False, interpolate: bool = True,
                scope: str = "global") -> Callable[[], None]:
        sec = Section(name=name, order=float(order), text=text,
                      complete=complete, interpolate=interpolate, scope=scope)
        target = self._sections if scope == "global" else self._scoped.setdefault(scope, {})
        target[name] = sec
        self.ctx.emit("system-prompt/change", {"section": name, "order": order})

        def dispose() -> None:
            if target.get(name) is sec:
                target.pop(name, None)

        return dispose

    def variable(self, name: str, provider: Callable[[], Any], *,
                 scope: str = "global") -> Callable[[], None]:
        target = self._variables if scope == "global" else self._scoped_vars.setdefault(scope, {})
        target[name] = provider

        def dispose() -> None:
            if target.get(name) is provider:
                target.pop(name, None)

        return dispose

    def tool_provider(self, provider: Callable[[], list[dict[str, Any]]]) -> Callable[[], None]:
        self._tool_providers.append(provider)

        def dispose() -> None:
            if provider in self._tool_providers:
                self._tool_providers.remove(provider)

        return dispose

    def get_section_order(self, name: str) -> float:
        sec = self._sections.get(name)
        return sec.order if sec else 1000.0

    # -- assembly ---------------------------------------------------------- #
    async def assemble_async(self, scope: str = "global",
                             tools: list[dict[str, Any]] | None = None) -> Assembly:
        sections = self._visible_sections(scope)
        variables = self._visible_variables(scope)
        resolved: dict[str, str] = {}
        for name, provider in variables.items():
            value = provider() if callable(provider) else provider
            resolved[name] = "" if value is None else str(value)

        schemas = tools if tools is not None else self._collect_tools()
        ordered = self.order_tools(schemas)

        assembly = Assembly(sections=sections, tools=ordered, variables=resolved)
        assembly = await self.ctx.waterfall("system-prompt/assemble", assembly)
        if not isinstance(assembly, Assembly):
            raise PromptError("system-prompt/assemble returned a non-assembly value")

        complete = [s for s in assembly.sections if s.complete and s.render_text().strip()]
        if len(complete) > 1:
            raise PromptError("more than one effective `complete` section")
        assembly.complete = bool(complete)
        assembly.report = self.report(assembly)
        return assembly

    def _visible_sections(self, scope: str) -> list[Section]:
        merged: dict[str, Section] = dict(self._sections)
        for name, sec in (self._scoped.get(scope) or {}).items():
            merged[name] = sec  # the agent scope shadows a same-named global
        return sorted(merged.values(), key=lambda s: (s.order, s.name))

    def _visible_variables(self, scope: str) -> dict[str, Callable[[], Any]]:
        merged = dict(self._variables)
        merged.update(self._scoped_vars.get(scope) or {})
        return merged

    def _collect_tools(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for provider in self._tool_providers:
            try:
                out.extend(provider() or [])
            except Exception:
                continue
        return out

    def order_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        order = self.config.get("toolOrder")
        if not order:
            return tools
        rest = [t for t in tools]
        picked: list[dict[str, Any]] = []
        for name in order:
            if name == "<unlisted-tools>":
                continue
            match = next((t for t in rest if t.get("name") == name), None)
            if match is not None:
                picked.append(match)
                rest.remove(match)
        return picked + rest

    def render(self, assembly: Assembly) -> str:
        blocks: list[str] = []
        for section in assembly.sections:
            if section.complete:
                continue
            text = section.render_text().strip()
            if not text:
                continue
            if section.interpolate:
                text = self.interpolate(text, assembly.variables)
            blocks.append(text)
        if assembly.complete:
            complete = next(s for s in assembly.sections if s.complete)
            text = complete.render_text().strip()
            if complete.interpolate:
                text = self.interpolate(text, assembly.variables)
            return text
        return "\n\n".join(blocks)

    @staticmethod
    def interpolate(text: str, variables: dict[str, str]) -> str:
        def sub(match: re.Match[str]) -> str:
            key = match.group(1)
            if key not in variables:
                raise PromptError(f"unknown prompt variable {{{{{key}}}}}")
            return variables[key]

        return _VAR.sub(sub, text)

    def report(self, assembly: Assembly) -> dict[str, Any]:
        rows = []
        total = 0
        for section in assembly.sections:
            text = section.render_text()
            chars = len(text)
            total += chars
            rows.append({"name": section.name, "order": section.order,
                         "chars": chars, "tokens": (chars + 3) // 4,
                         "complete": section.complete})
        return {
            "sections": rows,
            "total": {"chars": total, "tokens": (total + 3) // 4},
            "tools": [t.get("name") for t in assembly.tools],
            "variables": sorted(assembly.variables),
        }


class SystemPromptPlugin:
    """Mounted as the `system-prompt` row."""

    name = "system-prompt"
    inject: tuple[str, ...] = ()

    def __init__(self, ctx: Any, config: dict[str, Any] | None = None) -> None:
        self.ctx = ctx
        self.config = dict(config or {})
        self.prompt = SystemPrompt(ctx, self.config)
        ctx.provide("systemPrompt", self.prompt)

    def apply(self) -> None:
        pass
