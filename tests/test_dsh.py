"""Unit and integration tests for the DeepSeek Harness port.

Run with:  python3 -m unittest discover -s tests -v
No network, no app: a scripted adapter stands in for the model.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from dsh.compaction import CompactionBasic, CompactionPlugin
from dsh.kernel import Context, Plugin, PluginHost
from dsh.llm import StreamChunk
from dsh.loop import Agent, extract_tool_calls
from dsh.profiles import PROFILES, apply_patch, compose_profile, dump_config, row
from dsh.prompt import PromptError, SystemPrompt
from dsh.session import Session, SessionStore
from dsh.tools import ExecContext, Tool, ToolRuntime
from dsh import runtime


class FakeAdapter:
    """Scripted streaming adapter: one string per model call."""

    def __init__(self, script: list[str], *, usage: dict | None = None) -> None:
        self.script = list(script)
        self.calls: list[list[dict]] = []
        self.usage = usage or {"prompt_tokens": 120, "completion_tokens": 8}

    def prepare(self, prepared) -> None:
        pass

    async def stream(self, prepared, messages, tools, signal=None):
        self.calls.append(messages)
        text = self.script.pop(0) if self.script else "(no script left)"
        for i in range(0, len(text), 40):
            yield StreamChunk("text", text=text[i:i + 40])
        yield StreamChunk("usage", usage=dict(self.usage))


class EchoToolPlugin(Plugin):
    name = "echo-tool"
    inject = ("tools",)

    def apply(self) -> None:
        async def echo(args, exec_ctx):
            return f"echo:{args.get('text')}"

        self.ctx.require("tools").register(Tool(
            name="echo", description="echo text back",
            parameters={"type": "object", "properties": {"text": {"type": "string"}},
                        "required": ["text"]},
            execute=echo, server="test"))


def build_test_host(rows, adapter):
    host = PluginHost(runtime.REGISTRY)
    host.register(EchoToolPlugin)
    host.mount(rows + [row("echo-tool")])
    host.root.require("llm").register_adapter("openai-compatible", adapter)
    return host


async def _drive(agent):
    async for _ in agent.run("go"):
        pass
    return agent.last_result


class KernelTests(unittest.TestCase):
    def test_services_are_visible_to_siblings_and_effects_unwind_lifo(self):
        class A(Plugin):
            name = "a"

            def apply(self):
                self.provide("svc", {"v": 1})
                self.ctx.effect(lambda: order.append("a"))

        class B(Plugin):
            name = "b"
            inject = ("svc",)

            def apply(self):
                order.append("b:" + str(self.ctx.require("svc")["v"]))
                self.ctx.effect(lambda: order.append("b"))

        order = []
        host = PluginHost({"a": A, "b": B})
        host.mount([row("a"), row("b")])
        self.assertEqual(order, ["b:1"])
        self.assertIn("svc", host.services())
        host.dispose()
        self.assertEqual(order, ["b:1", "b", "a"])

    def test_pending_plugin_activates_when_service_arrives(self):
        seen = []

        class Late(Plugin):
            name = "late"
            inject = ("thing",)

            def apply(self):
                seen.append("applied")

        class Provider(Plugin):
            name = "provider"

            def apply(self):
                self.provide("thing", 1)

        host = PluginHost({"late": Late, "provider": Provider})
        host.mount([row("late"), row("provider")])
        self.assertEqual(seen, ["applied"])

    def test_waterfall_delegation_and_short_circuit(self):
        ctx = Context()
        calls = []

        async def first(payload, nxt):
            calls.append("first")
            return await nxt({**payload, "n": payload["n"] + 1})

        async def second(payload, nxt):
            calls.append("second")
            return {"decision": "reject"}

        async def third(payload, nxt):
            calls.append("third")
            return await nxt(payload)

        ctx.on("evt", first, kind="waterfall", order=1)
        ctx.on("evt", second, kind="waterfall", order=2)
        ctx.on("evt", third, kind="waterfall", order=3)
        result = asyncio.run(ctx.waterfall("evt", {"n": 0}))
        self.assertEqual(result, {"decision": "reject"})
        self.assertEqual(calls, ["first", "second"])


class ProfileTests(unittest.TestCase):
    def test_patch_replaces_config_and_appends_rows(self):
        rows = compose_profile(PROFILES["workbench"])
        ids = [r["id"] for r in rows]
        self.assertIn("agent-loop", ids)
        patched = apply_patch(rows, [
            {"id": "agent-loop", "config": {"maxToolRounds": 9}},
            {"id": "extra", "name": "extra", "config": {"a": 1}},
        ])
        loop_row = next(r for r in patched if r["id"] == "agent-loop")
        self.assertEqual(loop_row["config"], {"maxToolRounds": 9})
        self.assertEqual(patched[-1]["id"], "extra")
        self.assertTrue(all("name" in r for r in dump_config(patched)))

    def test_patch_can_disable_a_row(self):
        rows = compose_profile(PROFILES["workbench"])
        patched = apply_patch(rows, [{"id": "mcp-tools", "enabled": False}])
        self.assertFalse(next(r for r in patched if r["id"] == "mcp-tools")["enabled"])


class PromptTests(unittest.TestCase):
    def test_order_ties_complete_and_variables(self):
        ctx = Context()
        prompt = SystemPrompt(ctx, {"includeHarnessIdentity": False})
        prompt.section("z", 10, "second")
        prompt.section("a", 10, "first")
        prompt.section("v", 20, "model={{model}}")
        prompt.variable("model", lambda: "qwen")
        assembly = asyncio.run(prompt.assemble_async())
        self.assertEqual(prompt.render(assembly), "first\n\nsecond\n\nmodel=qwen")

    def test_unknown_variable_fails_loudly(self):
        prompt = SystemPrompt(Context(), {"includeHarnessIdentity": False})
        prompt.section("x", 1, "hello {{nope}}")
        assembly = asyncio.run(prompt.assemble_async())
        with self.assertRaises(PromptError):
            prompt.render(assembly)

    def test_complete_section_replaces_the_prompt(self):
        prompt = SystemPrompt(Context(), {"includeHarnessIdentity": False})
        prompt.section("a", 1, "dropped")
        prompt.section("all", 5, "only this", complete=True)
        assembly = asyncio.run(prompt.assemble_async())
        self.assertEqual(prompt.render(assembly), "only this")

    def test_tool_order_with_rest_entry(self):
        prompt = SystemPrompt(Context(), {"includeHarnessIdentity": False,
                                          "toolOrder": ["b", "<unlisted-tools>"]})
        ordered = prompt.order_tools([{"name": "a"}, {"name": "b"}])
        self.assertEqual([t["name"] for t in ordered], ["b", "a"])


class SessionTests(unittest.TestCase):
    def test_derive_messages_includes_tool_observations(self):
        session = Session("s")
        session.append("system/message", text="sys")
        session.append("user/message", text="hello")
        session.append("assistant/message", text='{"tool":"echo","arguments":{}}')
        session.append("tool/call", tool="echo", arguments={})
        session.append("tool/result", tool="echo", content="ok")
        session.append("assistant/message", text="done")
        messages = session.derive_messages()
        self.assertEqual(messages[0], {"role": "system", "content": "sys"})
        self.assertEqual([m["role"] for m in messages[1:]],
                         ["user", "assistant", "user", "assistant"])
        self.assertIn("Observation from echo", messages[3]["content"])

    def test_compaction_markers_replace_and_prune(self):
        session = Session("s")
        session.append("user/message", text="old one")
        target = session.append("tool/result", tool="echo", content="x" * 4000)
        session.append("compaction/prune", targets={target["seq"]: 100})
        session.append("compaction/replace", upto=target["seq"], summary="SUMMARY")
        session.append("user/message", text="tail")
        messages = session.derive_messages()
        self.assertIn("SUMMARY", messages[0]["content"])
        self.assertEqual(messages[-1]["content"], "tail")
        self.assertNotIn("old one", json.dumps(messages))

    def test_jsonl_store_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = SessionStore(Path(tmp))
            session = store.create(chat_id="chat1", persist=True)
            session.append("user/message", text="hi")
            session.append("assistant/message", text="hello")
            restored = SessionStore(Path(tmp)).restore(Path(tmp) / "chat1.jsonl")
            self.assertEqual(restored.seq, session.seq)
            self.assertEqual(restored.derive_messages()[-1]["content"], "hello")


class ToolTests(unittest.TestCase):
    def _runtime(self):
        ctx = Context()
        runtime_ = ToolRuntime(ctx)
        runtime_.register(Tool(name="echo", description="e",
                               parameters={"type": "object",
                                           "properties": {"text": {"type": "string"}},
                                           "required": ["text"]},
                               execute=lambda a, e: f"echo:{a['text']}"))
        return ctx, runtime_

    def test_validation_and_unknown_tool(self):
        ctx, tools = self._runtime()

        async def go():
            bad = await tools.run("echo", {}, ExecContext())
            unknown = await tools.run("nope", {}, ExecContext())
            return bad, unknown

        bad, unknown = asyncio.run(go())
        self.assertEqual(bad.status, "error")
        self.assertEqual(unknown.error, "UNKNOWN_TOOL")

    def test_pre_execute_denial_and_guard_are_final(self):
        ctx, tools = self._runtime()

        async def deny(payload, nxt):
            return {"decision": "deny", "reason": "blocked"}

        ctx.on("tools/pre-execute", deny, kind="waterfall")
        result = asyncio.run(tools.run("echo", {"text": "x"}, ExecContext()))
        self.assertEqual(result.status, "denied")

        ctx2, tools2 = self._runtime()
        tools2.guard(lambda tool, args: "guard says no")
        result2 = asyncio.run(tools2.run("echo", {"text": "x"}, ExecContext()))
        self.assertEqual(result2.status, "denied")

    def test_post_execute_can_replace_result(self):
        ctx, tools = self._runtime()

        async def rewrite(result, nxt):
            result.content = result.content.upper()
            return await nxt(result)

        ctx.on("tools/post-execute", rewrite, kind="waterfall")
        result = asyncio.run(tools.run("echo", {"text": "x"}, ExecContext()))
        self.assertEqual(result.content, "ECHO:X")

    def test_restriction_masks_tools(self):
        ctx, tools = self._runtime()
        tools.restrict(deny=["echo"])
        self.assertEqual(tools.names(), [])
        tools.restrict(deny=[])  # replaces the rule
        self.assertEqual(tools.names(), ["echo"])


class LoopTests(unittest.TestCase):
    def test_turn_runs_tool_then_answers(self):
        harness = {"name": "t", "system_prompt": "Be terse.", "guardrails": "",
                   "memory": "", "tool_guidance": {}, "subagents": [], "skills": [],
                   "scripts": [], "plugins": [], "rules": {"blocked_input": [],
                                                           "forbid_output": []}}
        rows = runtime.build_rows(harness, {"context_window": 32768}, chat_id="c1")
        adapter = FakeAdapter([
            '```json\n{"tool":"echo","arguments":{"text":"hi"}}\n```',
            "All done.",
        ])
        host = build_test_host(rows, adapter)
        session = host.root.require("agentSessionStore").create(chat_id="c1")
        agent = host.root.require("agentLoop").create(session, {"model": "fake"})

        frames = []

        async def drive():
            async for frame in agent.run("say hi"):
                frames.append(frame)
            return agent.last_result

        result = asyncio.run(drive())
        kinds = [f["type"] for f in frames]
        self.assertIn("delta", kinds)
        self.assertIn("tool", kinds)
        self.assertEqual(result["text"], "All done.")
        self.assertEqual(result["rounds"], 1)
        types = [e["type"] for e in session.events]
        self.assertIn("tool/call", types)
        self.assertIn("tool/result", types)
        messages = session.derive_messages()
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("Be terse.", messages[0]["content"])
        self.assertEqual(messages[-1], {"role": "assistant", "content": "All done."})
        self.assertEqual(len(adapter.calls), 2)

    def test_guardrail_rejects_input_without_a_step(self):
        harness = {"system_prompt": "x", "guardrails": "y",
                   "rules": {"blocked_input": ["password"]}}
        rows = runtime.build_rows(harness, {"context_window": 32768}, chat_id="c2")
        host = build_test_host(rows, FakeAdapter(["unused"]))
        session = host.root.require("agentSessionStore").create(chat_id="c2")
        agent = host.root.require("agentLoop").create(session, {"model": "fake"})

        async def drive():
            return [f async for f in agent.run("my password is hunter2")]

        frames = asyncio.run(drive())
        self.assertEqual(frames[-1]["type"], "rejected")
        self.assertNotIn("step/start", [e["type"] for e in session.events])

    def test_extract_tool_calls_handles_multiple_and_unknown(self):
        text = ('```json\n{"tool":"a","arguments":{"x":1}}\n```\n'
                '```json\n{"tool":"b","arguments":{"y":2}}\n```\n'
                '```json\n{"tool":"zzz","arguments":{}}\n```')
        calls = extract_tool_calls(text, {"a", "b"})
        self.assertEqual([c["tool"] for c in calls], ["a", "b"])


class CompactionTests(unittest.TestCase):
    def test_condense_summarizes_old_span_and_keeps_tail(self):
        ctx = Context()
        session = Session("c")
        for i in range(20):
            session.append("user/message", text=f"question {i} " + "x" * 300)
            session.append("assistant/message", text=f"answer {i} " + "y" * 300)
        service = CompactionBasic(ctx, {"contextWindow": 3000, "maxTokens": 100,
                                        "headroomTokens": 100, "retainRatio": 0.1})
        adapter = FakeAdapter(["CONDENSED SUMMARY"])
        from dsh.llm import LLM, LLMPlugin  # noqa: F401
        llm_plugin = LLMPlugin(ctx, {"adapter": "openai-compatible"})
        ctx.get("llm").register_adapter("openai-compatible", adapter)
        self.assertTrue(service.over(session))
        result = asyncio.run(service.condense(session, config={"model": "fake"}))
        self.assertTrue(result["summary"])
        messages = session.derive_messages()
        self.assertIn("CONDENSED SUMMARY", messages[0]["content"])
        self.assertEqual(messages[-1]["content"], "answer 19 " + "y" * 300)
        self.assertLess(len(messages), 40)

    def test_prune_trims_long_tool_results_without_a_model_call(self):
        ctx = Context()
        session = Session("p")
        session.append("user/message", text="go")
        session.append("tool/result", tool="echo", content="z" * 5000)
        service = CompactionBasic(ctx, {"pruneToolResultChars": 100})
        pruned = service.prune_tool_results(session)
        self.assertEqual(pruned, 1)
        self.assertIn("(trimmed)", session.derive_messages()[-1]["content"])


class HarnessComponentTests(unittest.TestCase):
    def test_skills_register_a_catalog_section_and_loader_tool(self):
        harness = {
            "system_prompt": "Be helpful.", "guardrails": "", "memory": "",
            "tool_guidance": {"echo": "Use for short echoes."},
            "skills": [{"name": "review", "body": "Review the code.", "description": "code review"}],
            "scripts": [{"name": "plan", "body": "1. plan"}],
            "plugins": [{"name": "tone", "body": "Be direct."}],
            "subagents": [], "rules": {},
        }
        rows = runtime.build_rows(harness, {"context_window": 32768}, chat_id="c3")
        host = build_test_host(rows, FakeAdapter(["ok"]))
        prompt = host.root.require("systemPrompt")
        tools = host.root.require("tools")
        assembly = asyncio.run(prompt.assemble_async("agent", tools.schemas("agent")))
        text = prompt.render(assembly)
        self.assertIn("Skills you can load", text)
        self.assertIn("### script: plan", text)
        self.assertIn("Behavioural hooks", text)
        self.assertIn("Use for short echoes.", json.dumps(assembly.tools))
        self.assertIn("skill", [t["name"] for t in assembly.tools])

        async def call():
            return await tools.run("skill", {"name": "review"}, ExecContext())

        self.assertEqual(asyncio.run(call()).content, "Review the code.")

    def test_subagent_delegation_runs_a_child_agent(self):
        harness = {
            "system_prompt": "Be terse.", "guardrails": "", "memory": "",
            "tool_guidance": {}, "skills": [], "scripts": [], "plugins": [],
            "subagents": [{"name": "helper", "description": "helps",
                           "system": "You are the helper."}],
            "rules": {},
        }
        holder: dict[str, Any] = {}

        async def runner(**kw):
            return await runtime.run_subagent(holder["host"], params={"model": "fake"}, **kw)

        rows = runtime.build_rows(harness, {"context_window": 32768}, chat_id="c5",
                                  subagent_runner=runner)
        adapter = FakeAdapter([
            '```json\n{"tool":"helper","arguments":{"task":"say hi"}}\n```',
            "child says hi",
            "main done",
        ])
        host = build_test_host(rows, adapter)
        holder["host"] = host
        session = host.root.require("agentSessionStore").create(chat_id="c5")
        agent = host.root.require("agentLoop").create(session, {"model": "fake"})

        result = asyncio.run(_drive(agent))
        self.assertEqual(result["text"], "main done")
        self.assertEqual(result["tool_events"][0]["result"], "child says hi")
        self.assertEqual(len(adapter.calls), 3)

    def test_composition_report_lists_plugins_and_services(self):
        rows = runtime.build_rows({"system_prompt": "x", "rules": {}},
                                  {"context_window": 8192}, chat_id="c4")
        host = build_test_host(rows, FakeAdapter(["ok"]))
        report = runtime.composition_report(rows, host)
        names = [r["name"] for r in report["rows"]]
        self.assertIn("agent-loop", names)
        self.assertIn("compaction-basic", names)
        self.assertIn("tools", report["services"])
        self.assertFalse(report["failed"])


if __name__ == "__main__":
    unittest.main()
