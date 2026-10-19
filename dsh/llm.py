"""The model adapter seam (ctx.llm).

Adapters are swappable; the loop never imports a provider. `prepare_call`
validates the routed config once, `stream` yields normalized chunks, and provider
errors are classified so the compaction plugin can distinguish a genuine context
overflow from every other failure.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable

__all__ = ["LLM", "LLMPlugin", "StreamChunk", "PreparedCall", "LLMError",
           "NoAdapter", "ContextOverflow", "estimate_tokens", "OpenAICompatibleAdapter"]


class LLMError(RuntimeError):
    pass


class NoAdapter(LLMError):
    pass


class ContextOverflow(LLMError):
    pass


@dataclass
class StreamChunk:
    kind: str                 # text | reasoning | usage | done
    text: str = ""
    usage: dict[str, int] = field(default_factory=dict)
    finish_reason: str = ""


@dataclass
class PreparedCall:
    adapter: str
    model: str
    params: dict[str, Any]
    meta: dict[str, Any] = field(default_factory=dict)


def estimate_tokens(messages: list[dict[str, Any]]) -> int:
    chars = 0
    for message in messages:
        chars += len(str(message.get("content") or "")) + 8
    return (chars + 3) // 4


OVERFLOW_MARKERS = (
    "context length", "context window", "maximum context", "too many tokens",
    "exceeds the maximum", "input is too long", "context_length_exceeded",
)


class LLM:
    """ctx.llm — the adapter seam."""

    def __init__(self, ctx: Any, config: dict[str, Any] | None = None) -> None:
        self.ctx = ctx
        self.config = dict(config or {})
        self._adapters: dict[str, Any] = {}

    def register_adapter(self, name: str, adapter: Any) -> Callable[[], None]:
        if name in self._adapters:
            raise ValueError(f"adapter {name!r} already registered")
        self._adapters[name] = adapter

        def dispose() -> None:
            if self._adapters.get(name) is adapter:
                self._adapters.pop(name, None)

        return dispose

    def prepare_call(self, config: dict[str, Any]) -> PreparedCall:
        name = config.get("adapter") or self.config.get("adapter") or "openai-compatible"
        adapter = self._adapters.get(name)
        if adapter is None:
            raise NoAdapter(f"no adapter registered for {name!r}")
        model = config.get("model")
        if not model:
            raise NoAdapter("no model routed for this request")
        params = {k: v for k, v in config.items()
                  if k not in ("adapter", "model", "provider")}
        prepared = PreparedCall(adapter=name, model=str(model), params=params)
        if hasattr(adapter, "prepare"):
            adapter.prepare(prepared)
        return prepared

    async def stream(self, prepared: PreparedCall, messages: list[dict[str, Any]],
                     tools: list[dict[str, Any]] | None,
                     signal: asyncio.Event | None = None) -> AsyncIterator[StreamChunk]:
        adapter = self._adapters.get(prepared.adapter)
        if adapter is None:
            raise NoAdapter(f"adapter {prepared.adapter!r} disappeared")
        try:
            async for chunk in adapter.stream(prepared, messages, tools or [], signal):
                yield chunk
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            text = f"{type(exc).__name__}: {exc}"
            lowered = text.lower()
            if any(marker in lowered for marker in OVERFLOW_MARKERS):
                raise ContextOverflow(text) from exc
            raise LLMError(text) from exc


class OpenAICompatibleAdapter:
    """SGLang / vLLM / llama.cpp / OpenAI: one streaming wire format."""

    def __init__(self, base_url: str, *, api_key: str = "",
                 timeout: tuple[float, float] = (10.0, 600.0)) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def prepare(self, prepared: PreparedCall) -> None:
        if not prepared.params.get("max_tokens"):
            prepared.params["max_tokens"] = 512

    def _body(self, prepared: PreparedCall, messages: list[dict[str, Any]],
              tools: list[dict[str, Any]]) -> dict[str, Any]:
        params = prepared.params
        body: dict[str, Any] = {
            "model": prepared.model,
            "messages": messages,
            "max_tokens": int(params.get("max_tokens") or 512),
            "temperature": float(params.get("temperature", 0.7)),
            "top_p": float(params.get("top_p", 0.95)),
            "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": bool(params.get("enable_thinking", False))},
        }
        if params.get("top_k"):
            body["top_k"] = int(params["top_k"])
        if params.get("repeat_penalty"):
            body["repetition_penalty"] = float(params["repeat_penalty"])
        if params.get("seed") not in (None, ""):
            body["seed"] = int(params["seed"])
        if params.get("stop"):
            stop = params["stop"]
            body["stop"] = stop if isinstance(stop, list) else [stop]
        # The workbench speaks the JSON tool protocol in the prompt; native
        # `tools` are only forwarded when the route declares native support.
        if tools and params.get("native_tools"):
            body["tools"] = [
                {"type": "function",
                 "function": {"name": t["name"], "description": t.get("description", ""),
                              "parameters": t.get("input_schema") or {"type": "object"}}}
                for t in tools
            ]
        return body

    async def stream(self, prepared: PreparedCall, messages: list[dict[str, Any]],
                     tools: list[dict[str, Any]],
                     signal: asyncio.Event | None = None) -> AsyncIterator[StreamChunk]:
        import httpx

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        timeout = httpx.Timeout(connect=self.timeout[0], read=self.timeout[1],
                                write=30.0, pool=10.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream("POST", f"{self.base_url}/v1/chat/completions",
                                     json=self._body(prepared, messages, tools),
                                     headers=headers) as resp:
                if resp.status_code != 200:
                    detail = (await resp.aread()).decode()[:800]
                    raise LLMError(f"upstream {resp.status_code}: {detail}")
                async for line in resp.aiter_lines():
                    if signal is not None and signal.is_set():
                        raise asyncio.CancelledError()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    if obj.get("usage"):
                        yield StreamChunk("usage", usage={
                            "prompt_tokens": int(obj["usage"].get("prompt_tokens") or 0),
                            "completion_tokens": int(obj["usage"].get("completion_tokens") or 0),
                        })
                    choice = (obj.get("choices") or [{}])[0]
                    delta = choice.get("delta") or {}
                    if delta.get("reasoning_content"):
                        yield StreamChunk("reasoning", text=delta["reasoning_content"])
                    if delta.get("content"):
                        yield StreamChunk("text", text=delta["content"])
                    if choice.get("finish_reason"):
                        yield StreamChunk("done", finish_reason=choice["finish_reason"])


class LLMPlugin:
    name = "llm"
    inject: tuple[str, ...] = ()

    def __init__(self, ctx: Any, config: dict[str, Any] | None = None) -> None:
        self.ctx = ctx
        self.config = dict(config or {})
        self.llm = LLM(ctx, self.config)
        ctx.provide("llm", self.llm)

    def apply(self) -> None:
        pass
