"""Thinking layer: the pluggable reasoning module (paper Sec. 3.1).

The thinking layer is any async, streaming "LLM/agent" that receives a
non-blocking request from the interaction layer and streams result
fragments back.  It must be *abortable*: when the interaction layer raises
``[WAIT]`` (e.g. the user interrupted), the in-flight request is cancelled
and its un-streamed fragments are dropped.

Backends:

* :class:`EchoThinkingLayer` — deterministic offline reasoning simulator;
* :class:`FunctionThinkingLayer` — wraps any (async) callable;
* :class:`OpenAICompatThinkingLayer` — any OpenAI-compatible chat endpoint
  (stdlib urllib; fragments are re-streamed token-by-token to emulate SSE).
"""

from __future__ import annotations

import asyncio
import inspect
import json
import urllib.request
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class ThinkingContext:
    """What the interaction layer sends along with a [THINK] request.

    Paper Sec. 3.1: "sends the context (dialogue text, video info, task
    state) to the thinking layer as a non-blocking request".
    """

    dialogue_text: list[tuple[str, str]] = field(default_factory=list)  # (speaker, text)
    video_info: str = ""
    task_state: dict = field(default_factory=dict)

    def summary(self) -> str:
        recent = " | ".join(f"{sp}: {t[:40]}" for sp, t in self.dialogue_text[-6:])
        return f"dialogue[{recent}] video[{self.video_info[:40]}] task[{self.task_state}]"


@dataclass
class ThinkingRequest:
    request_id: str
    context: ThinkingContext
    query: str = ""


class ThinkingLayer(Protocol):
    def think(self, request: ThinkingRequest) -> AsyncIterator[str]: ...


class EchoThinkingLayer:
    """Deterministic offline reasoning: echoes a multi-fragment "analysis".

    Fragments are emitted with small asyncio delays so tests can observe
    true asynchronous overlap with the interaction loop.
    """

    def __init__(self, *, delay_s: float = 0.01, n_fragments: int = 3) -> None:
        self.delay_s = delay_s
        self.n_fragments = n_fragments
        self.requests: list[ThinkingRequest] = []

    async def think(self, request: ThinkingRequest) -> AsyncIterator[str]:
        self.requests.append(request)
        for i in range(self.n_fragments):
            await asyncio.sleep(self.delay_s)
            yield f"[{request.request_id}#{i}] analysis of {request.query or 'the request'}"


class FunctionThinkingLayer:
    """Wrap a callable into a streaming thinking layer.

    The callable may be sync (str -> str) or async (str -> AsyncIterator[str]
    or str).  Results are chunked into fragments.
    """

    def __init__(self, fn: Callable[..., Any], *, chunk_size: int = 24) -> None:
        self.fn = fn
        self.chunk_size = chunk_size

    async def think(self, request: ThinkingRequest) -> AsyncIterator[str]:
        prompt = request.query or request.context.summary()
        result = self.fn(prompt)
        if inspect.isawaitable(result):
            result = await result
        if inspect.isasyncgen(result):
            async for fragment in result:
                yield fragment
            return
        text = str(result)
        for i in range(0, len(text), self.chunk_size):
            await asyncio.sleep(0)
            yield text[i: i + self.chunk_size]


class OpenAICompatThinkingLayer:
    """OpenAI-compatible endpoint as the thinking layer (paper default:
    Gemini-3.1-Flash-Lite; any strong LLM/agent works)."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "",
        *,
        system: str = "You are the thinking layer of a full-duplex voice assistant. Reason step by step, then answer concisely.",
        temperature: float = 0.2,
        max_tokens: int = 1024,
        timeout_s: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.system = system
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s

    async def think(self, request: ThinkingRequest) -> AsyncIterator[str]:
        messages = [
            {"role": "system", "content": self.system},
            {
                "role": "user",
                "content": (
                    f"Context:\n{json.dumps(request.context.dialogue_text[-10:], ensure_ascii=False)}\n"
                    f"Video: {request.context.video_info or 'none'}\n"
                    f"Task state: {request.context.task_state}\n"
                    f"Request: {request.query}"
                ),
            },
        ]
        payload = json.dumps(
            {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "stream": False,
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}),
            },
        )
        loop = asyncio.get_running_loop()

        def _call() -> str:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            return body["choices"][0]["message"]["content"]

        text = await loop.run_in_executor(None, _call)
        for word in text.split():
            yield word + " "
