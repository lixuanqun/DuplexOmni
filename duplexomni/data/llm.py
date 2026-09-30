"""Pluggable LLM backends for the Writer-Director data pipeline.

The paper uses Qwen3.5-397B-A27B for generation/annotation.  This module
abstracts that behind :class:`LLMBackend` so the pipeline can run with:

* :class:`MockLLM` — deterministic, offline (tests, examples);
* :class:`OpenAICompatLLM` — any OpenAI-compatible endpoint (vLLM, DashScope,
  OpenRouter, local ``llama.cpp-server`` ...), implemented with the standard
  library only.
"""

from __future__ import annotations

import json
import urllib.request
from typing import Protocol


class LLMBackend(Protocol):
    """Minimal chat-completion interface used by Writer / Director."""

    def generate(self, prompt: str, *, system: str | None = None) -> str: ...


class MockLLM:
    """Deterministic offline backend.

    It does not produce free-form text; instead it recognises the pipeline's
    prompts (writer / director modes) and returns templated content seeded by
    the prompt itself.  This keeps unit tests and the offline demo fully
    reproducible without network access.
    """

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        sys = system or ""
        if "WRITER" in sys or "script" in sys.lower():
            return self._writer_reply(prompt)
        if "DIRECTOR" in sys or "annotat" in sys.lower():
            return self._director_reply(prompt)
        return f"[mock-llm] {prompt[:120]}"

    # -- templates ----------------------------------------------------------- #

    @staticmethod
    def _writer_reply(prompt: str) -> str:
        topic = _first_quoted(prompt) or "the topic"
        return (
            "[U] hi, I wanted to ask about " + topic + ", do you have a moment?\n"
            "[A] of course, fire away.\n"
            "[U] so, what is the key idea behind " + topic + "?\n"
            "[A] sure, let me check that properly and get back to you.\n"
            "[U] great, take your time.\n"
            "[A] the key idea is that listening, thinking and speaking can run concurrently.\n"
            "[U] hmm right, that makes sense, thanks!\n"
            "[A] happy to help, ask me anything else."
        )

    @staticmethod
    def _director_reply(prompt: str) -> str:
        return (
            "[U] hi, I wanted to ask about the plan for tomorrow, do you have a moment? ˆ\n"
            "[A] of course, fire away.\n"
            "[U] so, can you compute the exact travel budget for three people? \n"
            "[A] sure, let me check that properly [THINK] and get back to you, "
            "I will verify the numbers meanwhile.\n"
            "<step one: list the fixed costs>\n"
            "<step two: add the per-person variable costs>\n"
            "[A] the total comes to about two thousand, and here is the breakdown.\n"
            "[U] wait, actually make it four people ˆ\n"
            "[A] [CUT] here is the two thousand breakdown, transport, hotel, food "
            "and the museum tickets [WAIT]\n"
            "[PEND2S]\n"
            "[A] re-computed for four people it is about twenty six hundred."
        )


def _first_quoted(text: str) -> str | None:
    import re

    m = re.search(r"['\"“]([^'\"“”]{2,80})['\"”]", text)
    return m.group(1).strip() if m else None


class OpenAICompatLLM:
    """Chat backend for any OpenAI-compatible ``/chat/completions`` endpoint.

    Parameters
    ----------
    base_url:
        e.g. ``https://dashscope.aliyuncs.com/compatible-mode/v1``.
    model:
        e.g. ``qwen3.5-397b-a27b`` (paper) or any local model.
    api_key:
        Bearer token.  May also be provided via ``env`` by the caller.
    temperature, max_tokens:
        Generation parameters (annotation favours low temperature).
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "",
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        timeout_s: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = json.dumps(
            {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
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
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return body["choices"][0]["message"]["content"]
