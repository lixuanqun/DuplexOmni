"""Writer role of the Writer-Director pipeline (paper Sec. 4.1).

The Writer turns a :class:`~duplexomni.data.scenario.ScenarioSeed` into a
*natural dialogue script*: plain speaker-tagged lines (``[U]`` / ``[A]``)
without any temporal control tokens.  Backends:

* :class:`MockWriter` — deterministic offline templates (zh/en);
* :class:`LLMWriter` — prompts any :class:`~duplexomni.data.llm.LLMBackend`
  (paper uses Qwen3.5-397B-A27B).
"""

from __future__ import annotations

import random
from typing import Protocol

from .llm import LLMBackend, MockLLM
from .scenario import ScenarioSeed

__all__ = ["Writer", "MockWriter", "LLMWriter", "parse_script_lines"]

WRITER_SYSTEM = """You are the WRITER in a Writer-Director data pipeline for a
full-duplex spoken dialogue model. Given a scenario (topic, user persona,
language, target difficulty), write a short, natural, conversational script
(4-10 utterances) between the user [U] and the assistant [A].
Rules:
- Output ONLY speaker-tagged lines: "[U] ..." / "[A] ...", one utterance per line.
- Use the scenario's language. Casual spoken style, contractions, fillers.
- Do NOT add stage directions, timestamps or any control tokens; the Director
  will add them.
"""


class Writer(Protocol):
    def write(self, seed: ScenarioSeed) -> str: ...


class MockWriter:
    """Deterministic template writer for offline runs and tests."""

    _ZH_TEMPLATES = [
        (
            "[U] 你好，想请教一下{topic}方面的事，方便吗？\n"
            "[A] 方便呀，你说。\n"
            "[U] 是这样的，{topic}具体该怎么理解，能给我讲清楚一点吗？\n"
            "[A] 可以，这个稍等我理一下思路，先跟你确认几个前提。\n"
            "[U] 行，你慢慢想，不着急。\n"
            "[A] 简单说，核心是把它拆成两块来看，一块负责即时响应，一块负责深入处理。\n"
            "[U] 嗯嗯，听起来挺清楚的，那下一步呢？\n"
            "[A] 下一步你可以先试一个小例子，跑通整个流程再说。"
        ),
        (
            "[U] 帮我看看{topic}，我有点拿不准。\n"
            "[A] 好的，我先了解一下你的具体情况。\n"
            "[U] 主要就是时间不够用，节奏总被打断。\n"
            "[A] 明白了，我需要算一下几个方案的成本，你等我一小会儿。\n"
            "[U] 好，我先把材料发你。\n"
            "[A] 收到，我看完了，给你两个建议：先固定时间段，再减少并行任务。"
        ),
    ]

    _EN_TEMPLATES = [
        (
            "[U] hey, do you have a second? I wanted to ask about {topic}.\n"
            "[A] sure, go ahead.\n"
            "[U] so how does {topic} actually work in practice?\n"
            "[A] good question, let me get the details straight first.\n"
            "[U] take your time.\n"
            "[A] the short version: split it into a fast path and a slow path,\n"
            "and let them run in parallel.\n"
            "[U] right, that makes sense. what should I do next?\n"
            "[A] try a small end-to-end example first, then scale it up."
        ),
        (
            "[U] can you help me plan around {topic}?\n"
            "[A] of course, tell me the constraints.\n"
            "[U] mostly it is time, I keep getting interrupted.\n"
            "[A] okay, I need to check a couple of options, one moment.\n"
            "[U] sure, I will send the notes over.\n"
            "[A] thanks, got them. two suggestions: block fixed hours, and cut parallel tasks."
        ),
    ]

    def __init__(self, rng: random.Random | None = None) -> None:
        self.rng = rng or random.Random(0)

    def write(self, seed: ScenarioSeed) -> str:
        pool = self._ZH_TEMPLATES if seed.language == "zh" else self._EN_TEMPLATES
        template = pool[self.rng.randrange(len(pool))]
        topic = seed.topic if len(seed.topic) <= 24 else seed.topic[:24] + "…"
        return template.format(topic=topic) + "\n"


class LLMWriter:
    """LLM-backed writer (paper's Qwen3.5-397B-A27B role)."""

    def __init__(self, backend: LLMBackend) -> None:
        self.backend = backend

    def write(self, seed: ScenarioSeed) -> str:
        prompt = (
            f"Scenario id: {seed.scenario_id}\n"
            f"Language: {'Chinese' if seed.language == 'zh' else 'English'}\n"
            f"Topic: {seed.topic}\n"
            f"User persona: {seed.user_persona}\n"
            f"Difficulty (1 casual .. 3 needs deep reasoning): {seed.difficulty}\n"
            "Source snippet for inspiration:\n"
            f"{seed.source_snippet[:400]}\n"
        )
        out = self.backend.generate(prompt, system=WRITER_SYSTEM)
        return _strip_fences(out) + "\n"


def _strip_fences(text: str) -> str:
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines)


def parse_script_lines(script: str) -> list[tuple[str, str]]:
    """Fast structural parse: ``[(speaker_tag, text), ...]`` for QA/debug."""
    out: list[tuple[str, str]] = []
    for ln in script.splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        if ln[:3] in ("[U]", "[A]"):
            out.append((ln[:3], ln[3:].strip()))
        else:
            out.append(("", ln))
    return out


_ = MockLLM  # re-exported for convenience imports
