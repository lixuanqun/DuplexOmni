"""Behavioural benchmark harness (systems-level analogue of DuplexBench).

Runs scripted full-duplex scenarios through the interaction layer and scores
the runtime policy: floor discipline, turn-taking latency, async thinking
delivery, and interruption resets — plus the paper's RTF < 1 budget on every
slice.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from ..data import MockTTS
from ..runtime import (
    InteractionLayer,
    SimClock,
    ThinkingBridge,
    UserSliceInput,
)
from ..runtime.events import ThinkingFragment
from ..runtime.thinking import EchoThinkingLayer
from .scenarios import Scenario, UserTurn, default_scenarios

__all__ = ["CaseResult", "BenchReport", "run_benchmark", "format_report"]

# deterministic turn-taking knobs for the benchmark runtime
DEFAULT_LAYER_KWARGS: dict = {
    "tokens_per_slice": 4,
    "max_turn_slices": 3,
    "assistant_initiative_after": 2,
}

_PCM_SLICE_BYTES = 15360  # 480 ms of 16-bit mono @16 kHz


@dataclass
class CaseResult:
    scenario: str
    category: str
    passed: bool
    violations: list[str] = field(default_factory=list)
    fragments: int = 0
    max_rtf: float = 0.0
    n_slices: int = 0
    description: str = ""


@dataclass
class BenchReport:
    results: list[CaseResult]

    @property
    def overall(self) -> float:
        return sum(r.passed for r in self.results) / max(len(self.results), 1)

    def categories(self) -> dict[str, float]:
        cats: dict[str, list[bool]] = {}
        for r in self.results:
            cats.setdefault(r.category, []).append(r.passed)
        return {c: sum(ok) / len(ok) for c, ok in cats.items()}

    def as_dict(self) -> dict:
        return {
            "overall": self.overall,
            "categories": self.categories(),
            "results": [r.__dict__ for r in self.results],
        }


async def _warm_up(model, with_thinking: bool) -> list:
    """One throwaway session slice so lazy allocations and cache buffers are
    in place before measurement — the deployment practice the RTF<1 budget
    assumes (paper Sec. 3.4: KV-cache + graph-optimised steady state)."""
    import torch

    torch.manual_seed(0)
    bridge = ThinkingBridge(EchoThinkingLayer(delay_s=0.0, n_fragments=1)) if with_thinking else None
    layer = InteractionLayer(model, bridge=bridge, clock=SimClock(), **DEFAULT_LAYER_KWARGS)

    async def stream():
        yield UserSliceInput(0, b"\x00" * _PCM_SLICE_BYTES)

    return [out async for out in layer.run(stream())]


def run_benchmark(
    model,
    scenarios: list[Scenario] | None = None,
    *,
    seed: int = 0,
    layer_kwargs: dict | None = None,
    thinking: dict | None = None,
    verbose: bool = False,
) -> BenchReport:
    """Run every scenario; ``thinking`` overrides scenario thinking configs
    (used by the ablation to swap thinking-layer strength)."""
    scenarios = scenarios if scenarios is not None else default_scenarios()
    results = [
        run_case(model, sc, seed=seed, layer_kwargs=layer_kwargs, thinking=thinking)
        for sc in scenarios
    ]
    report = BenchReport(results=results)
    if verbose:
        print(format_report(report))
    return report


def run_case(
    model,
    scenario: Scenario,
    *,
    seed: int = 0,
    layer_kwargs: dict | None = None,
    thinking: dict | None = None,
) -> CaseResult:
    thinking_cfg = thinking if thinking is not None else scenario.thinking
    return asyncio.run(
        _run_case_async(model, scenario, seed=seed, layer_kwargs=layer_kwargs, thinking=thinking_cfg)
    )


async def _run_case_async(
    model,
    scenario: Scenario,
    *,
    seed: int,
    layer_kwargs: dict | None,
    thinking: dict | None,
) -> CaseResult:
    import torch

    _ = await _warm_up(model, thinking is not None)
    torch.manual_seed(seed)

    bridge = None
    if thinking is not None:
        bridge = ThinkingBridge(EchoThinkingLayer(**thinking))

    merged = {**DEFAULT_LAYER_KWARGS, **(layer_kwargs or {})}
    layer = InteractionLayer(model, bridge=bridge, clock=SimClock(), **merged)

    tts = MockTTS()
    pcm_cache = {
        t.text: tts.synthesize(t.text, speaker="user").pcm for t in scenario.turns
    }
    turns_by_slice = {t.slice_index: t for t in scenario.turns}

    async def user_stream():
        if bridge is not None and thinking is not None:
            # seed the reasoning request right before slice 0 (a proactive
            # thinking layer), so its fragments race the conversation
            from ..runtime import SessionState

            bridge.request(SessionState().build_context(), query="help with the user's last request")
        for i in range(scenario.n_slices):
            turn: UserTurn | None = turns_by_slice.get(i)
            if turn is None:
                yield UserSliceInput(i, b"\x00" * _PCM_SLICE_BYTES)
                continue
            pcm = pcm_cache[turn.text][: _PCM_SLICE_BYTES] if turn.loud else b"\x00" * _PCM_SLICE_BYTES
            yield UserSliceInput(i, pcm, asr_text=turn.text)

    outputs = [o async for o in layer.run(user_stream())]

    violations = scenario.check(outputs, scenario)
    fragments = sum(len(o.events_of(ThinkingFragment)) for o in outputs)
    max_rtf = max((o.rtf for o in outputs), default=0.0)
    return CaseResult(
        scenario=scenario.name,
        category=scenario.category,
        passed=not violations,
        violations=violations,
        fragments=fragments,
        max_rtf=max_rtf,
        n_slices=len(outputs),
        description=scenario.description,
    )


def format_report(report: BenchReport) -> str:
    lines = ["full-duplex behavioural benchmark", "=" * 64]
    for r in report.results:
        status = "PASS" if r.passed else "FAIL"
        lines.append(
            f"[{status}] {r.scenario:<20} cat={r.category:<18} "
            f"frag={r.fragments:<3} max_rtf={r.max_rtf:.2f}"
        )
        if not r.passed:
            for v in r.violations:
                lines.append(f"       - {v}")
    lines.append("-" * 64)
    for cat, rate in report.categories().items():
        lines.append(f"{cat:<24} {rate:.0%}")
    lines.append(f"{'OVERALL':<24} {report.overall:.0%}")
    return "\n".join(lines)
