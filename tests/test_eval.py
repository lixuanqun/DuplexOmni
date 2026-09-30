"""Behavioural benchmark + thinking-layer ablation tests (paper Sec. 5)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from duplexomni.config import tiny_config  # noqa: E402
from duplexomni.eval import (  # noqa: E402
    format_ablation,
    format_report,
    run_benchmark,
    run_thinking_ablation,
)
from duplexomni.model import DuplexOmni  # noqa: E402


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return DuplexOmni(tiny_config())


def test_benchmark_all_scenarios_pass(model):
    report = run_benchmark(model)
    assert report.results, "scenarios ran"
    assert report.overall == 1.0, [
        (r.scenario, r.violations) for r in report.results if not r.passed
    ]
    # all four behavioural categories are covered
    assert set(report.categories()) == {
        "interruption", "turn_taking", "delayed_reasoning", "interruption_reset",
    }
    # the RTF budget held on every slice of every scenario
    assert all(r.max_rtf < 1.0 for r in report.results)


def test_report_is_human_readable(model):
    report = run_benchmark(model)
    text = format_report(report)
    assert "PASS" in text and "OVERALL" in text
    assert "100%" in text


def test_floor_discipline_fails_without_policy(model):
    """Sanity: the checks really test the runtime — a deliberately broken
    expectation (assistant must stay silent forever) must fail."""
    from duplexomni.eval.scenarios import Scenario, floor_discipline

    sc = floor_discipline()

    def never_speak(outputs, sc):
        return ["injected failure"] if any(o.assistant_speaking for o in outputs) else []

    broken = Scenario(
        name="broken", category=sc.category, n_slices=sc.n_slices,
        turns=list(sc.turns), check=never_speak,
    )
    report = run_benchmark(model, [broken])
    assert report.overall == 0.0


def test_ablation_policy_independent_of_thinking_strength(model):
    """Paper Sec. 5: the full-duplex behaviour does not depend on the
    thinking layer's strength; fragment volume does."""
    result = run_thinking_ablation(model)
    assert result["policy_independent_of_thinking"], format_ablation(result)
    assert result["fragments_scale_with_strength"], format_ablation(result)
    variants = result["variants"]
    assert variants["none"].fragments == 0
    assert variants["weak"].fragments < variants["strong"].fragments
    assert all(v.max_rtf < 1.0 for v in variants.values())


def test_benchmark_deterministic(model):
    """Behavioural outcomes are deterministic.

    ``fragments`` is excluded: whether an in-flight fragment lands before an
    abort depends on wall-clock timing (by design — that is what async
    delivery means), so only the abort *decision* is asserted deterministic.
    """
    a = run_benchmark(model, seed=0)
    b = run_benchmark(model, seed=0)

    def behavioural(report):
        return [
            (r.scenario, r.passed, tuple(r.violations), r.n_slices)
            for r in report.results
        ]

    assert behavioural(a) == behavioural(b)
    assert a.overall == b.overall == 1.0
