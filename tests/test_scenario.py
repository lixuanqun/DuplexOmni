"""Scenario-seed sampling tests (Stage 1)."""

from __future__ import annotations

import random

from duplexomni.data.scenario import (
    PAPER_PATTERN_PROBS,
    Pattern,
    detect_language,
    pattern_coverage_stats,
    sample_patterns,
    sample_scenario_seeds,
    topic_from_text,
)


def test_language_detection(corpus_file):
    assert detect_language("今天天气很好") == "zh"
    assert detect_language("plain english text") == "en"


def test_topic_extraction_short():
    assert topic_from_text("how to cook rice\nmore detail") == "how to cook rice"
    zh = topic_from_text("如何做一道简单的番茄炒蛋？下面是详细步骤")
    assert "番茄炒蛋" in zh and len(zh) <= 16


def test_sample_patterns_guarantees_one():
    rng = random.Random(0)
    for _ in range(50):
        assert len(sample_patterns(rng)) >= 1


def test_pattern_probs_match_paper():
    assert PAPER_PATTERN_PROBS[Pattern.DELAYED_REASONING] == pytest_approx(0.943)
    assert PAPER_PATTERN_PROBS[Pattern.BACKCHANNEL] == pytest_approx(0.031)


def pytest_approx(x):
    import pytest

    return pytest.approx(x)


def test_seeds_deterministic_and_stats(corpus_file):
    a = sample_scenario_seeds(corpus_file, 200, seed=11)
    b = sample_scenario_seeds(corpus_file, 200, seed=11)
    assert [s.scenario_id for s in a] == [s.scenario_id for s in b]
    assert [sorted(p.value for p in s.patterns) for s in a] == [
        sorted(p.value for p in s.patterns) for s in b
    ]

    stats = pattern_coverage_stats(a)
    # coverage should be near the paper's probabilities (loose bounds)
    assert 0.85 <= stats["delayed_reasoning"] <= 1.0
    assert 0.55 <= stats["shared_silence"] <= 0.8
    assert stats["multi_pattern>=2"] > 0.8
    assert 0.0 <= stats["backchannel"] <= 0.1
    # languages from the corpus appear
    langs = {s.language for s in a}
    assert langs == {"zh", "en"}


def test_seed_roundtrip_dict(corpus_file):
    s = sample_scenario_seeds(corpus_file, 1, seed=3)[0]
    from duplexomni.data.scenario import ScenarioSeed

    s2 = ScenarioSeed.from_dict(s.to_dict())
    assert s2.patterns == s.patterns
    assert s2.topic == s.topic
