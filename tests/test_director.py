"""Director annotation tests."""

from __future__ import annotations

import random

import pytest

from duplexomni.data.checks import check_annotated_dialog
from duplexomni.data.director import RuleDirector, UtteranceKind
from duplexomni.data.scenario import Pattern, ScenarioSeed, sample_scenario_seeds
from duplexomni.data.writer import MockWriter
from duplexomni.tokens import (
    Cut,
    GhostText,
    OverlapOnset,
    ResultFragment,
    SharedSilence,
    Speaker,
    ThinkTrigger,
    Wait,
    parse_script,
)


def make_seed(patterns, *, lang="en", difficulty=3, sid="seed-x-0"):
    return ScenarioSeed(
        scenario_id=sid,
        topic="trip planning",
        language=lang,
        patterns=list(patterns),
        difficulty=difficulty,
    )


ALL_PATTERNS = list(Pattern)


@pytest.mark.parametrize("patterns", [ALL_PATTERNS, ALL_PATTERNS[:3], ALL_PATTERNS[3:]])
def test_rule_director_passes_checks(patterns, corpus_file):
    writer = MockWriter(random.Random(0))
    director = RuleDirector()
    seed = make_seed(patterns, sid=f"s-{len(patterns)}")
    dialog = director.annotate(writer.write(seed), seed)
    result = check_annotated_dialog(dialog)
    assert result.ok, result.errors


def test_delayed_reasoning_structure():
    director = RuleDirector()
    seed = make_seed([Pattern.DELAYED_REASONING])
    dialog = director.annotate(_script(), seed)
    events = parse_script(dialog.to_script())
    # [THINK] must precede its fragments
    kinds = [type(e) for e in events]
    assert ThinkTrigger in kinds
    assert ResultFragment in kinds
    assert kinds.index(ThinkTrigger) < kinds.index(ResultFragment)


def test_interruption_reset_structure():
    director = RuleDirector()
    seed = make_seed([Pattern.INTERRUPTION_RESET, Pattern.DELAYED_REASONING])
    dialog = director.annotate(_script(), seed)
    events = parse_script(dialog.to_script())
    kinds = [type(e) for e in events]
    assert Cut in kinds and Wait in kinds
    assert kinds.index(Cut) < kinds.index(Wait)
    # ghost text follows the cut within the same utterance
    assert any(isinstance(e, GhostText) for e in events)
    # overlap marker on the interrupting user turn
    assert any(isinstance(e, OverlapOnset) for e in events)


def test_shared_silence_event():
    director = RuleDirector()
    seed = make_seed([Pattern.SHARED_SILENCE])
    dialog = director.annotate(_script(), seed)
    silence = [u for u in dialog.utterances if u.kind == UtteranceKind.SILENCE]
    assert len(silence) == 1
    assert silence[0].duration_s >= 1.0
    assert any(isinstance(e, SharedSilence) for e in parse_script(dialog.to_script()))


def test_backchannel_is_overlap_and_short():
    director = RuleDirector()
    seed = make_seed([Pattern.BACKCHANNEL])
    dialog = director.annotate(_script(), seed)
    bc = [u for u in dialog.utterances if u.backchannel]
    assert bc, "backchannel pattern must instantiate"
    assert all(u.overlap for u in bc)
    assert all(u.duration_s <= 0.6 for u in bc)
    assert all(u.speaker is Speaker.USER for u in bc)


def test_assistant_initiated_prepends_opener():
    director = RuleDirector()
    seed = make_seed([Pattern.ASSISTANT_INITIATED])
    dialog = director.annotate(_script(), seed)
    first = dialog.utterances[0]
    assert first.speaker is Speaker.ASSISTANT


def test_no_think_no_fragments():
    director = RuleDirector()
    seed = make_seed([Pattern.OVERLAP])  # no delayed reasoning requested
    dialog = director.annotate(_script(), seed)
    events = parse_script(dialog.to_script())
    assert not any(isinstance(e, ResultFragment) for e in events)
    assert not any(isinstance(e, ThinkTrigger) for e in events)


def test_full_pipeline_samples_all_pass(corpus_file):
    from duplexomni.data import WriterDirectorPipeline

    pipe = WriterDirectorPipeline(corpus_file, seed=99)
    records, stats = pipe.run(n=40)
    assert stats.n_filtered == 0, stats.warnings
    assert len(records) == 40
    for rec in records:
        assert rec.slices, "every record has slices"
        assert rec.total_ms % 480 == 0 or rec.total_ms > 0


def _script():
    return (
        "[U] hey, can you help me plan a small trip this weekend?\n"
        "[A] sure, tell me your constraints and preferences.\n"
        "[U] two days, low budget, somewhere quiet nearby.\n"
        "[A] okay, let me work out an itinerary with costs for you.\n"
        "[U] great, take your time.\n"
        "[A] here is a two-day plan with a lakeside walk and a small town visit.\n"
    )


def test_seeds_from_corpus_annotate_cleanly(corpus_file):
    writer = MockWriter(random.Random(1))
    director = RuleDirector()
    for seed in sample_scenario_seeds(corpus_file, 25, seed=5):
        dialog = director.annotate(writer.write(seed), seed)
        result = check_annotated_dialog(dialog)
        assert result.ok, (seed.scenario_id, result.errors)
