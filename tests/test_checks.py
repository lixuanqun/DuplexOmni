"""Consistency-check tests (the pipeline's filtering gate)."""

from __future__ import annotations

from duplexomni.data.checks import check_annotated_dialog, filter_dialogs
from duplexomni.data.director import AnnotatedDialog, AnnotatedUtterance, UtteranceKind
from duplexomni.data.scenario import Pattern, ScenarioSeed
from duplexomni.tokens import Speaker


def make_dialog(lines, patterns=None, sid="t-0"):
    """lines: list of (speaker, text, kind) triples (kind optional)."""
    utterances = []
    for item in lines:
        sp, text, *rest = item
        kind = rest[0] if rest else UtteranceKind.SPEECH
        utterances.append(
            AnnotatedUtterance(
                speaker=sp,
                text=text,
                kind=kind,
                duration_s=1.0 if kind == UtteranceKind.SPEECH else 0.0,
                overlap="ˆ" in text,
            )
        )
    seed = ScenarioSeed(
        scenario_id=sid, topic="t", language="en",
        patterns=patterns if patterns is not None else [Pattern.DELAYED_REASONING],
    )
    return AnnotatedDialog(seed=seed, utterances=utterances)


GOOD = [
    (Speaker.USER, "hello can you help me"),
    (Speaker.ASSISTANT, "sure let me check [THINK] and keep talking"),
    (Speaker.ASSISTANT, "<first result>", UtteranceKind.EVENT),
    (Speaker.USER, "ˆ wait a second"),
    (Speaker.ASSISTANT, "so I was saying [CUT] the rest is ghost [WAIT]"),
    (Speaker.ASSISTANT, "[PEND2S]", UtteranceKind.SILENCE),
    (Speaker.ASSISTANT, "here is the final answer"),
]


def test_good_dialog_passes():
    res = check_annotated_dialog(make_dialog(GOOD))
    assert res.ok, res.errors


def test_orphan_fragment_rejected():
    lines = [
        (Speaker.USER, "hi"),
        (Speaker.ASSISTANT, "<result with no think>", UtteranceKind.EVENT),
        (Speaker.ASSISTANT, "answer"),
    ]
    res = check_annotated_dialog(make_dialog(lines, patterns=[Pattern.OVERLAP]))
    assert not res.ok
    assert any("orphan" in e for e in res.errors)


def test_unresolved_think_rejected():
    lines = [
        (Speaker.USER, "hi"),
        (Speaker.ASSISTANT, "let me think [THINK]"),
        (Speaker.ASSISTANT, "never mind, no result ever arrives"),
    ]
    res = check_annotated_dialog(make_dialog(lines))
    assert not res.ok
    assert any("never resolved" in e for e in res.errors)


def test_think_aborted_by_wait_is_resolved():
    lines = [
        (Speaker.USER, "hi"),
        (Speaker.ASSISTANT, "thinking [THINK]"),
        (Speaker.ASSISTANT, "[WAIT]"),
        (Speaker.ASSISTANT, "dropped that thread"),
    ]
    res = check_annotated_dialog(make_dialog(lines))
    assert res.ok, res.errors


def test_missing_speaker_rejected():
    lines = [(Speaker.USER, "only the user speaks here, no assistant")]
    res = check_annotated_dialog(make_dialog(lines))
    assert not res.ok


def test_malformed_silence_rejected():
    lines = GOOD + [(Speaker.ASSISTANT, "[PENDTWO S]", UtteranceKind.SILENCE)]
    res = check_annotated_dialog(make_dialog(lines))
    assert not res.ok


def test_no_pattern_at_all_warns_and_errors_without_any():
    lines = [
        (Speaker.USER, "hi"),
        (Speaker.ASSISTANT, "hello"),
    ]
    res = check_annotated_dialog(make_dialog(lines, patterns=[]))
    assert not res.ok  # nothing instantiated
    assert any("no interaction pattern" in e for e in res.errors)


def test_backchannel_without_overlap_rejected():
    lines = [
        (Speaker.USER, "hi"),
        (Speaker.ASSISTANT, "so anyway, a long explanation follows here"),
        (Speaker.USER, "yeah"),  # backchannel shape but overlap=False
    ]
    dialog = make_dialog(lines, patterns=[Pattern.BACKCHANNEL])
    dialog.utterances[2].backchannel = True
    dialog.utterances[2].duration_s = 0.4
    res = check_annotated_dialog(dialog)
    assert not res.ok
    assert any("backchannel" in e for e in res.errors)


def test_filter_dialogs_stats():
    good = make_dialog(GOOD, sid="g")
    bad = make_dialog(
        [(Speaker.USER, "hi"), (Speaker.ASSISTANT, "<orphan>", UtteranceKind.EVENT)],
        sid="b",
    )
    kept, results, stats = filter_dialogs([good, bad])
    assert [d.seed.scenario_id for d in kept] == ["g"]
    assert stats["n_kept"] == 1 and stats["keep_rate"] == 0.5
    assert stats["n_errors"] >= 1
