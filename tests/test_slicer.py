"""Time-slicing tests: timeline, dual tracks, 480 ms slices, JSONL round-trip."""

from __future__ import annotations

import json

from duplexomni.config import RuntimeConfig
from duplexomni.data.director import AnnotatedDialog, AnnotatedUtterance, UtteranceKind
from duplexomni.data.scenario import Pattern, ScenarioSeed
from duplexomni.data.slicer import (
    DatasetRecord,
    build_timeline,
    iter_jsonl,
    slice_dialog,
    write_jsonl,
)
from duplexomni.data.synthesis import MockTTS
from duplexomni.tokens import Speaker

RUNTIME = RuntimeConfig()


def make_dialog(utterances, sid="s-0"):
    seed = ScenarioSeed(
        scenario_id=sid, topic="t", language="en",
        patterns=[Pattern.OVERLAP, Pattern.DELAYED_REASONING],
    )
    return AnnotatedDialog(seed=seed, utterances=utterances)


def test_timeline_monotonic_and_silence_advances():
    utterances = [
        AnnotatedUtterance(Speaker.USER, "hello there", duration_s=1.0),
        AnnotatedUtterance(Speaker.ASSISTANT, "[PEND2S]", kind=UtteranceKind.SILENCE, duration_s=2.0),
        AnnotatedUtterance(Speaker.ASSISTANT, "hi! how can I help", duration_s=1.5),
    ]
    tl = build_timeline(make_dialog(utterances))
    # the silence itself is not an entry, but it advances the clock
    assert len(tl.entries) == 2
    starts = [e["start_s"] for e in tl.entries]
    ends = [e["end_s"] for e in tl.entries]
    assert starts[0] == 0.0
    assert abs(starts[1] - (ends[0] + 2.0)) < 1e-6  # 2 s shared silence consumed
    assert tl.total_s == ends[1]
    assert any(e["event"] == "SharedSilence" for e in tl.timed_events)


def test_overlap_user_starts_inside_assistant_speech():
    utterances = [
        AnnotatedUtterance(Speaker.USER, "question one about the trip", duration_s=1.5),
        AnnotatedUtterance(Speaker.ASSISTANT, "a long answer that keeps going", duration_s=2.0),
        AnnotatedUtterance(Speaker.USER, "ˆ wait stop", duration_s=0.8, overlap=True),
        AnnotatedUtterance(Speaker.ASSISTANT, "yes?", duration_s=0.6),
    ]
    tl = build_timeline(make_dialog(utterances))
    a = tl.entries[1]
    u = tl.entries[2]
    assert u["start_s"] < a["end_s"], "overlapping user turn must start before assistant ends"
    assert u["start_s"] >= a["start_s"]


def test_backchannel_does_not_steal_the_floor():
    utterances = [
        AnnotatedUtterance(Speaker.USER, "question about the plan", duration_s=1.2),
        AnnotatedUtterance(Speaker.ASSISTANT, "a fairly long explanation", duration_s=2.0),
        AnnotatedUtterance(Speaker.USER, "yeah", duration_s=0.45, overlap=True, backchannel=True),
        AnnotatedUtterance(Speaker.ASSISTANT, "continuing the explanation", duration_s=1.5),
    ]
    tl = build_timeline(make_dialog(utterances))
    cont = tl.entries[3]
    bc = tl.entries[2]
    assert cont["start_s"] <= bc["end_s"] + 0.05, "assistant continues right after backchannel"


def test_slice_dialog_shapes_and_events():
    utterances = [
        AnnotatedUtterance(Speaker.USER, "hello there friend", duration_s=1.4),
        AnnotatedUtterance(Speaker.ASSISTANT, "sure, let me check [THINK] and keep talking", duration_s=2.4),
        AnnotatedUtterance(Speaker.ASSISTANT, "<step one>", kind=UtteranceKind.EVENT),
        AnnotatedUtterance(Speaker.USER, "ˆ wait", duration_s=0.5, overlap=True),
        AnnotatedUtterance(Speaker.ASSISTANT, "answer part [CUT] ghost tail [WAIT]", duration_s=1.2),
        AnnotatedUtterance(Speaker.ASSISTANT, "[PEND2S]", kind=UtteranceKind.SILENCE, duration_s=2.0),
    ]
    dialog = make_dialog(utterances, sid="slice-0")
    slices, tl = slice_dialog(dialog, MockTTS(), runtime=RUNTIME)

    n_frames = RUNTIME.codec.frames_per_slice
    k = RUNTIME.codec.num_codebooks
    assert slices, "slices produced"
    for s in slices:
        assert s.end_ms - s.start_ms == RUNTIME.slice_ms
        assert len(s.user_codes) == n_frames and len(s.user_codes[0]) == k
        assert len(s.assistant_codes) == n_frames and len(s.assistant_codes[0]) == k
    # total tiling covers the timeline
    assert slices[-1].end_ms >= int((tl.total_s - 1e-6) * 1000) - RUNTIME.slice_ms

    # user speech and assistant speech coexist in some slice (full duplex)
    assert any(s.user_speech and s.assistant_speech for s in slices)

    # events attached to their slices
    all_events = [e["event"] for s in slices for e in s.events]
    assert "ThinkTrigger" in all_events
    assert "SharedSilence" in all_events
    assert "Cut" in all_events

    # assistant text is aligned and non-empty somewhere
    assert any(s.target_text for s in slices)
    # user text captured
    assert any(s.user_text for s in slices)


def test_jsonl_roundtrip(tmp_path):
    utterances = [
        AnnotatedUtterance(Speaker.USER, "hi there friend", duration_s=1.2),
        AnnotatedUtterance(Speaker.ASSISTANT, "hello! how can I help you today", duration_s=1.8),
        AnnotatedUtterance(Speaker.ASSISTANT, "<frag>", kind=UtteranceKind.EVENT),
    ]
    dialog = make_dialog(utterances, sid="rt-0")
    slices, _tl = slice_dialog(dialog, MockTTS(), runtime=RUNTIME)
    rec = DatasetRecord(
        id="rt-0", language="en", patterns=["overlap"],
        annotated_script=dialog.to_script(), slices=slices,
        total_ms=slices[-1].end_ms,
    )
    path = tmp_path / "ds.jsonl"
    assert write_jsonl([rec], path) == 1
    loaded = list(iter_jsonl(path))
    assert len(loaded) == 1
    r = loaded[0]
    assert r.id == "rt-0"
    assert r.slices[-1].target_text == rec.slices[-1].target_text
    assert r.slices[0].user_codes == rec.slices[0].user_codes
    # JSON-serialisable end to end
    json.dumps(rec.to_dict())
