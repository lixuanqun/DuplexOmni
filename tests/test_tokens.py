"""Control-token grammar and streaming parser tests."""

from __future__ import annotations

from duplexomni.tokens import (
    ControlTokenParser,
    Cut,
    GhostText,
    OverlapOnset,
    ResultFragment,
    SharedSilence,
    Speaker,
    TextDelta,
    ThinkTrigger,
    TurnStart,
    Wait,
    iter_turns,
    parse_script,
)

FULL_SCRIPT = (
    "[U] hi there ˆ wait actually\n"
    "[A] sure, let me check [THINK] and I will keep talking meanwhile\n"
    "<step one done>\n"
    "[A] hmm [CUT] this part is never spoken [WAIT]\n"
    "[PEND2S]\n"
    "[A] okay here is the answer\n"
)


def types(events):
    return [type(e).__name__ for e in events]


def test_batch_parse_all_tokens():
    events = parse_script(FULL_SCRIPT)
    kinds = types(events)
    assert kinds.count("TurnStart") == 4
    assert ThinkTrigger in [type(e) for e in events]
    assert any(isinstance(e, ResultFragment) and e.text == "step one done" for e in events)
    assert any(isinstance(e, Cut) for e in events)
    assert any(isinstance(e, Wait) for e in events)
    assert any(isinstance(e, OverlapOnset) for e in events)
    assert any(isinstance(e, SharedSilence) and e.seconds == 2 for e in events)


def test_incremental_equals_batch():
    """Char-by-char streaming must be semantically identical to batch parse."""
    batch = parse_script(FULL_SCRIPT)
    parser = ControlTokenParser()
    inc = []
    for ch in FULL_SCRIPT:
        inc.extend(parser.feed(ch))
    inc.extend(parser.close())

    def signature(events):
        plain = (TextDelta, GhostText)
        controls = [type(e).__name__ for e in events if not isinstance(e, plain)]
        text = "".join(
            getattr(e, "text", "") for e in events if isinstance(e, plain)
        )
        return controls, text

    assert signature(batch) == signature(inc)
    # batch mode packs plain runs into single events
    assert any(isinstance(e, TextDelta) and len(e.text) > 2 for e in batch)


def test_ghost_text_after_cut():
    events = parse_script("[A] spoken part [CUT] ghost tail [WAIT]")
    texts = [(type(e).__name__, getattr(e, "text", "")) for e in events]
    assert ("TextDelta", " spoken part ") in texts
    assert ("GhostText", " ghost tail ") in texts
    assert not any(isinstance(e, GhostText) and "spoken" in e.text for e in events)


def test_ghost_resets_at_next_turn():
    events = parse_script("[A] a [CUT] ghost [U] next")
    after_user = False
    ghosts = []
    for e in events:
        if isinstance(e, TurnStart) and e.speaker.value == "user":
            after_user = True
        if isinstance(e, GhostText) and after_user:
            ghosts.append(e)
    assert ghosts == []  # no ghost leakage across turns


def test_pend_float_and_bounds():
    assert any(
        isinstance(e, SharedSilence) and abs(e.seconds - 1.5) < 1e-9
        for e in parse_script("[PEND1.5S]")
    )
    # malformed PEND falls through as literal text, never crashes
    events = parse_script("[PENDX]")
    assert any(isinstance(e, TextDelta) for e in events)


def test_bracket_literal_text_survives():
    events = parse_script("[A] array indexing looks like xs[0] and [weird")
    text = "".join(e.text for e in events if isinstance(e, (TextDelta, GhostText)))
    assert "xs[0]" in text
    assert "[weird" in text


def test_result_fragment_content_is_raw():
    events = parse_script("[A] [THINK] ok <[not] a [THINK] token>")
    frags = [e for e in events if isinstance(e, ResultFragment)]
    assert len(frags) == 1
    assert frags[0].text == "[not] a [THINK] token"


def test_unterminated_angle_flushes_on_close():
    parser = ControlTokenParser()
    events = parser.feed("[A] think <half") + parser.close()
    text = "".join(e.text for e in events if isinstance(e, TextDelta))
    assert "<half" in text  # recoverable, flagged by the consistency checks


def test_strict_paper_grammar_rejects_speaker_tags():
    events = parse_script("[U] hi", strict_paper_grammar=True)
    assert not any(isinstance(e, TurnStart) for e in events)


def test_overlap_alias():
    assert any(isinstance(e, OverlapOnset) for e in parse_script("a^b"))
    assert any(isinstance(e, OverlapOnset) for e in parse_script("aˆb"))


def test_iter_turns():
    turns = list(iter_turns(parse_script(FULL_SCRIPT)))
    assert [t.speaker for t in turns] == [
        Speaker.USER, Speaker.ASSISTANT, Speaker.ASSISTANT, Speaker.ASSISTANT,
    ]
    cut_turn = turns[2]
    assert cut_turn.spoken_text.strip() == "hmm"
    assert "never spoken" in cut_turn.ghost_text
