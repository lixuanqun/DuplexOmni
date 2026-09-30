"""Control-token grammar and streaming parser for DuplexOmni.

Implements the temporal control symbols of the paper (Appendix A):

======================  =======================================================
Symbol                  Meaning
======================  =======================================================
``[THINK]``             Trigger the thinking layer; the interaction layer keeps
                        listening/speaking while waiting (delayed reasoning).
``<...>``               Wrap one streamed fragment of the thinking layer's
                        result; fragments are injected progressively.
``ˆ``                   Overlap onset: user and assistant speech overlap.
``[CUT]``               Stop current assistant speech; the remaining text of the
                        turn becomes *ghost text* (kept in history, never
                        spoken).
``[WAIT]``              Suspend / reset the current reasoning request.
``[PENDNS]``            ``N`` seconds of shared silence (e.g. ``[PEND2S]``).
``[U]`` / ``[A]``       Speaker tags used by the script format (engineering
                        extension of the raw paper grammar).
======================  =======================================================

The parser is *incremental*: at inference time the Thinker emits text token by
token and the parser must raise events as soon as a control token is complete.
It is also used in batch mode by the data pipeline's consistency checks.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from enum import Enum

__all__ = [
    "Speaker",
    "Event",
    "TextDelta",
    "GhostText",
    "ThinkTrigger",
    "ResultFragment",
    "OverlapOnset",
    "Cut",
    "Wait",
    "SharedSilence",
    "TurnStart",
    "ControlTokenParser",
    "parse_script",
    "iter_turns",
    "CONTROL_TOKENS",
    "PEND_RE",
]

# The paper writes the overlap token as "ˆ" (U+02C4); accept "^" as an alias.
OVERLAP_TOKENS = ("ˆ", "^")

# characters that can start (or are) a control token; plain-text runs end here
_TOKEN_STARTS = frozenset("[<>ˆ^")

PEND_RE = re.compile(r"^\[PEND(\d+(?:\.\d+)?)S\]$")
_MAX_BRACKET = 16  # longest bracket sequence we are willing to buffer

CONTROL_TOKENS = ("[THINK]", "[CUT]", "[WAIT]", "[U]", "[A]")

_SPEAKER_TAGS = {"[U]": "user", "[A]": "assistant"}


class Speaker(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"


# --------------------------------------------------------------------------- #
# Events
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Event:
    """Base class for parser events."""


@dataclass(frozen=True)
class TextDelta(Event):
    """Plain assistant/user text that should be spoken."""

    text: str


@dataclass(frozen=True)
class GhostText(Event):
    """Text after a ``[CUT]``: preserved in dialogue history, never spoken."""

    text: str


@dataclass(frozen=True)
class ThinkTrigger(Event):
    """``[THINK]``: dispatch a (non-blocking) request to the thinking layer."""


@dataclass(frozen=True)
class ResultFragment(Event):
    """``<...>``: one streamed fragment of the thinking layer's answer."""

    text: str


@dataclass(frozen=True)
class OverlapOnset(Event):
    """``ˆ``: user speech starts while the assistant is still speaking."""


@dataclass(frozen=True)
class Cut(Event):
    """``[CUT]``: stop the current assistant speech (ghost text follows)."""


@dataclass(frozen=True)
class Wait(Event):
    """``[WAIT]``: suspend / reset the current reasoning request."""


@dataclass(frozen=True)
class SharedSilence(Event):
    """``[PENDNS]``: ``N`` seconds of shared silence."""

    seconds: float


@dataclass(frozen=True)
class TurnStart(Event):
    """``[U]``/``[A]``: a new utterance by the given speaker starts."""

    speaker: Speaker


# --------------------------------------------------------------------------- #
# Incremental parser
# --------------------------------------------------------------------------- #


class ControlTokenParser:
    """Incremental state machine over the control-token grammar.

    Feed text chunks with :meth:`feed`; collect the raised events; call
    :meth:`close` at end of stream to flush any incomplete buffer as literal
    text.
    """

    def __init__(self, *, strict_paper_grammar: bool = False) -> None:
        self.strict_paper_grammar = strict_paper_grammar
        self._buffer = ""
        self._state = "text"  # "text" | "angle" | "ghost"
        self._angle_buf = ""
        self._ghost = False  # True once [CUT] was seen in the current turn

    # -- public API --------------------------------------------------------- #

    def feed(self, chunk: str) -> list[Event]:
        events: list[Event] = []
        self._buffer += chunk
        self._drain(events, final=False)
        return events

    def close(self) -> list[Event]:
        events: list[Event] = []
        self._drain(events, final=True)
        if self._state == "angle":
            # unterminated <...>: keep the raw text so the sample can be
            # flagged by the consistency checker instead of being lost
            self._emit_text(events, "<" + self._angle_buf)
            self._angle_buf = ""
            self._state = "ghost" if self._ghost else "text"
        if self._buffer:
            self._emit_text(events, self._buffer)
            self._buffer = ""
        return events

    @property
    def in_ghost(self) -> bool:
        return self._ghost

    # -- internals ----------------------------------------------------------- #

    def _emit_text(self, events: list[Event], text: str) -> None:
        if not text:
            return
        events.append(GhostText(text) if self._ghost else TextDelta(text))

    def _drain(self, events: list[Event], *, final: bool) -> None:
        while self._buffer:
            ch = self._buffer[0]

            if self._state == "angle":
                if ch == ">":
                    self._buffer = self._buffer[1:]
                    events.append(ResultFragment(self._angle_buf))
                    self._angle_buf = ""
                    self._state = "ghost" if self._ghost else "text"
                else:
                    self._angle_buf += ch
                    self._buffer = self._buffer[1:]
                continue

            if ch == "<":
                self._buffer = self._buffer[1:]
                self._state = "angle"
                self._angle_buf = ""
                continue

            if ch in OVERLAP_TOKENS:
                self._buffer = self._buffer[1:]
                events.append(OverlapOnset())
                continue

            if ch == "[":
                consumed = self._try_bracket(events)
                if consumed:
                    continue
                if final:
                    self._emit_text(events, self._buffer[0])
                    self._buffer = self._buffer[1:]
                    continue
                # ambiguous: wait for more input (bounded by _MAX_BRACKET)
                if len(self._buffer) > _MAX_BRACKET:
                    self._emit_text(events, self._buffer[0])
                    self._buffer = self._buffer[1:]
                return  # cannot decide yet

            if ch == "]":
                # stray closing bracket: literal text
                self._emit_text(events, ch)
                self._buffer = self._buffer[1:]
                continue

            # ordinary character: emit the whole plain-text run in one event
            j = 1
            while j < len(self._buffer) and self._buffer[j] not in _TOKEN_STARTS:
                j += 1
            self._emit_text(events, self._buffer[:j])
            self._buffer = self._buffer[j:]

    def _try_bracket(self, events: list[Event]) -> bool:
        """Try to complete a bracketed token at the start of the buffer.

        Returns True if a full token was matched and consumed.
        """
        end = self._buffer.find("]")
        if end == -1:
            return False
        token = self._buffer[: end + 1]

        if token in _SPEAKER_TAGS:
            if self.strict_paper_grammar:
                return self._flush_literal(events)
            self._buffer = self._buffer[end + 1 :]
            self._ghost = False
            self._state = "text"
            events.append(TurnStart(Speaker(_SPEAKER_TAGS[token])))
            return True

        if token == "[THINK]":
            self._buffer = self._buffer[end + 1 :]
            events.append(ThinkTrigger())
            return True
        if token == "[CUT]":
            self._buffer = self._buffer[end + 1 :]
            self._ghost = True
            events.append(Cut())
            return True
        if token == "[WAIT]":
            self._buffer = self._buffer[end + 1 :]
            events.append(Wait())
            return True

        m = PEND_RE.match(token)
        if m:
            self._buffer = self._buffer[end + 1 :]
            events.append(SharedSilence(float(m.group(1))))
            return True

        return self._flush_literal(events)

    def _flush_literal(self, events: list[Event]) -> bool:
        # Not a control token: emit '[' literally and continue scanning.
        self._emit_text(events, self._buffer[0])
        self._buffer = self._buffer[1:]
        return True


# --------------------------------------------------------------------------- #
# Batch helpers
# --------------------------------------------------------------------------- #


def parse_script(script: str, *, strict_paper_grammar: bool = False) -> list[Event]:
    """Parse a whole annotated script into events."""
    parser = ControlTokenParser(strict_paper_grammar=strict_paper_grammar)
    events = parser.feed(script)
    return events + parser.close()


@dataclass
class Turn:
    """A utterance-level grouping of events between two TurnStart events."""

    speaker: Speaker
    events: list[Event] = field(default_factory=list)

    @property
    def spoken_text(self) -> str:
        return "".join(e.text for e in self.events if isinstance(e, TextDelta))

    @property
    def ghost_text(self) -> str:
        return "".join(e.text for e in self.events if isinstance(e, GhostText))


def iter_turns(events: Iterable[Event]) -> Iterator[Turn]:
    """Group a flat event stream into turns by ``TurnStart`` markers.

    Events before the first speaker tag are assigned to a synthetic assistant
    turn (system preambles etc.).
    """
    turn: Turn | None = None
    for ev in events:
        if isinstance(ev, TurnStart):
            if turn is not None:
                yield turn
            turn = Turn(speaker=ev.speaker)
        else:
            if turn is None:
                turn = Turn(speaker=Speaker.ASSISTANT)
            turn.events.append(ev)
    if turn is not None:
        yield turn
