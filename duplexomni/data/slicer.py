"""Time-slicing: turn an annotated dialog into 480 ms training slices.

Faithful to the paper's offline pipeline (Sec. 3.3 / 4.1):

1. place every utterance on a wall-clock timeline (overlap ``ˆ`` turns start
   *inside* the previous assistant utterance; backchannels do not advance the
   cursor; ``[PENDnS]`` advances it by ``n`` seconds);
2. render two *parallel* PCM tracks (user / assistant) — full-duplex means
   both parties can be audible at the same time;
3. tile the timeline into ``slice_ms`` (480 ms) slices; per slice emit:
   user/assistant speech flags, the slice's control-token events, the
   assistant text aligned to this slice, and RVQ codes (6 frames x K
   codebooks) for both tracks.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..codecs import CodecBackend, MockCodec, rms
from ..config import SAMPLE_RATE_HZ, RuntimeConfig
from ..tokens import (
    Event,
    GhostText,
    ResultFragment,
    SharedSilence,
    Speaker,
    TextDelta,
    TurnStart,
    parse_script,
)
from .director import AnnotatedDialog, AnnotatedUtterance, UtteranceKind
from .synthesis import TTSBackend

__all__ = ["Timeline", "Slice", "DatasetRecord", "slice_dialog", "iter_jsonl", "write_jsonl"]

_TURN_GAP_S = 0.25          # silence between consecutive non-overlapped turns
_OVERLAP_LEAD_S = 0.5       # user starts this long before assistant finishes
_BACKCHANNEL_LEAD_S = 0.3
_SILENCE_RMS = 0.01


@dataclass
class Timeline:
    entries: list[dict] = field(default_factory=list)  # {utterance, start_s, end_s}
    timed_events: list[dict] = field(default_factory=list)  # {time_s, event(str)}
    total_s: float = 0.0


def build_timeline(dialog: AnnotatedDialog, *, turn_gap_s: float = _TURN_GAP_S) -> Timeline:
    cursor = 0.0
    tl = Timeline()
    prev_speech: AnnotatedUtterance | None = None
    prev_entry: dict | None = None

    for utt in dialog.utterances:
        if utt.kind == UtteranceKind.EVENT:
            # fragments arrive at the current stream position (no time passes)
            for ev in _inline_events(utt.text):
                tl.timed_events.append(
                    {"time_s": round(cursor, 3), "event": _ev_name(ev), "text": _ev_text(ev)}
                )
            continue
        if utt.kind == UtteranceKind.SILENCE:
            n = float(utt.text.strip()[5:-2])
            tl.timed_events.append({"time_s": round(cursor, 3), "event": "SharedSilence", "text": n})
            cursor += n
            prev_speech = None
            prev_entry = None
            continue

        if (
            utt.overlap
            and prev_speech is not None
            and prev_speech.speaker is Speaker.ASSISTANT
        ):
            lead = _BACKCHANNEL_LEAD_S if utt.backchannel else _OVERLAP_LEAD_S
            floor = (prev_entry or {}).get("start_s", 0.0) + 0.1
            start = max(cursor - lead, floor)
        else:
            gap = 0.0 if (prev_speech is not None and prev_speech.backchannel) else turn_gap_s
            start = cursor + (gap if prev_speech is not None else 0.0)
        end = start + max(utt.duration_s, 0.05)

        # inline events of this utterance fire at its start
        for ev in _inline_events(utt.text):
            tl.timed_events.append(
                {"time_s": round(start, 3), "event": _ev_name(ev), "text": _ev_text(ev)}
            )

        entry = {"utterance": utt, "start_s": round(start, 3), "end_s": round(end, 3)}
        tl.entries.append(entry)

        if utt.backchannel:
            # the assistant keeps the floor: the cursor does not advance
            cursor = max(cursor, end)
        else:
            cursor = end
        prev_speech = utt
        prev_entry = entry

    tl.total_s = cursor
    tl.timed_events.sort(key=lambda e: e["time_s"])
    return tl


def _inline_events(text: str) -> list[Event]:
    return [
        ev for ev in parse_script(text)
        if not isinstance(ev, (TextDelta, GhostText, TurnStart))
    ]


def _ev_name(ev: Event) -> str:
    return type(ev).__name__


def _ev_text(ev: Event) -> Any:
    if isinstance(ev, ResultFragment):
        return ev.text
    if isinstance(ev, SharedSilence):
        return ev.seconds
    return ""


# --------------------------------------------------------------------------- #
# PCM tracks
# --------------------------------------------------------------------------- #


class _Track:
    def __init__(self, sample_rate: int) -> None:
        self.sr = sample_rate
        self.buf = bytearray()

    def _ensure(self, n_samples: int) -> None:
        if len(self.buf) < n_samples * 2:
            self.buf.extend(b"\x00" * (n_samples * 2 - len(self.buf)))

    def place(self, pcm: bytes, start_s: float) -> None:
        offset = int(start_s * self.sr)
        self._ensure(offset + len(pcm) // 2)
        view = memoryview(self.buf)
        import array

        existing = array.array("h")
        existing.frombytes(view[offset: offset + len(pcm)])
        added = array.array("h")
        added.frombytes(pcm[: len(pcm) // 2 * 2])
        for i in range(len(added)):
            mixed = existing[i] + added[i] if i < len(existing) else added[i]
            added[i] = max(-32768, min(32767, mixed))
        view[offset: offset + len(added) * 2] = added.tobytes()

    def window(self, start_s: float, dur_s: float) -> bytes:
        i0 = int(start_s * self.sr) * 2
        n = int(dur_s * self.sr) * 2
        if i0 >= len(self.buf):
            return b"\x00" * n
        chunk = bytes(self.buf[i0: i0 + n])
        return chunk + b"\x00" * (n - len(chunk))


def render_tracks(
    dialog: AnnotatedDialog,
    tl: Timeline,
    tts: TTSBackend,
    *,
    sample_rate: int = SAMPLE_RATE_HZ,
) -> tuple[bytes, bytes]:
    """Render (user_pcm, assistant_pcm) over the whole timeline."""
    user = _Track(sample_rate)
    assistant = _Track(sample_rate)
    for entry in tl.entries:
        utt: AnnotatedUtterance = entry["utterance"]
        spoken = _spoken_text(utt)
        if not spoken:
            continue
        audio = tts.synthesize(spoken, speaker=utt.speaker.value, language=dialog.seed.language)
        track = user if utt.speaker is Speaker.USER else assistant
        track.place(audio.pcm, entry["start_s"])
    return bytes(user.buf), bytes(assistant.buf)


def _spoken_text(utt: AnnotatedUtterance) -> str:
    return "".join(
        e.text for e in parse_script(utt.text) if isinstance(e, TextDelta)
    ).strip()


# --------------------------------------------------------------------------- #
# Slicing
# --------------------------------------------------------------------------- #


@dataclass
class Slice:
    index: int
    start_ms: int
    end_ms: int
    user_speech: bool
    assistant_speech: bool
    events: list[dict]
    target_text: str = ""
    user_text: str = ""
    user_codes: list[list[int]] = field(default_factory=list)
    assistant_codes: list[list[int]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "user_speech": self.user_speech,
            "assistant_speech": self.assistant_speech,
            "events": self.events,
            "target_text": self.target_text,
            "user_text": self.user_text,
            "user_codes": self.user_codes,
            "assistant_codes": self.assistant_codes,
        }


def slice_dialog(
    dialog: AnnotatedDialog,
    tts: TTSBackend,
    codec: CodecBackend | None = None,
    *,
    runtime: RuntimeConfig | None = None,
) -> tuple[list[Slice], Timeline]:
    runtime = runtime or RuntimeConfig()
    codec = codec or MockCodec(runtime.codec)
    slice_s = runtime.slice_ms / 1000.0
    n_frames = runtime.codec.frames_per_slice
    tl = build_timeline(dialog)
    user_pcm, assistant_pcm = render_tracks(dialog, tl, tts)

    # char-time alignment for both speakers' text
    char_times: list[tuple[float, str, Speaker]] = []  # (center_time, char, speaker)
    for entry in tl.entries:
        utt: AnnotatedUtterance = entry["utterance"]
        spoken = _spoken_text(utt)
        if not spoken:
            continue
        dur = max(entry["end_s"] - entry["start_s"], 0.05)
        for i, ch in enumerate(spoken):
            char_times.append(
                (entry["start_s"] + dur * (i + 0.5) / len(spoken), ch, utt.speaker)
            )

    n_slices = max(1, math.ceil(tl.total_s / slice_s))
    slices: list[Slice] = []
    for k in range(n_slices):
        t0, t1 = k * slice_s, (k + 1) * slice_s
        u_win = _window(user_pcm, t0, slice_s)
        a_win = _window(assistant_pcm, t0, slice_s)
        events = [dict(e) for e in tl.timed_events if t0 <= e["time_s"] < t1]
        text = "".join(ch for tc, ch, sp in char_times if t0 <= tc < t1 and sp is Speaker.ASSISTANT)
        utext = "".join(ch for tc, ch, sp in char_times if t0 <= tc < t1 and sp is Speaker.USER)
        slices.append(
            Slice(
                index=k,
                start_ms=round(t0 * 1000),
                end_ms=round(t1 * 1000),
                user_speech=rms(u_win) > _SILENCE_RMS,
                assistant_speech=rms(a_win) > _SILENCE_RMS,
                events=events,
                target_text=text,
                user_text=utext,
                user_codes=codec.encode(
                    u_win, n_frames=n_frames, seed_key=f"{dialog.seed.scenario_id}-u-{k}"
                ),
                assistant_codes=codec.encode(
                    a_win, n_frames=n_frames, seed_key=f"{dialog.seed.scenario_id}-a-{k}"
                ),
            )
        )
    return slices, tl


def _window(pcm: bytes, start_s: float, dur_s: float, *, sample_rate: int = SAMPLE_RATE_HZ) -> bytes:
    i0 = int(start_s * sample_rate) * 2
    n = int(dur_s * sample_rate) * 2
    if i0 >= len(pcm):
        return b"\x00" * n
    chunk = pcm[i0: i0 + n]
    return chunk + b"\x00" * (n - len(chunk))


# --------------------------------------------------------------------------- #
# Dataset records
# --------------------------------------------------------------------------- #


@dataclass
class DatasetRecord:
    id: str
    language: str
    patterns: list[str]
    annotated_script: str
    slices: list[Slice]
    total_ms: int

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "language": self.language,
            "patterns": self.patterns,
            "annotated_script": self.annotated_script,
            "total_ms": self.total_ms,
            "n_slices": len(self.slices),
            "slices": [s.to_dict() for s in self.slices],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: dict) -> DatasetRecord:
        return cls(
            id=d["id"],
            language=d["language"],
            patterns=d["patterns"],
            annotated_script=d["annotated_script"],
            slices=[Slice(**{**s, "events": s["events"]}) for s in d["slices"]],
            total_ms=d["total_ms"],
        )

    @classmethod
    def from_json(cls, line: str) -> DatasetRecord:
        return cls.from_dict(json.loads(line))


def write_jsonl(records: Iterator[DatasetRecord] | list[DatasetRecord], path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(rec.to_json() + "\n")
            n += 1
    return n


def iter_jsonl(path: str | Path) -> Iterator[DatasetRecord]:
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield DatasetRecord.from_json(line)
