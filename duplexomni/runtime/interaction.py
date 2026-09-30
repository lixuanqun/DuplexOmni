"""Interaction layer: the 480 ms time-sliced full-duplex loop (paper 3.3-3.4).

Each slice:

1. ingest one user slice (PCM + optional front-end ASR text + video tokens);
2. the VAD decides whether the user is speaking; a rising edge while the
   assistant speaks is a **barge-in**: current speech is cut (ghost text
   preserved) and any pending thinking request is reset (``[WAIT]``);
3. drain the thinking bridge's ``<...>`` fragments and the user text into
   the Thinker's input stream;
4. generate a bounded number of assistant tokens with the KV-cached
   Thinker; the streaming control-token parser routes ``[THINK]`` /
   ``[CUT]`` / ``[WAIT]`` / ``[PENDnS]`` / ``ˆ`` events and produces the
   slice's spoken/ghost text split;
5. the Talker produces the slice's 6 codec frames (layer-0 AR + MTP) and
   Code2Wav decodes 480 ms of audio — replaced by silence while muted;
6. RTF is measured per slice (paper target: < 1).

Turn-taking policy notes (runtime policy, not model internals): an
assistant turn opens when there is fresh context (user text or thinking
fragment) or after ``assistant_initiative_after`` slices of silence
(assistant-initiated turns); it closes after ``max_turn_slices`` speech
slices, on ``[CUT]`` or on a barge-in.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass
from typing import Protocol

from ..config import SAMPLE_RATE_HZ, DuplexOmniConfig
from ..model.full import DuplexOmni, SessionGeneration
from ..model.tokenizer import SPECIAL_TOKENS, TextTokenizer
from ..tokens import (
    ControlTokenParser,
    Cut,
    GhostText,
    OverlapOnset,
    SharedSilence,
    TextDelta,
    ThinkTrigger,
    Wait,
)
from .bridge import ThinkingBridge
from .events import (
    BargeInDetected,
    Overlap,
    RuntimeEvent,
    SliceOutput,
    SpeechCut,
)
from .metrics import RTFMeter
from .session import SessionState
from .vad import EnergyVAD

__all__ = ["UserSliceInput", "InteractionLayer", "SimClock", "WallClock"]


class Clock(Protocol):
    def now(self) -> float: ...

    async def tick(self, slice_s: float) -> None: ...


class SimClock:
    """Deterministic clock for offline demos/tests.

    ``tick`` still yields to the event loop (``asyncio.sleep(0)``) so that
    concurrently scheduled thinking-layer tasks make progress each slice —
    without that, the slice loop would starve them.
    """

    def __init__(self) -> None:
        self._t = 0.0

    def now(self) -> float:
        return self._t

    async def tick(self, slice_s: float) -> None:
        self._t += slice_s
        await asyncio.sleep(0)


class WallClock:
    def now(self) -> float:
        return time.monotonic()

    async def tick(self, slice_s: float) -> None:
        await asyncio.sleep(slice_s)


@dataclass
class UserSliceInput:
    index: int
    pcm: bytes = b""
    asr_text: str | None = None
    video_tokens: list | None = None


@dataclass
class _LoopState:
    assistant_turn_open: bool = False
    muted: bool = False
    silence_budget_s: float = 0.0
    turn_slices: int = 0


class InteractionLayer:
    def __init__(
        self,
        model: DuplexOmni,
        *,
        config: DuplexOmniConfig | None = None,
        bridge: ThinkingBridge | None = None,
        vad: EnergyVAD | None = None,
        clock: Clock | None = None,
        tokens_per_slice: int = 8,
        max_turn_slices: int = 8,
        assistant_initiative_after: int = 6,
    ) -> None:
        self.model = model
        self.config = config or model.config
        self.bridge = bridge
        self.vad = vad or EnergyVAD(self.config.runtime)
        self.clock = clock or SimClock()
        self.tokens_per_slice = tokens_per_slice
        self.max_turn_slices = max_turn_slices
        self.assistant_initiative_after = assistant_initiative_after

        self.tokenizer: TextTokenizer = model.tokenizer
        self.state = SessionState()
        self.rtf = RTFMeter(slice_s=self.config.runtime.slice_ms / 1000)
        self._reverse_special = {v: k for k, v in SPECIAL_TOKENS.items()}
        self._byte_buf = bytearray()

    # ------------------------------------------------------------------ #

    async def run(
        self, user_stream: AsyncIterator[UserSliceInput] | Iterable[UserSliceInput]
    ) -> AsyncIterator[SliceOutput]:
        session: SessionGeneration = self.model.new_session()
        parser = ControlTokenParser()
        self.state = SessionState()
        self.state.task_state["slice_ms"] = self.config.runtime.slice_ms
        loop = _LoopState()

        session.push_context([self.tokenizer.bos()])

        user_speaking_prev = False
        slices_since_context = self.assistant_initiative_after  # open the session
        pcm_silence_len = (
            SAMPLE_RATE_HZ * self.config.runtime.slice_ms // 1000
        ) * 2

        index = 0
        async for user_in in self._aimport(user_stream):
            t0 = time.perf_counter()
            events: list[RuntimeEvent] = list(self.bridge.collect_events()) if self.bridge else []

            # ---- 1-2. user side, VAD, barge-in ------------------------------ #
            user_speaking = self.vad.update(user_in.pcm)
            new_context = False
            if user_speaking and not user_speaking_prev and loop.assistant_turn_open:
                events.append(BargeInDetected(slice_index=index))
                events.append(SpeechCut(ghost_text=""))
                loop.assistant_turn_open = False
                loop.turn_slices = 0
                loop.muted = True
                if self.bridge:
                    self.bridge.halt("barge-in")
                    self.state.aborts += 1
                    events.extend(self.bridge.collect_events())
            if user_speaking and loop.assistant_turn_open:
                events.append(Overlap(slice_index=index))
            user_speaking_prev = user_speaking

            # ---- 3. inject thinking fragments + user text -------------------- #
            context_ids: list[int] = []
            if self.bridge:
                for fragment in self.bridge.take_pending():
                    context_ids.extend(self._encode_fragment(fragment))
                    new_context = True
            if user_in.asr_text:
                if user_speaking and loop.assistant_turn_open:
                    context_ids.append(SPECIAL_TOKENS["ˆ"])
                context_ids.append(SPECIAL_TOKENS["[U]"])
                context_ids.extend(self.tokenizer.encode_text(user_in.asr_text))
                self.state.add_user_text(user_in.asr_text)
                new_context = True
                loop.muted = False
            if context_ids:
                session.push_context(context_ids)
                slices_since_context = 0
            else:
                slices_since_context += 1

            # ---- open an assistant turn? -------------------------------------- #
            if (
                not loop.assistant_turn_open
                and not loop.muted
                and loop.silence_budget_s <= 0
                and (new_context or slices_since_context >= self.assistant_initiative_after)
            ):
                loop.assistant_turn_open = True
                loop.turn_slices = 0

            # ---- 4. generate assistant tokens -------------------------------- #
            spoken_parts: list[str] = []
            ghost_parts: list[str] = []
            if loop.assistant_turn_open:
                logits = session.begin_assistant_token(SPECIAL_TOKENS["[A]"])
                for ev in parser.feed("[A]"):
                    self._handle_event(ev, loop, events, index, spoken_parts, ghost_parts)
                for _ in range(self.tokens_per_slice):
                    tok = int(logits.argmax())
                    logits = session.assistant_token(tok)
                    for ev in parser.feed(self._decode_token(tok)):
                        self._handle_event(ev, loop, events, index, spoken_parts, ghost_parts)
                loop.turn_slices += 1
                if loop.turn_slices >= self.max_turn_slices:
                    loop.assistant_turn_open = False

            spoken = "".join(spoken_parts)
            ghost = "".join(ghost_parts)
            if spoken.strip() or ghost:
                self.state.add_assistant_text(spoken, ghost)

            # ---- 5. speech ----------------------------------------------------- #
            slice_s = self.config.runtime.slice_ms / 1000
            if loop.silence_budget_s > 0:
                loop.silence_budget_s = max(0.0, loop.silence_budget_s - slice_s)
            codes, pcm, _wav = session.finish_slice()
            speaking = (
                loop.assistant_turn_open and not loop.muted and loop.silence_budget_s <= 0
            )
            if not speaking:
                pcm = b"\x00" * (len(pcm) or pcm_silence_len)
            if loop.muted and not loop.assistant_turn_open:
                loop.muted = False

            compute_s = time.perf_counter() - t0
            rtf = self.rtf.measure(compute_s)
            await self.clock.tick(slice_s)

            yield SliceOutput(
                index=index,
                text=spoken,
                pcm=pcm,
                codes=codes,
                events=events,
                compute_s=compute_s,
                rtf=rtf,
                user_speaking=user_speaking,
                assistant_speaking=speaking,
            )
            index += 1

        if self.bridge:
            self.bridge.halt("session-end")

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    async def _aimport(stream):
        """Accept async iterators and plain iterables alike."""
        if hasattr(stream, "__anext__"):
            async for item in stream:
                yield item
        else:
            for item in stream:
                yield item

    def _encode_fragment(self, fragment: str) -> list[int]:
        inner = (
            fragment[1:-1] if fragment.startswith("<") and fragment.endswith(">") else fragment
        )
        return [
            SPECIAL_TOKENS["<"], *self.tokenizer.encode_text(inner), SPECIAL_TOKENS[">"]
        ]

    def _decode_token(self, tok: int) -> str:
        """Generated id -> parser-visible string (incremental UTF-8 safe)."""
        if tok >= 256:
            special = self._reverse_special.get(tok, "")
            if special in ("[BOS]", "[EOS]", "<pad>"):
                return ""  # structural tokens are not stream text
            return special
        self._byte_buf.append(tok)
        try:
            s = bytes(self._byte_buf).decode("utf-8")
        except UnicodeDecodeError:
            return ""
        self._byte_buf.clear()
        return s

    def _handle_event(
        self,
        ev,
        loop: _LoopState,
        events: list[RuntimeEvent],
        index: int,
        spoken_parts: list[str],
        ghost_parts: list[str],
    ) -> None:
        if isinstance(ev, TextDelta):
            spoken_parts.append(ev.text)
        elif isinstance(ev, GhostText):
            ghost_parts.append(ev.text)
        elif isinstance(ev, ThinkTrigger):
            if self.bridge:
                query = next(
                    (t["text"] for t in reversed(self.state.turns) if t["speaker"] == "user"),
                    "",
                )
                self.state.thinking_requests += 1
                self.bridge.request(self.state.build_context(), query=query)
                events.extend(self.bridge.collect_events())
        elif isinstance(ev, Cut):
            loop.muted = True
            events.append(SpeechCut(ghost_text="".join(ghost_parts)))
        elif isinstance(ev, Wait):
            if self.bridge:
                self.bridge.halt("model-wait")
                self.state.aborts += 1
                events.extend(self.bridge.collect_events())
        elif isinstance(ev, SharedSilence):
            loop.silence_budget_s = ev.seconds
            loop.muted = True
            events.append(SharedSilence(seconds=ev.seconds))
        elif isinstance(ev, OverlapOnset):
            events.append(Overlap(slice_index=index))
