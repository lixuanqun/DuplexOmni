"""Offline full-duplex demo.

Runs the untrained tiny DuplexOmni through a scripted full-duplex session:

* the "user" is scripted (PCM + ASR text per slice), including a barge-in;
* the interaction layer streams 480 ms slices of text + speech with
  KV-cached Thinker/Talker;
* the thinking layer answers [THINK] requests asynchronously and injects
  ``<...>`` fragments mid-conversation;
* RTF and interaction events are reported, and the session audio can be
  written as a WAV mixdown.

Note: the model is randomly initialised (no checkpoint), so the generated
*text/speech content* is noise — the demo exercises the full-duplex
machinery (timing, events, barge-in, async thinking), not the language.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from .codecs import write_wav
from .config import tiny_config
from .runtime import (
    InteractionLayer,
    SimClock,
    ThinkingBridge,
    UserSliceInput,
    summarise,
)
from .runtime.thinking import EchoThinkingLayer

USER_SCRIPT: list[tuple[int, str]] = [
    (2, "hey, I wanted to ask about planning a trip this weekend"),
    (9, "wait, actually make it four people"),  # barge-in
    (16, "thanks, that helps a lot"),
]
N_SLICES = 26


def run_offline_demo(
    *,
    turns: int = 3,
    seed: int = 0,
    out_wav: str | None = None,
    verbose: bool = True,
) -> dict:
    import torch

    from .data import MockTTS

    torch.manual_seed(seed)
    model = _lazy_model(seed)

    tts = MockTTS()
    lines = USER_SCRIPT[: max(1, turns)]
    n_slices = N_SLICES + 6 * max(0, turns - 3)

    speech_cache: dict[str, bytes] = {}

    def pcm_for(text: str) -> bytes:
        if text not in speech_cache:
            speech_cache[text] = tts.synthesize(text, speaker="user").pcm
        return speech_cache[text]

    def user_stream():
        by_index = dict(lines)
        for i in range(n_slices):
            if i in by_index:
                text = by_index[i]
                yield UserSliceInput(i, pcm_for(text)[:15360], asr_text=text)
            else:
                yield UserSliceInput(i, b"\x00" * 15360)

    async def main() -> list:
        bridge = ThinkingBridge(EchoThinkingLayer(delay_s=0.0, n_fragments=3))
        layer = InteractionLayer(
            model, bridge=bridge, clock=SimClock(),
            tokens_per_slice=6, max_turn_slices=5, assistant_initiative_after=3,
        )
        return [o async for o in layer.run(user_stream())]

    outputs = asyncio.run(main())

    if verbose:
        _print_transcript(outputs, lines)
        summary = summarise(outputs)
        print("\n--- session summary ---------------------------")
        for key in (
            "slices", "duration_s", "mean_rtf", "max_rtf", "rtf_ok",
            "barge_ins", "thinking_requests", "thinking_fragments",
            "thinking_aborts", "speech_cuts", "shared_silences",
        ):
            print(f"{key:>20}: {summary[key]}")

    if out_wav:
        mixed = _mixdown(outputs)
        write_wav(Path(out_wav), mixed)
        if verbose:
            print(f"\nwrote session audio -> {out_wav}")

    return {"outputs": outputs, "summary": summarise(outputs)}


def _lazy_model(seed: int):
    import torch

    from .model import DuplexOmni

    torch.manual_seed(seed)
    return DuplexOmni(tiny_config())


def _print_transcript(outputs, lines) -> None:
    by_index = dict(lines)
    print("--- full-duplex session ------------------------")
    for o in outputs:
        marks = []
        if o.index in by_index:
            marks.append(f"USER: {by_index[o.index][:34]!r}")
        for ev in o.events:
            name = type(ev).__name__
            if name in ("BargeInDetected", "SpeechCut", "ThinkingRequested",
                        "ThinkingFragment", "ThinkingAborted", "SharedSilence", "Overlap"):
                detail = getattr(ev, "text", getattr(ev, "seconds", ""))
                marks.append(f"{name}({str(detail)[:26]})" if detail else name)
        text = o.text.strip()
        if text:
            marks.append(f"ASST: {text[:26]!r}")
        status = f"u:{'Y' if o.user_speaking else '-'} a:{'Y' if o.assistant_speaking else '-'}"
        print(f"[{o.index:3d}] {status} rtf={o.rtf:4.2f} " + " | ".join(marks))


def _mixdown(outputs) -> bytes:
    """Sum assistant audio chunks into one continuous track."""
    import array

    total = array.array("h")
    for o in outputs:
        chunk = array.array("h")
        chunk.frombytes(o.pcm[: (len(o.pcm) // 2) * 2])
        total.extend(chunk)
    return total.tobytes()
