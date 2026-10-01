"""Speech ports for the duplex gateway.

ASR, TTS, and VAD are separate engines. The session owns the floor: it
keeps listening while speech is going out, and a barge-in cancels synthesis.
``scripted`` engines need no weights. A later model is another class behind
the same methods.
"""

from __future__ import annotations

from .asr import AsrEvent, AsrPort, ScriptedAsr
from .tts import ScriptedTts, TtsPort, scripted_frames
from .vad import EnergyVad

__all__ = [
    "AsrEvent",
    "AsrPort",
    "EnergyVad",
    "ScriptedAsr",
    "ScriptedTts",
    "TtsPort",
    "build_asr",
    "build_tts",
    "build_vad",
    "scripted_frames",
]


def build_vad(name: str, *, frame_ms: int):
    if name == "energy":
        return EnergyVad(frame_ms=frame_ms)
    raise ValueError(f"unknown vad engine {name}")


def build_asr(name: str):
    if name == "scripted":
        return ScriptedAsr()
    raise ValueError(f"unknown asr engine {name}")


def build_tts(name: str, *, sample_rate: int, frame_ms: int):
    if name == "scripted":
        return ScriptedTts(
            sample_rate=sample_rate,
            frame_ms=frame_ms,
            pace_s=frame_ms / 1000,
        )
    raise ValueError(f"unknown tts engine {name}")
