"""RVQ speech-codec abstraction plus stdlib WAV helpers.

The paper uses the **Mimi** codec (12.5 Hz, 8 codebooks, 2048 entries) for
both the user side (encode) and the assistant side (Talker targets /
Code2Wav decode).  Swapping in real Mimi requires ``mimi_*`` weights; this
module therefore defines the :class:`CodecBackend` protocol and ships a
deterministic :class:`MockCodec` so the whole pipeline (data + runtime demo)
runs offline.

WAV I/O uses only ``wave`` + ``array`` (``audioop`` was removed in Python 3.13).
"""

from __future__ import annotations

import hashlib
import math
import struct
import wave
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .config import SAMPLE_RATE_HZ, CodecConfig


class CodecBackend(Protocol):
    """Minimal streaming RVQ codec interface used by the pipeline/runtime."""

    def encode(self, pcm: bytes | None, *, n_frames: int, seed_key: str) -> list[list[int]]:
        """Encode PCM (16-bit mono LE) into ``n_frames x K`` codebook indices.

        ``pcm`` may be ``None`` for silence / event-only slices; ``seed_key``
        identifies the stream segment so deterministic mocks stay stable.
        """
        ...

    def decode(self, codes: Sequence[Sequence[int]], *, seed_key: str = "") -> bytes:
        """Decode ``frames x K`` codes back into 16-bit mono PCM."""
        ...


def _stable_rng(seed_key: str) -> tuple[int, int]:
    digest = hashlib.sha256(seed_key.encode("utf-8")).digest()
    return struct.unpack(">QQ", digest[:16])


class MockCodec:
    """Deterministic placeholder RVQ codec.

    ``encode`` maps each frame to pseudo-random-but-stable codebook entries
    derived from ``seed_key``; louder audio (higher RMS) shifts the first
    codebook entries so silence and speech are distinguishable downstream.
    ``decode`` renders each frame as a short voiced-ish tone whose pitch and
    amplitude depend on the codes, producing audible (if synthetic) audio.
    """

    def __init__(self, config: CodecConfig | None = None) -> None:
        self.config = config or CodecConfig()

    def encode(self, pcm: bytes | None, *, n_frames: int, seed_key: str) -> list[list[int]]:
        rng_state = _stable_rng(seed_key)
        level = rms(pcm) if pcm else 0.0
        k = self.config.num_codebooks
        size = self.config.codebook_size
        frames: list[list[int]] = []
        state = rng_state[0] ^ rng_state[1]
        for _f in range(n_frames):
            codes = []
            for _ in range(k):
                state = (state * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
                codes.append(state % size)
            # encode loudness into codebook 0 so speech/silence differ
            if level > 0.01:
                codes[0] = (codes[0] + int(level * 100)) % size
            frames.append(codes)
        return frames

    def decode(self, codes: Sequence[Sequence[int]], *, seed_key: str = "") -> bytes:
        import array

        frame_samples = int(SAMPLE_RATE_HZ / self.config.frame_rate_hz)
        pcm = array.array("h")
        for i, frame in enumerate(codes):
            if not frame:
                continue
            c0 = frame[0] % 256
            pitch = 120.0 + c0 * 1.2  # 120 .. 428 Hz
            amp = 6000 if c0 > 8 else 800
            for n in range(frame_samples):
                t = (i * frame_samples + n) / SAMPLE_RATE_HZ
                env = math.exp(-3.0 * ((n / frame_samples) - 0.3) ** 2)
                val = amp * env * (
                    math.sin(2 * math.pi * pitch * t)
                    + 0.35 * math.sin(4 * math.pi * pitch * t + frame[1] % 7)
                )
                pcm.append(int(max(-32767, min(32767, val))))
        return pcm.tobytes()


# --------------------------------------------------------------------------- #
# WAV helpers (stdlib only)
# --------------------------------------------------------------------------- #


def rms(pcm: bytes | None) -> float:
    """Root-mean-square of 16-bit mono PCM, normalised to [0, 1]."""
    if not pcm:
        return 0.0
    import array

    samples = array.array("h")
    samples.frombytes(pcm[: (len(pcm) // 2) * 2])
    if not samples:
        return 0.0
    acc = 0
    step = max(1, len(samples) // 4096)  # subsample long buffers
    count = 0
    for s in samples[::step]:
        acc += s * s
        count += 1
    return math.sqrt(acc / max(count, 1)) / 32768.0


def write_wav(path: str | Path, pcm: bytes, sample_rate: int = SAMPLE_RATE_HZ) -> Path:
    """Write 16-bit mono PCM to a WAV file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return path


def read_wav(path: str | Path) -> tuple[bytes, int]:
    """Read a WAV file as (16-bit mono PCM, sample_rate)."""
    with wave.open(str(path), "rb") as w:
        assert w.getnchannels() == 1, "only mono supported"
        assert w.getsampwidth() == 2, "only 16-bit supported"
        return w.readframes(w.getnframes()), w.getframerate()


@dataclass
class SliceAudio:
    """One 480 ms slice of audio in both representations."""

    pcm: bytes
    codes: list[list[int]]
    speech: bool  # False during [CUT] ghost region / shared silence


def concat_pcm(chunks: Sequence[bytes]) -> bytes:
    return b"".join(chunks)
