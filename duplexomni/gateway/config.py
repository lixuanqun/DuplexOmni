"""Runtime configuration for the full-duplex gateway."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class GatewayConfig:
    """Knobs that bound latency, memory, and concurrency.

    Defaults are for a single machine on loopback. Raise ``max_sessions``
    only after the host has the CPU budget; each session owns an audio queue
    and at most one delegated task.
    """

    host: str = "127.0.0.1"
    port: int = 8765
    sample_rate: int = 16_000
    frame_ms: int = 20
    max_sessions: int = 64
    max_sessions_per_ip: int = 8
    max_text: int = 4_000
    max_pcm_bytes: int = 32_000
    max_bad_messages: int = 8
    outbound_queue: int = 128
    audio_queue: int = 50
    fast_timeout_s: float = 2.5
    task_timeout_s: float = 30.0
    idle_timeout_s: float = 180.0
    echo_rms: float = 0.08
    hint_rms: float = 0.02
    vad: str = "energy"
    asr: str = "scripted"
    tts: str = "scripted"
    token: str = ""
    db_path: str = ":memory:"
    llm_base_url: str = ""
    llm_model: str = ""
    llm_api_key: str = ""

    @classmethod
    def from_env(cls, **overrides: object) -> GatewayConfig:
        cfg = cls(
            host=os.environ.get("DUPLEX_HOST", "127.0.0.1"),
            port=int(os.environ.get("DUPLEX_PORT", "8765")),
            vad=os.environ.get("DUPLEX_VAD", "energy"),
            asr=os.environ.get("DUPLEX_ASR", "scripted"),
            tts=os.environ.get("DUPLEX_TTS", "scripted"),
            token=os.environ.get("DUPLEX_TOKEN", ""),
            db_path=os.environ.get("DUPLEX_DB", "duplexomni.sqlite"),
            llm_base_url=os.environ.get("DUPLEX_LLM_BASE_URL", ""),
            llm_model=os.environ.get("DUPLEX_LLM_MODEL", ""),
            llm_api_key=os.environ.get("DUPLEX_LLM_API_KEY", ""),
        )
        for key, value in overrides.items():
            if not hasattr(cfg, key):
                raise TypeError(f"unknown gateway setting {key}")
            setattr(cfg, key, value)
        return cfg

    @property
    def llm_enabled(self) -> bool:
        return bool(self.llm_base_url and self.llm_model)
