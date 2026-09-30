"""Configuration objects for DuplexOmni.

The paper's numeric constants (480 ms interaction slices, Mimi codec at 12.5 Hz
=> 80 ms per frame => 6 codec frames per slice, 8 RVQ codebooks) are captured
in :class:`CodecConfig` / :class:`RuntimeConfig`.  Model dimensions are fully
configurable so the same code runs a tiny CPU-testable model or a
paper-scale backbone.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SLICE_MS = 480
SAMPLE_RATE_HZ = 16_000
SAMPLES_PER_SLICE = SAMPLE_RATE_HZ * SLICE_MS // 1000  # 7680


@dataclass
class CodecConfig:
    """Mimi-style RVQ speech codec (paper Sec. 3.3)."""

    frame_rate_hz: float = 12.5
    num_codebooks: int = 8
    codebook_size: int = 2048
    frames_per_slice: int = 6  # 480 ms / 80 ms

    @property
    def frame_ms(self) -> float:
        return 1000.0 / self.frame_rate_hz


@dataclass
class ThinkerConfig:
    """MLLM backbone ("Thinker") producing text tokens, embeddings and hiddens."""

    vocab_size: int = 512
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    max_positions: int = 2048
    dropout: float = 0.0


@dataclass
class TalkerConfig:
    """Speech decoder ("Talker") predicting layer-0 RVQ tokens, plus MTP residuals."""

    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.0


@dataclass
class ModelConfig:
    thinker: ThinkerConfig = field(default_factory=ThinkerConfig)
    talker: TalkerConfig = field(default_factory=TalkerConfig)
    # conditioning token: c = f_text(e) + f_hidden(h)  (paper Eq. in Sec. 3.3)
    conditioning_dim: int = 64
    code2wav_upsample: int = 4

    def validate(self) -> None:
        if self.conditioning_dim <= 0:
            raise ValueError("conditioning_dim must be positive")


@dataclass
class TrainingConfig:
    """Two-stage SFT with alternating Thinker/Talker optimization (paper Sec. 4.1)."""

    thinker_lr: float = 1e-5
    talker_lr: float = 1e-4
    batch_size: int = 128
    loss_weight_thinker: float = 1.0  # 1:1 loss ratio (paper)
    loss_weight_talker: float = 1.0
    max_steps: int = 1000
    log_every: int = 10


@dataclass
class RuntimeConfig:
    slice_ms: int = SLICE_MS
    target_rtf: float = 1.0
    # Barge-in sensitivity for the energy VAD shipped with the runtime demo.
    vad_threshold: float = 0.02
    vad_hangover_ms: int = 320
    # How long the interaction layer keeps waiting for the thinking layer
    # before answering on its own (delayed reasoning budget).
    thinking_budget_s: float = 8.0

    @property
    def frames_per_slice(self) -> int:
        return self.codec.frames_per_slice

    codec: CodecConfig = field(default_factory=CodecConfig)


@dataclass
class DuplexOmniConfig:
    """Top-level configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    seed: int = 1234

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DuplexOmniConfig:
        model = ModelConfig(
            thinker=ThinkerConfig(**d.get("model", {}).get("thinker", {})),
            talker=TalkerConfig(**d.get("model", {}).get("talker", {})),
            conditioning_dim=d.get("model", {}).get("conditioning_dim", 64),
            code2wav_upsample=d.get("model", {}).get("code2wav_upsample", 4),
        )
        training = TrainingConfig(**d.get("training", {}))
        runtime = RuntimeConfig(
            slice_ms=d.get("runtime", {}).get("slice_ms", SLICE_MS),
            target_rtf=d.get("runtime", {}).get("target_rtf", 1.0),
            vad_threshold=d.get("runtime", {}).get("vad_threshold", 0.02),
            vad_hangover_ms=d.get("runtime", {}).get("vad_hangover_ms", 320),
            thinking_budget_s=d.get("runtime", {}).get("thinking_budget_s", 8.0),
            codec=CodecConfig(**d.get("runtime", {}).get("codec", {})),
        )
        return cls(model=model, training=training, runtime=runtime, seed=d.get("seed", 1234))

    @classmethod
    def load(cls, path: str | Path) -> DuplexOmniConfig:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def tiny_config() -> DuplexOmniConfig:
    """CPU-friendly configuration used by tests and examples.

    LRs are larger than the paper's 1e-5/1e-4 (which target a ~7B backbone);
    see :func:`paper_scale_config` for the faithful values.
    """
    return DuplexOmniConfig(
        model=ModelConfig(
            thinker=ThinkerConfig(vocab_size=512, d_model=32, n_heads=4, n_layers=2),
            talker=TalkerConfig(d_model=32, n_heads=4, n_layers=2),
            conditioning_dim=32,
        ),
        training=TrainingConfig(batch_size=4, max_steps=50, log_every=5,
                                thinker_lr=3e-4, talker_lr=1e-3),
        seed=1234,
    )


def paper_scale_config() -> DuplexOmniConfig:
    """A Qwen3-Omni-scale reference configuration (not runnable here; documents
    the paper's setup: ~7B thinker backbone, Mimi codec, 480 ms slices)."""
    return DuplexOmniConfig(
        model=ModelConfig(
            thinker=ThinkerConfig(
                vocab_size=151_936,
                d_model=3584,
                n_heads=28,
                n_layers=28,
                max_positions=32_768,
            ),
            talker=TalkerConfig(d_model=1024, n_heads=16, n_layers=12),
            conditioning_dim=1024,
        ),
        training=TrainingConfig(
            thinker_lr=1e-5,
            talker_lr=1e-4,
            batch_size=128,
            max_steps=50_000,
        ),
    )
