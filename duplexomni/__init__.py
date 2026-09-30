"""DuplexOmni: engineering reference implementation of arXiv:2606.09186.

Real-time listening, seeing, thinking and speaking for full-duplex
interaction, decomposed into

* an **interaction layer** — a time-sliced (480 ms) Thinker-Talker speech
  model that streams text + audio, and
* a **thinking layer** — a pluggable reasoning module that collaborates
  asynchronously through control tokens ([THINK] / <...> / ˆ / [CUT] /
  [WAIT] / [PENDNS]).
"""

from .config import (
    SAMPLE_RATE_HZ,
    SLICE_MS,
    CodecConfig,
    DuplexOmniConfig,
    ModelConfig,
    RuntimeConfig,
    TalkerConfig,
    ThinkerConfig,
    TrainingConfig,
    paper_scale_config,
    tiny_config,
)
from .tokens import (
    ControlTokenParser,
    Event,
    parse_script,
)

__version__ = "0.1.0"

__all__ = [
    "SLICE_MS",
    "SAMPLE_RATE_HZ",
    "CodecConfig",
    "ControlTokenParser",
    "DuplexOmniConfig",
    "Event",
    "ModelConfig",
    "RuntimeConfig",
    "TalkerConfig",
    "ThinkerConfig",
    "TrainingConfig",
    "parse_script",
    "paper_scale_config",
    "tiny_config",
]
