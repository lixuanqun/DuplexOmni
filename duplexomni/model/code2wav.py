"""Code2Wav: decode summed codec embeddings into 480 ms of waveform.

Per paper Sec. 3.3 the Talker's per-frame codec representation
``r = u_0(q^0) + sum_k u_k(q^k)`` is decoded to PCM by a Code2Wav decoder.
This reference decoder is a small transposed-convolution upsampler
(80 ms/frame @16 kHz = 1280 samples; 1280 = 5 * 4^4).
"""

from __future__ import annotations

from ..config import CodecConfig
from .transformer import _require_torch

torch = _require_torch()
nn = torch.nn

__all__ = ["Code2Wav"]


def _upsample_plan(target_samples: int) -> list[int]:
    """Factor ``target_samples`` into small strides (largest first)."""
    strides: list[int] = []
    remaining = target_samples
    for stride in (8, 5, 4, 3, 2):
        while remaining % stride == 0 and remaining // stride >= 1 and len(strides) < 6:
            if remaining // stride == 1 and len(strides) >= 1:
                break
            strides.append(stride)
            remaining //= stride
            if remaining == 1:
                break
    if remaining != 1:
        strides.append(remaining)
    return strides


class Code2Wav(nn.Module):
    def __init__(self, conditioning_dim: int, codec: CodecConfig, *, sample_rate: int = 16_000) -> None:
        super().__init__()
        self.codec = codec
        self.frame_samples = int(sample_rate / codec.frame_rate_hz)
        strides = _upsample_plan(self.frame_samples)
        layers: list[nn.Module] = []
        ch = conditioning_dim
        for s in strides:
            layers.append(
                nn.ConvTranspose1d(ch, ch // 2 if ch // 2 >= 8 else ch, kernel_size=s, stride=s)
            )
            ch = ch // 2 if ch >= 16 else ch
            layers.append(nn.GELU())
        layers.append(nn.Conv1d(ch, 1, kernel_size=1))
        layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, r):
        """r: (B, F, D) summed codec embeddings -> (B, F * frame_samples) PCM."""
        y = self.net(r.transpose(1, 2))  # (B, 1, F * frame_samples)
        return y.squeeze(1)

    def to_pcm16(self, waveform) -> bytes:
        """Convert a (F * frame_samples,) float tensor to 16-bit mono PCM."""
        import numpy as np

        arr = waveform.detach().cpu().numpy().astype(np.float64)
        arr = np.clip(arr, -1.0, 1.0)
        return (arr * 32767.0).astype("<i2").tobytes()
