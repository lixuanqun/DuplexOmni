"""MTP module: multi-token prediction of the residual RVQ codebooks.

The Talker autoregressively predicts the *layer-0* codebook; the MTP module
then predicts the remaining ``K-1`` residual codebooks in parallel for each
codec frame, conditioned on the layer-0 code embedding and the Talker's
hidden state at that frame position (paper Sec. 3.3, "MTP module predicting
residual codebooks").
"""

from __future__ import annotations

from ..config import CodecConfig
from .transformer import _require_torch

torch = _require_torch()
nn = torch.nn

__all__ = ["MTPHead"]


class MTPHead(nn.Module):
    """Per-frame residual-codebook prediction heads.

    Input at training time is the *teacher-forced* layer-0 embedding
    ``u_0(q^0)``; at inference the embedding of the predicted (greedy)
    layer-0 token is used instead.
    """

    def __init__(self, conditioning_dim: int, codec: CodecConfig, *, hidden_dim: int | None = None) -> None:
        super().__init__()
        self.codec = codec
        self.n_residual = codec.num_codebooks - 1
        hidden = hidden_dim or conditioning_dim
        self.in_proj = nn.Linear(2 * conditioning_dim, hidden)
        self.heads = nn.ModuleList(
            nn.Linear(hidden, codec.codebook_size) for _ in range(self.n_residual)
        )

    def forward(self, layer0_embed, talker_hidden):
        """(B, F, D) + (B, F, D) -> list of (B, F, codebook_size) logits."""
        feats = torch.nn.functional.gelu(
            self.in_proj(torch.cat([layer0_embed, talker_hidden], dim=-1))
        )
        return [head(feats) for head in self.heads]
