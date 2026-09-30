"""Thinker: the MLLM backbone of the interaction layer (paper Sec. 3.3).

The Thinker consumes the interaction token stream (user text, control
tokens, thinking-layer result fragments, its own past text) and produces,
for every assistant token, both

* the input embedding ``e`` and
* the final-layer hidden state ``h``

which the Talker projects into conditioning tokens
``c = f_text(e) + f_hidden(h)``.

A small causal transformer with an explicit KV cache powers both training
(teacher forcing) and runtime incremental decoding; the cached path is
numerically equivalent to the full forward (covered by tests).
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import ThinkerConfig
from .transformer import BlockConfig, KVCache, TransformerStack, _require_torch

torch = _require_torch()
nn = torch.nn

__all__ = ["Thinker", "ThinkerOutput", "KVCache"]


@dataclass
class ThinkerOutput:
    logits: object
    embeddings: object  # (B, T, d) input embeddings of the tokens
    hiddens: object     # (B, T, d) final-layer hidden states


class Thinker(nn.Module):
    def __init__(self, cfg: ThinkerConfig, *, vocab_size: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, cfg.d_model)
        self.pos_embed = nn.Embedding(cfg.max_positions, cfg.d_model)
        self.stack = TransformerStack(
            BlockConfig(
                d_model=cfg.d_model, n_heads=cfg.n_heads, n_layers=cfg.n_layers,
                dropout=cfg.dropout,
            )
        )
        self.lm_head = nn.Linear(cfg.d_model, vocab_size, bias=False)

    def forward(
        self,
        input_ids,
        *,
        cache: KVCache | None = None,
        position_offset: int | None = None,
    ) -> ThinkerOutput:
        """Causal forward; with ``cache`` appends keys/values for incremental use.

        ``position_offset`` overrides the positional base (defaults to the
        cache length) so cached and full forwards align.
        """
        B, T = input_ids.shape
        offset = cache.length if cache is not None else 0
        if position_offset is not None:
            offset = position_offset
        if offset + T > self.cfg.max_positions:
            raise ValueError(
                f"sequence length {offset + T} exceeds max_positions "
                f"{self.cfg.max_positions}"
            )
        positions = torch.arange(offset, offset + T, device=input_ids.device)
        e = self.embed(input_ids)
        x = e + self.pos_embed(positions).unsqueeze(0)
        h = self.stack(x, cache=cache)
        return ThinkerOutput(logits=self.lm_head(h), embeddings=e, hiddens=h)

    @torch.no_grad()
    def step(self, token_id, cache: KVCache):
        """Incremental decoding of a single token; returns logits (B, V)."""
        out = self.forward(token_id, cache=cache)
        return out.logits[:, -1]
