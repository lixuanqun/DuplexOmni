"""Talker: streaming speech decoder over the codec-token prefix (paper 3.3).

Layout follows the paper exactly.  For each 480 ms slice ``i`` the Talker
sees the prefix

    P_t = [C_1, BOS, R_1, EOS, C_2, BOS, R_2, EOS, ..., C_t, BOS, ...]

where ``C_i`` are conditioning tokens derived from the Thinker
(``c = f_text(e) + f_hidden(h)``) and ``R_i`` are the slice's codec frames
embedded as ``r = u_0(q^0) + sum_k u_k(q^k)``.  Positions ``BOS..r_4``
predict the six layer-0 codes of the slice (next-token style); the MTP
module predicts the residual codebooks from those positions.
"""

from __future__ import annotations

from ..config import CodecConfig, TalkerConfig
from .transformer import BlockConfig, KVCache, TransformerStack, _require_torch

torch = _require_torch()
nn = torch.nn

__all__ = ["Talker"]


class Talker(nn.Module):
    def __init__(
        self,
        cfg: TalkerConfig,
        *,
        thinker_dim: int,
        conditioning_dim: int,
        codec: CodecConfig,
        max_positions: int = 4096,
    ) -> None:
        super().__init__()
        if cfg.d_model != conditioning_dim:
            raise ValueError(
                "conditioning_dim must equal TalkerConfig.d_model "
                f"(got {conditioning_dim} vs {cfg.d_model})"
            )
        self.cfg = cfg
        self.codec = codec
        self.max_positions = max_positions

        # conditioning projections from the Thinker (paper: c = f_text(e)+f_hidden(h))
        self.f_text = nn.Linear(thinker_dim, conditioning_dim)
        self.f_hidden = nn.Linear(thinker_dim, conditioning_dim)

        # per-codebook embedding tables u_k; r = u_0(q^0) + sum_k u_k(q^k)
        self.codebook_embeds = nn.ModuleList(
            nn.Embedding(codec.codebook_size, conditioning_dim)
            for _ in range(codec.num_codebooks)
        )
        self.bos = nn.Parameter(torch.zeros(conditioning_dim))
        self.eos = nn.Parameter(torch.zeros(conditioning_dim))
        self.pos_embed = nn.Embedding(max_positions, conditioning_dim)

        self.stack = TransformerStack(
            BlockConfig(
                d_model=cfg.d_model, n_heads=cfg.n_heads, n_layers=cfg.n_layers,
                dropout=cfg.dropout,
            )
        )
        # layer-0 codebook prediction head
        self.code_head = nn.Linear(cfg.d_model, codec.codebook_size)

        nn.init.normal_(self.bos, std=0.02)
        nn.init.normal_(self.eos, std=0.02)

    # -- conditioning ---------------------------------------------------------- #

    def conditioning(self, embeddings, hiddens):
        """Thinker (e, h) -> conditioning tokens c = f_text(e) + f_hidden(h)."""
        return self.f_text(embeddings) + self.f_hidden(hiddens)

    def sum_codec_embedding(self, codes):
        """codes: (..., K) ints -> r = u_0(q^0) + sum_k u_k(q^k), (..., D)."""
        r = self.codebook_embeds[0](codes[..., 0])
        for k in range(1, self.codec.num_codebooks):
            r = r + self.codebook_embeds[k](codes[..., k])
        return r

    # -- sequence construction -------------------------------------------------- #

    def build_training_sequence(self, cond_per_slice, codes_per_slice):
        """Teacher-forced Talker input sequence for one sample.

        Parameters
        ----------
        cond_per_slice: list of (L_i, D) conditioning tensors (may be empty)
        codes_per_slice: list of (F, K) gold codes per slice

        Returns
        -------
        input_embeds: (1, T, D), positions: (T,), codec_target_positions:
        list of index arrays (one per slice, into the flat sequence) whose
        Talker hiddens predict that slice's layer-0 codes, and
        layer0_targets: list of (F,) gold layer-0 codes.
        """
        embeds: list = []
        codec_positions: list = []
        layer0_targets: list = []
        eos_target_positions: list = []
        n_frames = self.codec.frames_per_slice
        pos = 0
        for cond, codes in zip(cond_per_slice, codes_per_slice, strict=False):
            if len(cond):
                embeds.append(cond)
                pos += len(cond)
            # BOS position predicts frame 0; r_f predicts frame f+1
            codec_positions.append(
                list(range(pos, pos + n_frames))
            )
            layer0_targets.append(codes[:, 0])
            embeds.append(self.bos.unsqueeze(0))  # (1, D)
            pos += 1
            r = self.sum_codec_embedding(codes)  # (F, D)
            embeds.append(r)
            pos += n_frames
            eos_target_positions.append(pos - 1)  # r_last position predicts EOS
            embeds.append(self.eos.unsqueeze(0))
            pos += 1
        input_embeds = torch.cat(embeds, dim=0).unsqueeze(0)  # (1, T, D)
        positions = torch.arange(pos)
        return input_embeds, positions, codec_positions, layer0_targets, eos_target_positions

    def forward_embeds(self, input_embeds, positions, *, cache: KVCache | None = None):
        x = input_embeds + self.pos_embed(positions).unsqueeze(0)
        return self.stack(x, cache=cache)

    # -- incremental runtime API ------------------------------------------------ #

    @torch.no_grad()
    def append_cond(self, cond, cache: KVCache):
        """Feed one slice's conditioning tokens + BOS into the cache.

        Returns the hidden at the BOS position — the state that predicts the
        slice's first layer-0 code.
        """
        self._append(cond, cache)
        h_bos = self._append(self.bos.unsqueeze(0), cache)
        return h_bos

    @torch.no_grad()
    def append_frames(self, r, cache: KVCache):
        """Feed the slice's codec embeddings (F, D); returns (F, D) hiddens."""
        return self._append(r, cache)

    @torch.no_grad()
    def append_eos(self, cache: KVCache):
        return self._append(self.eos.unsqueeze(0), cache)

    def _append(self, emb, cache: KVCache):
        pos = cache.length
        if pos + len(emb) > self.max_positions:
            raise RuntimeError("talker KV cache exceeded max_positions")
        positions = torch.arange(pos, pos + len(emb))
        x = emb.unsqueeze(0) + self.pos_embed(positions).unsqueeze(0)
        return self.stack(x, cache=cache)[0]
