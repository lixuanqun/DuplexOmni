"""Shared causal-transformer building blocks (used by Thinker and Talker)."""

from __future__ import annotations

import math
from dataclasses import dataclass


def _require_torch():
    try:
        import torch

        return torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "PyTorch is required for duplexomni.model — install with "
            "`pip install duplexomni[torch]`"
        ) from exc


torch = _require_torch()
nn = torch.nn


@dataclass
class KVCache:
    """Per-layer key/value cache for incremental decoding."""

    k: list
    v: list

    @classmethod
    def empty(cls) -> KVCache:
        return KVCache(k=[], v=[])

    @property
    def length(self) -> int:
        return int(self.k[0].shape[-2]) if self.k else 0


@dataclass
class BlockConfig:
    d_model: int
    n_heads: int
    n_layers: int
    dropout: float = 0.0


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: BlockConfig) -> None:
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0, "d_model must divide n_heads"
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)

    def forward(self, x, cache: KVCache | None, layer_idx: int):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        if cache is not None:
            if layer_idx < len(cache.k):
                k = torch.cat([cache.k[layer_idx], k], dim=-2)
                v = torch.cat([cache.v[layer_idx], v], dim=-2)
                cache.k[layer_idx] = k
                cache.v[layer_idx] = v
            else:
                cache.k.append(k)
                cache.v.append(v)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        kv_len = k.shape[-2]
        # query at absolute position (kv_len - T + i) attends columns <= itself
        q_abs = torch.arange(kv_len - T, kv_len, device=x.device).unsqueeze(1)
        k_abs = torch.arange(kv_len, device=x.device).unsqueeze(0)
        att = att.masked_fill(k_abs > q_abs, float("-inf"))
        att = att.softmax(dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class Block(nn.Module):
    def __init__(self, cfg: BlockConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model),
            nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model),
        )

    def forward(self, x, cache: KVCache | None, layer_idx: int):
        x = x + self.attn(self.ln1(x), cache, layer_idx)
        x = x + self.mlp(self.ln2(x))
        return x


class TransformerStack(nn.Module):
    """Stack of pre-norm causal blocks with optional shared KV cache."""

    def __init__(self, cfg: BlockConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.ln_f = nn.LayerNorm(cfg.d_model)

    def forward(self, x, *, cache: KVCache | None = None):
        for i, block in enumerate(self.blocks):
            x = block(x, cache, i)
        return self.ln_f(x)
