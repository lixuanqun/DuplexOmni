"""The full DuplexOmni model: losses, alternating trainer, session inference.

Structure (paper Sec. 3.2-3.4):

* :class:`DuplexOmni` — Thinker + Talker + MTP + Code2Wav;
* ``losses()`` — Thinker text CE (assistant spans) + Talker layer-0 CE +
  MTP residual CE, combined 1:1 (paper's loss ratio);
* :class:`AlternateTrainer` — Thinker and Talker optimised alternately with
  the counterpart frozen (LRs 1e-5 / 1e-4; paper Sec. 4.1);
* :func:`train_two_stage` — two-stage SFT driver (stage 1: large-scale
  speech interaction; stage 2: high-quality interaction + video calls);
* :class:`SessionGeneration` — KV-cached runtime state implementing the
  480 ms time-sliced inference loop used by the interaction layer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

from ..config import DuplexOmniConfig
from .code2wav import Code2Wav
from .collate import SampleTensors
from .mtp import MTPHead
from .talker import Talker
from .thinker import Thinker
from .tokenizer import TEXT_VOCAB_SIZE, TextTokenizer
from .transformer import KVCache, _require_torch

torch = _require_torch()
nn = torch.nn

__all__ = ["DuplexOmni", "AlternateTrainer", "train_two_stage", "SessionGeneration"]


class DuplexOmni(nn.Module):
    def __init__(self, config: DuplexOmniConfig) -> None:
        super().__init__()
        config.model.validate()
        self.config = config
        self.tokenizer = TextTokenizer()
        self.thinker = Thinker(config.model.thinker, vocab_size=TEXT_VOCAB_SIZE)
        self.talker = Talker(
            config.model.talker,
            thinker_dim=config.model.thinker.d_model,
            conditioning_dim=config.model.conditioning_dim,
            codec=config.runtime.codec,
        )
        self.mtp = MTPHead(config.model.conditioning_dim, config.runtime.codec)
        self.code2wav = Code2Wav(config.model.conditioning_dim, config.runtime.codec)

    # ------------------------------------------------------------------ #
    # Training losses
    # ------------------------------------------------------------------ #

    def losses(self, sample: SampleTensors) -> dict[str, torch.Tensor]:
        cfg = self.config.training
        ids = torch.tensor(sample.ids, dtype=torch.long)

        # ---- Thinker: next-token CE over assistant spans ------------------- #
        out = self.thinker(ids.unsqueeze(0))
        logits = out.logits[0, :-1]          # position t-1 predicts ids[t]
        targets = ids[1:]
        mask = torch.zeros_like(targets, dtype=torch.bool)
        for s, e in sample.assistant_spans:
            mask[max(s - 1, 0): max(e - 1, 0)] = True
        if mask.any():
            text_loss = torch.nn.functional.cross_entropy(
                logits[mask], targets[mask]
            )
        else:
            text_loss = torch.zeros((), dtype=logits.dtype)

        # ---- conditioning tokens per slice --------------------------------- #
        cond_per_slice = []
        for (s, e) in sample.assistant_spans:
            cond_per_slice.append(
                self.talker.conditioning(out.embeddings[0, s:e], out.hiddens[0, s:e])
            )

        # ---- Talker: layer-0 CE over codec positions ----------------------- #
        codes_tensors = [
            torch.tensor(c, dtype=torch.long) for c in sample.codes_per_slice
        ]
        if not codes_tensors:
            return {"total": text_loss, "thinker": text_loss.detach()}
        input_embeds, positions, codec_positions, layer0_targets, _eos_pos = (
            self.talker.build_training_sequence(cond_per_slice, codes_tensors)
        )
        hiddens = self.talker.forward_embeds(input_embeds, positions)[0]  # (T, D)

        layer0_logits = self.talker.code_head(
            torch.cat([hiddens[torch.tensor(p)] for p in codec_positions])
        )
        layer0_target = torch.cat(layer0_targets)
        talker_loss = torch.nn.functional.cross_entropy(layer0_logits, layer0_target)

        # ---- MTP: residual codebooks from gold layer-0 + talker hiddens ----- #
        mtp_logits = self.mtp(
            self.talker.codebook_embeds[0](layer0_target),   # u_0(gold q^0)
            torch.cat([hiddens[torch.tensor(p)] for p in codec_positions]),
        )
        residual_target = torch.cat(
            [c[:, 1:] for c in codes_tensors], dim=0
        ).transpose(0, 1)  # (K-1, total_frames)
        mtp_loss = sum(
            torch.nn.functional.cross_entropy(mtp_logits[k], residual_target[k])
            for k in range(len(mtp_logits))
        ) / max(len(mtp_logits), 1)

        total = (
            cfg.loss_weight_thinker * text_loss
            + cfg.loss_weight_talker * (talker_loss + mtp_loss)
        )
        return {
            "total": total,
            "thinker": text_loss.detach(),
            "talker": talker_loss.detach(),
            "mtp": mtp_loss.detach(),
        }

    # ------------------------------------------------------------------ #
    # Runtime session (time-sliced inference)
    # ------------------------------------------------------------------ #

    def new_session(self) -> SessionGeneration:
        return SessionGeneration(self)

    # ------------------------------------------------------------------ #
    # Checkpointing
    # ------------------------------------------------------------------ #

    def save_checkpoint(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": self.config.to_dict(),
            "state_dict": {k: v.cpu() for k, v in self.state_dict().items()},
        }
        torch.save(payload, path)
        return path

    @classmethod
    def load_checkpoint(cls, path: str | Path) -> DuplexOmni:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        model = cls(DuplexOmniConfig.from_dict(payload["config"]))
        model.load_state_dict(payload["state_dict"])
        return model


@dataclass
class SessionGeneration:
    """KV-cached state for one full-duplex session.

    The interaction layer drives it slice by slice:

    1. :meth:`push_context` — feed user-side tokens (user text, ``ˆ``,
       ``<fragment>``, ``[PENDnS]``) as they occur;
    2. :meth:`begin_assistant_token` / :meth:`assistant_token` — generate
       the slice's assistant text token by token (each also recorded as a
       conditioning source);
    3. :meth:`finish_slice` — project the slice's (e, h) into conditioning
       tokens, append them + BOS to the Talker cache, then generate the
       slice's 6 codec frames (layer-0 AR + MTP residuals) and decode PCM.
    """

    model: DuplexOmni
    thinker_cache: KVCache = field(default_factory=KVCache.empty)
    talker_cache: KVCache = field(default_factory=KVCache.empty)
    _cond_e: list = field(default_factory=list)
    _cond_h: list = field(default_factory=list)
    _started: bool = False

    # -- thinker side --------------------------------------------------------- #

    @torch.no_grad()
    def push_context(self, ids: list[int]) -> None:
        """Feed context tokens (no loss, no conditioning collection)."""
        if not ids:
            return
        t = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
        self.model.thinker(t, cache=self.thinker_cache)
        self._started = True

    @torch.no_grad()
    def begin_assistant_token(self, token_id: int):
        """Feed the ``[A]`` marker; returns next-token logits and records cond."""
        return self._assistant_step(token_id)

    @torch.no_grad()
    def assistant_token(self, token_id: int):
        """Feed one generated assistant token; returns next-token logits."""
        return self._assistant_step(token_id)

    def _assistant_step(self, token_id: int):
        t = torch.tensor([[token_id]], dtype=torch.long)
        out = self.model.thinker(t, cache=self.thinker_cache)
        self._cond_e.append(out.embeddings[0, -1])
        self._cond_h.append(out.hiddens[0, -1])
        return out.logits[0, -1]

    # -- talker side ----------------------------------------------------------- #

    @torch.no_grad()
    def finish_slice(self) -> tuple[list[list[int]], bytes, torch.Tensor]:
        """Generate this slice's speech: returns (codes F x K, pcm, waveform)."""
        model = self.model
        codec = model.config.runtime.codec
        n_frames = codec.frames_per_slice

        if self._cond_e:
            e = torch.stack(self._cond_e)
            h = torch.stack(self._cond_h)
            cond = model.talker.conditioning(e, h)
        else:
            cond = torch.zeros((0, model.config.model.conditioning_dim))
        self._cond_e.clear()
        self._cond_h.clear()

        h_pred = model.talker.append_cond(cond, self.talker_cache)  # BOS position
        r_list: list = []
        codes: list[list[int]] = []
        for _f in range(n_frames):
            q0 = int(torch.argmax(model.talker.code_head(h_pred)))
            # MTP residual codebooks from the predicting position
            layer0_embed = model.talker.codebook_embeds[0](
                torch.tensor([q0])
            )
            residual_logits = model.mtp(layer0_embed, h_pred)
            residual = [int(torch.argmax(rl[0, 0])) for rl in residual_logits]
            frame_codes = [q0, *residual]
            codes.append(frame_codes)
            r = model.talker.sum_codec_embedding(
                torch.tensor([frame_codes])
            )  # (1, D)
            r_list.append(r[0])
            h_pred = model.talker.append_frames(r, self.talker_cache)  # (1, D)
        model.talker.append_eos(self.talker_cache)

        r_stack = torch.stack(r_list).unsqueeze(0)  # (1, F, D)
        waveform = model.code2wav(r_stack)[0]       # (F * frame_samples,)
        pcm = model.code2wav.to_pcm16(waveform)
        return codes, pcm, waveform

    # -- misc ------------------------------------------------------------------- #

    def token_to_id(self, text: str) -> int:
        return self.model.tokenizer.encode_control(text)


class AlternateTrainer:
    """Alternating Thinker/Talker optimisation (paper Sec. 4.1).

    Each step trains one side with the other frozen (requires_grad off),
    while the loss always combines both terms at the paper's 1:1 ratio.
    Gradient accumulation over ``batch_size`` samples replaces data padding;
    LR defaults follow the paper (Thinker 1e-5, Talker 1e-4).
    """

    def __init__(self, model: DuplexOmni, train_cfg=None) -> None:
        from ..config import TrainingConfig

        self.model = model
        self.cfg = train_cfg or model.config.training or TrainingConfig()
        self.opt_thinker = torch.optim.Adam(
            self.model.thinker.parameters(), lr=self.cfg.thinker_lr
        )
        self.opt_talker = torch.optim.Adam(
            list(self.model.talker.parameters())
            + list(self.model.mtp.parameters())
            + list(self.model.code2wav.parameters()),
            lr=self.cfg.talker_lr,
        )
        self.step_count = 0
        self.history: list[dict[str, float]] = []

    def talker_params(self):
        return (
            list(self.model.talker.parameters())
            + list(self.model.mtp.parameters())
            + list(self.model.code2wav.parameters())
        )

    def _set_requires_grad(self, module_params: list, flag: bool) -> None:
        for p in module_params:
            p.requires_grad_(flag)

    def train_step(self, samples: list[SampleTensors]) -> dict[str, float]:
        train_thinker = self.step_count % 2 == 0
        thinker_params = list(self.model.thinker.parameters())
        talker_params = self.talker_params()

        self.opt_thinker.zero_grad(set_to_none=True)
        self.opt_talker.zero_grad(set_to_none=True)
        self._set_requires_grad(thinker_params, train_thinker)
        self._set_requires_grad(talker_params, not train_thinker)

        agg: dict[str, float] = {"total": 0.0, "thinker": 0.0, "talker": 0.0, "mtp": 0.0}
        n = max(len(samples), 1)
        for sample in samples:
            losses = self.model.losses(sample)
            (losses["total"] / n).backward()
            for k in agg:
                agg[k] += float(losses.get(k, losses["total"]).detach()) / n

        if train_thinker:
            torch.nn.utils.clip_grad_norm_(thinker_params, 1.0)
            self.opt_thinker.step()
        else:
            torch.nn.utils.clip_grad_norm_(talker_params, 1.0)
            self.opt_talker.step()

        self._set_requires_grad(thinker_params, True)
        self._set_requires_grad(talker_params, True)
        self.step_count += 1
        self.history.append(agg)
        return agg

    def train(
        self,
        samples: list[SampleTensors],
        *,
        max_steps: int | None = None,
        log_every: int | None = None,
    ) -> list[dict[str, float]]:
        max_steps = max_steps or self.cfg.max_steps
        log_every = log_every or self.cfg.log_every
        n_batches = max(1, math.ceil(len(samples) / self.cfg.batch_size))
        for step in range(max_steps):
            batch = samples[(step % n_batches) * self.cfg.batch_size:
                            ((step % n_batches) + 1) * self.cfg.batch_size]
            stats = self.train_step(batch)
            if log_every and (step + 1) % log_every == 0:
                print(
                    f"step {step + 1}/{max_steps} "
                    f"total={stats['total']:.4f} thinker={stats['thinker']:.4f} "
                    f"talker={stats['talker']:.4f} mtp={stats['mtp']:.4f}"
                )
        return self.history


def train_two_stage(
    model: DuplexOmni,
    stage1_samples: list[SampleTensors],
    stage2_samples: list[SampleTensors],
    *,
    trainer: AlternateTrainer | None = None,
    stage1_steps: int | None = None,
    stage2_steps: int | None = None,
) -> AlternateTrainer:
    """Two-stage SFT (paper Sec. 4.1): stage 1 = large-scale speech
    interaction, stage 2 = high-quality interaction (+ video-call) data."""
    trainer = trainer or AlternateTrainer(model)
    print(f"[stage 1] {len(stage1_samples)} samples, {stage1_steps or trainer.cfg.max_steps} steps")
    trainer.train(stage1_samples, max_steps=stage1_steps)
    print(f"[stage 2] {len(stage2_samples)} samples, {stage2_steps or trainer.cfg.max_steps} steps")
    trainer.train(stage2_samples, max_steps=stage2_steps)
    return trainer
