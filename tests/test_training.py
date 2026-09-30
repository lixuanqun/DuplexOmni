"""Training tests: alternating optimisation, convergence on a tiny batch."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from duplexomni.config import tiny_config  # noqa: E402
from duplexomni.data import WriterDirectorPipeline  # noqa: E402
from duplexomni.model import AlternateTrainer, DuplexOmni, record_to_sample  # noqa: E402


@pytest.fixture(scope="module")
def samples(corpus_file):
    pipe = WriterDirectorPipeline(corpus_file, seed=21)
    records, stats = pipe.run(n=4)
    assert stats.n_filtered == 0, stats.warnings
    return [record_to_sample(r) for r in records]


def test_alternate_freezes_correct_side(samples):
    torch.manual_seed(0)
    model = DuplexOmni(tiny_config())
    trainer = AlternateTrainer(model)

    thinker_params = list(model.thinker.parameters())
    talker_params = trainer.talker_params()

    before_tk = [p.detach().clone() for p in thinker_params]
    before_ta = [p.detach().clone() for p in talker_params]

    trainer.train_step(samples[:2])  # even step -> thinker
    assert any(not torch.equal(a, b) for a, b in zip(before_tk, thinker_params, strict=False)), "thinker updated"
    assert all(torch.equal(a, b) for a, b in zip(before_ta, talker_params, strict=False)), "talker frozen"
    assert all(p.requires_grad for p in thinker_params), "requires_grad restored"

    before_tk = [p.detach().clone() for p in thinker_params]
    trainer.train_step(samples[:2])  # odd step -> talker
    assert all(torch.equal(a, b) for a, b in zip(before_tk, thinker_params, strict=False)), "thinker frozen"
    assert any(not torch.equal(a, b) for a, b in zip(before_ta, talker_params, strict=False)), "talker updated"


def test_lr_defaults_follow_paper():
    from duplexomni.config import paper_scale_config

    cfg = paper_scale_config()
    assert cfg.training.thinker_lr == pytest.approx(1e-5)
    assert cfg.training.talker_lr == pytest.approx(1e-4)
    assert cfg.training.batch_size == 128
    assert cfg.training.loss_weight_thinker == cfg.training.loss_weight_talker  # 1:1


def test_loss_decreases_on_overfit(samples):
    torch.manual_seed(0)
    model = DuplexOmni(tiny_config())
    trainer = AlternateTrainer(model)
    first = trainer.train_step(samples)
    for _ in range(24):
        last = trainer.train_step(samples)
    assert last["total"] < first["total"], (first["total"], last["total"])
    assert last["thinker"] < first["thinker"]
    assert last["talker"] < first["talker"]


def test_two_stage_driver(samples):
    torch.manual_seed(0)
    model = DuplexOmni(tiny_config())
    from duplexomni.model import train_two_stage

    trainer = train_two_stage(model, samples[:2], samples[:2], stage1_steps=2, stage2_steps=2)
    assert trainer.step_count == 4
