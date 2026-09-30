"""Model structure tests: conditioning formula, prefix layout, MTP, caches.

These tests need PyTorch.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from duplexomni.config import (  # noqa: E402
    DuplexOmniConfig,
    ModelConfig,
    RuntimeConfig,
    TalkerConfig,
    ThinkerConfig,
)
from duplexomni.model import TEXT_VOCAB_SIZE, DuplexOmni  # noqa: E402
from duplexomni.model.transformer import KVCache  # noqa: E402


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    cfg = DuplexOmniConfig(
        model=ModelConfig(
            thinker=ThinkerConfig(d_model=32, n_heads=4, n_layers=2),
            talker=TalkerConfig(d_model=32, n_heads=4, n_layers=2),
            conditioning_dim=32,
        ),
        runtime=RuntimeConfig(),
    )
    return DuplexOmni(cfg)


def test_thinker_forward_shapes(model):
    ids = torch.randint(0, TEXT_VOCAB_SIZE, (1, 17))
    out = model.thinker(ids)
    assert out.logits.shape == (1, 17, TEXT_VOCAB_SIZE)
    assert out.embeddings.shape == (1, 17, 32)
    assert out.hiddens.shape == (1, 17, 32)


def test_conditioning_formula(model):
    """c must equal f_text(e) + f_hidden(h) (paper Sec. 3.3)."""
    e = torch.randn(2, 5, 32)
    h = torch.randn(2, 5, 32)
    c = model.talker.conditioning(e, h)
    expected = model.talker.f_text(e) + model.talker.f_hidden(h)
    assert torch.allclose(c, expected, atol=1e-6)


def test_codec_embedding_sum(model):
    """r = u_0(q^0) + sum_k u_k(q^k)."""
    codes = torch.randint(0, 2048, (6, 8))
    r = model.talker.sum_codec_embedding(codes)
    expected = model.talker.codebook_embeds[0](codes[:, 0])
    for k in range(1, 8):
        expected = expected + model.talker.codebook_embeds[k](codes[:, k])
    assert torch.allclose(r, expected, atol=1e-6)
    assert r.shape == (6, 32)


def test_prefix_layout(model):
    """Sequence layout: per slice [C_i, BOS, R_i (6 frames), EOS]."""
    cond = [torch.randn(4, 32), torch.randn(2, 32)]
    codes = [torch.randint(0, 2048, (6, 8)) for _ in range(2)]
    ie, pos, codec_positions, layer0, eos_pos = model.talker.build_training_sequence(cond, codes)
    T = (4 + 1 + 6 + 1) + (2 + 1 + 6 + 1)
    assert ie.shape == (1, T, 32)
    # slice 0: BOS at 4, frames r_0..r_5 at 5..10, EOS input at 11
    assert codec_positions[0] == [4, 5, 6, 7, 8, 9]  # BOS + r_0..r_4 predict frames
    # slice 1: cond(2) + BOS at 14, frames 15..20, EOS input at 21
    assert codec_positions[1] == [14, 15, 16, 17, 18, 19]
    assert all(len(p) == 6 for p in codec_positions)
    # r_last position predicts the slice's EOS
    assert eos_pos == [10, 20]
    assert [len(c) for c in layer0] == [6, 6]


def test_mtp_heads(model):
    codes = torch.randint(0, 2048, (6, 8))
    r0 = model.talker.codebook_embeds[0](codes[:, 0])
    hidden = torch.randn(6, 32)
    logits = model.mtp(r0.unsqueeze(0), hidden.unsqueeze(0))
    assert len(logits) == 7  # K-1 residual codebooks
    assert all(lg.shape == (1, 6, 2048) for lg in logits)


def test_code2wav_output_is_480ms(model):
    r = torch.randn(1, 6, 32)
    wav = model.code2wav(r)
    assert wav.shape == (1, 6 * 1280)
    assert float(wav.detach().abs().max()) <= 1.0


def test_thinker_kv_cache_equivalence(model):
    ids = torch.randint(0, TEXT_VOCAB_SIZE, (1, 23))
    full = model.thinker(ids)
    cache = KVCache.empty()
    chunks = []
    step = 5
    for i in range(0, 23, step):
        chunks.append(model.thinker(ids[:, i: i + step], cache=cache).logits)
    inc = torch.cat(chunks, dim=1)
    assert torch.allclose(full.logits, inc, atol=1e-5)


def test_talker_kv_cache_equivalence(model):
    cond = torch.randn(5, 32)
    codes = torch.randint(0, 2048, (6, 8))
    ie, pos, cp, lt, ep = model.talker.build_training_sequence([cond], [codes])
    h_full = model.talker.forward_embeds(ie, pos)[0]

    cache = KVCache.empty()
    h_bos = model.talker.append_cond(cond, cache)
    h_frames = model.talker.append_frames(model.talker.sum_codec_embedding(codes), cache)
    model.talker.append_eos(cache)

    assert torch.allclose(h_full[5], h_bos, atol=1e-5)
    assert torch.allclose(h_full[6:12], h_frames, atol=1e-5)
    assert cache.length == ie.shape[1]


def test_session_generation_slice(model):
    tk = model.tokenizer
    sess = model.new_session()
    sess.push_context([tk.bos(), tk.encode_control("[U]"), *tk.encode_text("hello"), tk.encode_control("[A]")])
    logits = None
    for tok in [tk.encode_control("[A]"), *tk.encode_text("hi")]:
        logits = sess.assistant_token(tok)
    assert logits.shape == (TEXT_VOCAB_SIZE,)
    codes, pcm, wav = sess.finish_slice()
    assert len(codes) == 6 and len(codes[0]) == 8
    assert wav.shape == (6 * 1280,)
    assert len(pcm) == 6 * 1280 * 2
    # second slice continues from caches without error
    sess.assistant_token(tk.encode_control("[A]"))
    codes2, _, _ = sess.finish_slice()
    assert len(codes2) == 6


def test_losses_structure(model):
    from duplexomni.data.director import AnnotatedDialog, AnnotatedUtterance, UtteranceKind
    from duplexomni.data.scenario import Pattern, ScenarioSeed
    from duplexomni.data.slicer import DatasetRecord, slice_dialog
    from duplexomni.data.synthesis import MockTTS
    from duplexomni.model import record_to_sample
    from duplexomni.tokens import Speaker

    seed = ScenarioSeed(
        scenario_id="m-0", topic="t", language="en",
        patterns=[Pattern.DELAYED_REASONING, Pattern.OVERLAP],
    )
    dialog = AnnotatedDialog(seed=seed, utterances=[
        AnnotatedUtterance(Speaker.USER, "hello friend how are you", duration_s=1.2),
        AnnotatedUtterance(Speaker.ASSISTANT, "I am great, let me think [THINK] about it", duration_s=2.0),
        AnnotatedUtterance(Speaker.ASSISTANT, "<thinking result>", kind=UtteranceKind.EVENT),
        AnnotatedUtterance(Speaker.ASSISTANT, "here is my answer", duration_s=1.4),
    ])
    slices, _tl = slice_dialog(dialog, MockTTS(), runtime=model.config.runtime)
    rec = DatasetRecord(id="m-0", language="en", patterns=["p"],
                        annotated_script=dialog.to_script(), slices=slices,
                        total_ms=slices[-1].end_ms)
    sample = record_to_sample(rec)
    losses = model.losses(sample)
    for key in ("total", "thinker", "talker", "mtp"):
        assert key in losses and torch.isfinite(losses[key]).all()
    # random-init losses are near the entropy of their vocabularies
    assert 4.0 < float(losses["thinker"]) < 7.0
    assert 6.5 < float(losses["talker"]) < 9.0


def test_checkpoint_roundtrip(model, tmp_path):
    path = model.save_checkpoint(tmp_path / "ckpt.pt")
    loaded = DuplexOmni.load_checkpoint(path)
    assert torch.equal(
        loaded.thinker.lm_head.weight, model.thinker.lm_head.weight
    )
