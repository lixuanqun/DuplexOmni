# DuplexOmni (reference implementation)

[![CI](https://github.com/lixuanqun/DuplexOmni/actions/workflows/ci.yml/badge.svg)](https://github.com/lixuanqun/DuplexOmni/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

An engineering reference implementation of **[DuplexOmni: Real-Time Listening,
Seeing, Thinking, and Speaking for Full-Duplex Interaction](https://arxiv.org/abs/2606.09186)**
(Huang et al., 2026) — a full-duplex spoken-dialogue architecture that decouples
real-time interaction from deep reasoning:

```
┌──────────────────────── interaction layer (streaming, < 1 RTF) ────────────────────────┐
│                                                                                         │
│  user audio/video ──► ┌─────────┐  embeddings E_t   ┌─────────┐   layer-0 RVQ   ┌─────┐ │
│  (480 ms slices)      │ Thinker │ ───────────────► │  Talker │ ──────────────► │ MTP │ │ │
│                       │ (MLLM)  │  hiddens H_t     │   AR    │  + residuals    └──┬──┘ │ │
│                       └────┬────┘  c = f_text(e)+f_hidden(h) └─────────┘  r = Σu_k(qᵏ)│ │
│                            │                                   └──► Code2Wav ──► 🎤  │ │
└──────────┬─────────────────▲───────────────────────────────────────────────────────────┘
     [THINK] (non-blocking)  │  <fragment> progressive injection
     [WAIT]  (abort/reset)   │
┌──────────▼─────────────────┴───────────────────────────────────────────────────────────┐
│                     thinking layer (pluggable LLM / agent, async)                       │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

The two layers collaborate through a small control-token grammar
([paper Appendix A](docs/PAPER_ANALYSIS.md)):

| Token | Meaning |
|---|---|
| `[THINK]` | trigger the thinking layer; keep speaking while it works (delayed reasoning) |
| `<...>` | one streamed fragment of the thinking result, injected progressively |
| `ˆ` | overlap onset — user speech starts during assistant speech |
| `[CUT]` | stop current speech; the rest of the turn becomes *ghost text* (kept in history, never spoken) |
| `[WAIT]` | suspend/reset the pending reasoning request |
| `[PENDnS]` | `n` seconds of shared silence |

**Scope.** This repo is a faithful, tested, runnable *engineering* rendition of
the paper's method — control-token grammar, Writer-Director data pipeline,
Thinker-Talker-MTP model with alternating optimisation and KV-cached
time-sliced inference, and the async interaction/thinking runtime. It does
**not** ship the paper's 7B-scale weights: the bundled model is a tiny
CPU-trainable configuration (`configs/tiny.json`); `configs/paper_scale.json`
documents the paper's hyperparameters.

## Install

```bash
pip install -e ".[torch,dev]"        # torch from PyPI, or:
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e .
```

The data pipeline and control-token grammar are **stdlib-only** (no torch
needed); torch is required for `duplexomni.model` and the runtime demo.

## Quickstart

```bash
# 1) full-duplex simulation: barge-in, overlap, async thinking, RTF metrics
python -m duplexomni demo --wav outputs/session.wav

# 2) build a training dataset with the Writer-Director pipeline
python -m duplexomni build-data --corpus your_chat_corpus.jsonl -n 64 \
    --out data/processed/train.jsonl
python -m duplexomni check data/processed/train.jsonl

# 3) train the tiny model (two-stage SFT, alternating Thinker/Talker)
python -m duplexomni train --config configs/tiny.json \
    --data data/processed/train.jsonl --steps 30

# 4) behavioural benchmark + thinking-layer ablation (paper Sec. 5)
python -m duplexomni bench --ablation

# 5) websocket demo (minimal RFC 6455 server + simulated-mic client)
python examples/websocket_demo.py --serve --port 8765 &
python examples/websocket_demo.py --client --port 8765 --wav outputs/ws.wav
```

See `examples/` and `docs/PAPER_ANALYSIS.md` (Chinese) for a detailed paper
walkthrough and the concept→code mapping.

## Repository layout

```
duplexomni/
├── tokens.py            control-token grammar + incremental parser  (paper App. A)
├── config.py            codec / model / training / runtime configs
├── codecs.py            RVQ codec protocol, mock codec, stdlib WAV IO
├── data/                Writer-Director pipeline (paper Sec. 4.1)
│   ├── scenario.py        stage 1: scenario seeds from chat corpora
│   ├── writer.py          natural script generation (LLM or mock)
│   ├── director.py        temporal control-token annotation (rule or LLM)
│   ├── checks.py          consistency checks / filtering
│   ├── synthesis.py       TTS backends (mock renders real audio)
│   ├── slicer.py          dual-track timeline + 480 ms slice records
│   └── pipeline.py        end-to-end pipeline -> JSONL dataset
├── model/               Thinker-Talker model (paper Sec. 3.2-3.3, 4.1)
│   ├── tokenizer.py       byte-level tokenizer + control specials
│   ├── transformer.py     shared causal blocks with KV cache
│   ├── thinker.py         MLLM backbone -> embeddings E_t, hiddens H_t
│   ├── talker.py          c = f_text(e)+f_hidden(h); prefix (C_i, BOS, R_i, EOS)
│   ├── mtp.py             residual codebook prediction heads
│   ├── code2wav.py        codec embeddings -> 480 ms waveform
│   ├── collate.py         slice records -> training samples
│   └── full.py            DuplexOmni, losses (1:1), AlternateTrainer,
│                          two-stage SFT, KV-cached session generation
├── runtime/             full-duplex runtime (paper Sec. 3.1, 3.4)
│   ├── thinking.py        pluggable thinking layer (echo/function/OpenAI-compat)
│   ├── bridge.py          [THINK]/<...>/[WAIT] async collaboration
│   ├── interaction.py     the 480 ms slice loop (barge-in, overlap, muting)
│   ├── vad.py / metrics.py / session.py / events.py
├── eval/                behavioural benchmark (paper Sec. 5)
│   ├── scenarios.py       scripted Full-DuplexBench-style cases
│   ├── harness.py         runner + per-category scores (RTF budget enforced)
│   └── ablation.py        none/weak/strong thinking-layer ablation
├── demo.py              offline full-duplex simulation
└── cli.py               `python -m duplexomni ...`
tests/                   70 unit/integration tests (grammar, pipeline, model,
                         caches, training, runtime, benchmark)
```

## Paper → code mapping

| Paper concept | Module |
|---|---|
| control symbols `[THINK] <...> ˆ [CUT] [WAIT] [PENDNS]` | `duplexomni/tokens.py` |
| 480 ms time-sliced inference, 6 Mimi frames/slice | `config.py`, `runtime/interaction.py` |
| Thinker embeddings `E_t` / hiddens `H_t` → conditioning `c = f_text(e)+f_hidden(h)` | `model/thinker.py`, `model/talker.py` |
| Talker prefix `P_t = Σᵢ (Cᵢ, BOS, Rᵢ, EOS)`; layer-0 AR | `model/talker.py::build_training_sequence` |
| MTP residual codebooks; `r = u₀(q⁰)+Σₖuₖ(qᵏ)` | `model/mtp.py`, `model/talker.py::sum_codec_embedding` |
| Code2Wav | `model/code2wav.py` |
| two-stage SFT, alternating Thinker/Talker optimisation (1:1 loss, LR 1e-5/1e-4) | `model/full.py::AlternateTrainer`, `train_two_stage` |
| scenario seeds (620K) → Writer → Director → checks → TTS → slicing | `data/*` |
| non-blocking thinking request with dialogue/video/task context | `runtime/bridge.py`, `runtime/thinking.py` |
| progressive fragment injection, halt on condition change | `runtime/bridge.py::take_pending/halt` |
| RTF < 1 target | `runtime/metrics.py` |

## Evaluation

`duplexomni/eval/` mirrors the paper's evaluation *methodology* at the
systems level (the bundled model is untrained, so language quality is out of
scope — behaviour and timing are not):

```text
$ python -m duplexomni bench --ablation
full-duplex behavioural benchmark
================================================================
[PASS] floor-discipline     cat=interruption       frag=0   max_rtf=0.11
[PASS] turn-taking          cat=turn_taking        frag=0   max_rtf=0.07
[PASS] thinking-delivery    cat=delayed_reasoning  frag=3   max_rtf=0.17
[PASS] thinking-abort       cat=interruption_reset frag=0   max_rtf=0.20
----------------------------------------------------------------
interruption             100%
turn_taking              100%
delayed_reasoning        100%
interruption_reset       100%
OVERALL                  100%

thinking-layer ablation (paper Sec. 5)
================================================================
variant      policy score   fragments   max_rtf
none                100%           0      0.19
weak                100%           1      0.19
strong              100%           6      0.16
----------------------------------------------------------------
policy independent of thinking layer: True
fragment volume scales with layer strength: True
```

The ablation reproduces the paper's systems-level finding: swapping the
thinking layer (none / weak / strong) leaves the full-duplex *behavioural*
score unchanged while delivered fragment volume scales with layer strength.

## Testing

```bash
pytest            # 65 tests: grammar, pipeline, checks, slicer, model shapes,
                    # KV-cache equivalence, alternating training, async bridge
```

CI additionally runs the core (torch-free) tests on a bare interpreter and the
full suite on Python 3.10–3.13 with CPU torch.

## Honest limitations

* No pretrained weights: the tiny model initialises randomly — demos exercise
  the full-duplex machinery (timing, events, barge-in, async thinking), not
  language quality. The paper's model is initialised from Qwen3-Omni and
  trained on 3M conversations; that part needs the authors' released weights.
* The mock codec/TTS are deterministic placeholders (Mimi/Qwen3-TTS plug in
  behind `CodecBackend`/`TTSBackend`).
* Runtime turn-taking policy (when to open/close an assistant turn, muting
  after `[CUT]`) is implemented at the interaction layer as demo-grade policy,
  not learned end-to-end as in the paper.
* Video input is represented as token hooks (`video_tokens`, video summary in
  the thinking context) but not encoded.

## Citation

```bibtex
@article{huang2026duplexomni,
  title={DuplexOmni: Real-Time Listening, Seeing, Thinking, and Speaking for
         Full-Duplex Interaction},
  author={Huang, Muye and Zhang, Lingling and Yu, Xingyu and Shi, Lei and Ma,
          Zhanyu and Xu, Jun and Gao, Jiuchong and Hao, Jinghua and He,
          Renqing and Liu, Jun},
  journal={arXiv preprint arXiv:2606.09186},
  year={2026}
}
```

## License

MIT — see [LICENSE](LICENSE).
