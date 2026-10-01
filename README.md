<div align="center">

# DuplexOmni

**Full-duplex voice agent: barge-in, task delegation, and async thinking, with a runnable browser gateway**

[English](README.md) | [简体中文](README.zh-CN.md)

[![CI](https://github.com/lixuanqun/DuplexOmni/actions/workflows/ci.yml/badge.svg)](https://github.com/lixuanqun/DuplexOmni/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![Paper](https://img.shields.io/badge/paper-arXiv%3A2606.09186-b31b1b.svg)](https://arxiv.org/abs/2606.09186)

*Reference implementation of
["DuplexOmni: Real-Time Listening, Seeing, Thinking, and Speaking for Full-Duplex Interaction"](https://arxiv.org/abs/2606.09186)
(Huang et al., 2026 — Xi'an Jiaotong University, Peking University, Meituan).*

**Keywords:** full-duplex speech, realtime voice assistant, voice agent, barge-in, turn-taking, streaming ASR, streaming TTS, spoken dialogue, task delegation, tool use, Thinker-Talker, WebSocket, duplex conversation

</div>

---

## What is this?

DuplexOmni is a full-duplex spoken-dialogue architecture: the model **listens, sees, thinks, and speaks at the same time**, instead of the classic half-duplex "user speaks → VAD → model responds" loop. The paper's key idea is to split the system into two asynchronously collaborating parts:

- an **interaction layer** — an end-to-end streaming speech model (480 ms time-sliced Thinker–Talker) that keeps the conversation going in real time, and
- a **thinking layer** — a pluggable LLM/agent that handles deep reasoning and tool use, dispatched non-blocking via `[THINK]` and streaming its result back as `<...>` fragments while the interaction layer keeps talking.

This repository is a **faithful, tested, fully runnable engineering rendition** of that method: the control-token grammar, the Writer–Director data pipeline, the Thinker–Talker-MTP model with alternating optimisation and KV-cached time-sliced inference, the async interaction/thinking runtime, and a behavioural benchmark with the paper's thinking-layer ablation.

You can also run it as a **voice agent** today: `python -m duplexomni serve` opens a browser session that keeps listening while it speaks, cuts playback on barge-in, delegates work (calculate, clock, memory) to a background worker, and streams the result back into the conversation. ASR and TTS are replaceable streaming engines. No GPU and no model weights are required for the default path.

**Paper results this architecture achieves** (DuplexBench v1.5 ToR / latency):

| Model | ToR ↑ | Latency ↓ |
|---|---|---|
| **DuplexOmni (paper)** | **72.6%** | 0.506 s |
| MiniCPM-o 4.5 | 36.3% | — |
| Doubao | 27.8% | — |
| Qwen3-Omni-Realtime-Flash | 25.2% | — |
| Gemini-3.1-Flash-Live | 24.1% | — |

> **Scope & honest limitations.** This repo ships the *method*, not the paper's 7B-scale weights. The bundled model is a tiny CPU-trainable configuration (`configs/tiny.json`, randomly initialised — demos exercise the full-duplex machinery, not language quality); `configs/paper_scale.json` documents the paper's hyperparameters. See [Limitations](#limitations) for details.

## Architecture

```
┌──────────────────────── interaction layer (streaming, RTF < 1) ────────────────────────┐
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

The two layers collaborate through a small control-token grammar (paper Appendix A):

| Token | Meaning |
|---|---|
| `[THINK]` | trigger the thinking layer; keep speaking while it works (**delayed reasoning**) |
| `<...>` | one streamed fragment of the thinking result, injected progressively |
| `ˆ` | overlap onset — user speech starts during assistant speech |
| `[CUT]` | stop current speech; the rest of the turn becomes *ghost text* (kept in history, never spoken) |
| `[WAIT]` | suspend/reset the pending reasoning request |
| `[PENDnS]` | `n` seconds of shared silence |

## Features

- **Control-token grammar** (`duplexomni/tokens.py`) — incremental streaming parser; token-by-token streaming and batch parsing are semantically identical, with robust handling of malformed input.
- **Writer–Director data pipeline** (`duplexomni/data/`) — chat corpora → scenario seeds (paper's pattern coverage: delayed reasoning 94.3%, shared silence 68.2%, assistant-initiated 50%, overlap 49.8%, interruption-with-reset 41.9%, backchannel 3.1%) → natural scripts → temporal control-token annotation → consistency checks → TTS → dual-track 480 ms slice records (JSONL). Rule-based backends run fully offline; LLM/TTS backends are pluggable.
- **Thinker–Talker-MTP model** (`duplexomni/model/`) — conditioning `c = f_text(e)+f_hidden(h)`, codec-token prefix `(C_i, BOS, R_i, EOS)`, layer-0 RVQ autoregression, MTP residual codebooks (`r = u_0(q⁰)+Σₖuₖ(qᵏ)`), Code2Wav; KV-cached incremental decoding **verified numerically equivalent** to the full forward; alternating Thinker/Talker optimisation (1:1 loss, paper LRs 1e-5/1e-4) with two-stage SFT driver.
- **Full-duplex runtime** (`duplexomni/runtime/`) — non-blocking `[THINK]` requests, progressive fragment injection, `[WAIT]` aborts on barge-in, floor discipline (never talks over the user), shared-silence handling, RTF budget measured per slice.
- **Evaluation** (`duplexomni/eval/`) — scripted behavioural benchmark (paper Sec. 5 methodology at the systems level) plus a none/weak/strong thinking-layer ablation.
- **Full-duplex gateway** (`duplexomni/gateway/`, `web/`) — browser UI over WebSocket, 20 ms PCM, server-side floor control, barge-in that cancels speech and the in-flight task, pluggable ASR/TTS/VAD, SQLite task log, optional bearer token, optional OpenAI-compatible thinking model.

## Installation

```bash
# with torch from PyPI
pip install -e ".[torch,dev]"

# or with CPU torch
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e .
```

The data pipeline and control-token grammar are **stdlib-only** (no torch needed); torch is required for `duplexomni.model`, the runtime demo, and the benchmark.

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

# 5) full-duplex gateway: browser conversation, task delegation, backend tools
python -m duplexomni serve
# open http://127.0.0.1:8765
```

Any JSONL/JSON/TXT chat corpus works for `build-data` — UltraChat / WildChat / BELLE / COIG / no-robots / OASST2 exports all fit (the corpora the paper uses). Without `--corpus`, a demo corpus is generated.

## Full-duplex gateway

`python -m duplexomni serve` (install with `pip install -e ".[serve]"`) binds `127.0.0.1:8765` and opens a browser UI:

- the interaction layer answers immediately. The server streams PCM and holds the floor for the audio it has sent. A barge-in stops playback and the in-flight task;
- ASR and TTS are replaceable engines. The default `scripted` engines do not decode speech and emit short, assertable PCM. Keyboard text uses the same final path. Swap in a model later with `DUPLEX_ASR` / `DUPLEX_TTS`;
- delegated tasks are stored in SQLite (`DUPLEX_DB`, default `duplexomni.sqlite`) and can be read back when the client reconnects with the same session id.

With no model configured, the worker only runs local tools: a restricted calculator, the clock, and session memory. There is no shell and no outbound network. Set `DUPLEX_TOKEN` to require that bearer token (or `token` query) on `/ws`; without it, only loopback peers are accepted. Point `DUPLEX_LLM_BASE_URL`, `DUPLEX_LLM_MODEL`, and `DUPLEX_LLM_API_KEY` at an OpenAI-compatible endpoint to enable the thinking model. Use headphones; quiet speaker echo is ignored while the server is playing.

## Evaluation

`duplexomni/eval/` mirrors the paper's evaluation *methodology* at the systems level (the bundled model is untrained, so language quality is out of scope — behaviour and timing are not):

```text
$ python -m duplexomni bench --ablation
full-duplex behavioural benchmark
================================================================
[PASS] floor-discipline     cat=interruption       frag=0   max_rtf=0.05
[PASS] turn-taking          cat=turn_taking        frag=0   max_rtf=0.03
[PASS] thinking-delivery    cat=delayed_reasoning  frag=3   max_rtf=0.05
[PASS] thinking-abort       cat=interruption_reset frag=0   max_rtf=0.04
----------------------------------------------------------------
interruption             100%
turn_taking              100%
delayed_reasoning        100%
interruption_reset       100%
OVERALL                  100%

thinking-layer ablation (paper Sec. 5)
================================================================
variant      policy score   fragments   max_rtf
none                100%           0      0.03
weak                100%           1      0.25
strong              100%           6      0.17
----------------------------------------------------------------
policy independent of thinking layer: True
fragment volume scales with layer strength: True
```

The ablation reproduces the paper's systems-level finding: swapping the thinking layer (none / weak / strong) leaves the full-duplex *behavioural* score unchanged while delivered fragment volume scales with layer strength. Steady-state RTF stays far below the paper's RTF < 1 budget even on CPU.

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
configs/                 tiny.json (CPU-runnable) · paper_scale.json (paper hyperparams)
examples/                build_dataset · train_tiny · offline_demo · run_benchmark ·
                         websocket_demo (minimal RFC 6455 server + client)
tests/                   70 unit/integration tests
docs/PAPER_ANALYSIS.md   deep-dive paper walkthrough in Chinese
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
| RTF < 1 target | `runtime/metrics.py`, `eval/harness.py` |
| Full-DuplexBench-style evaluation, thinking-layer ablation | `eval/*` |

## Testing

```bash
pytest                       # 70 tests: grammar, pipeline, checks, slicer, model
                             # shapes, KV-cache equivalence, alternating training,
                             # async bridge, runtime loop, benchmark, ablation
ruff check duplexomni tests examples
```

CI runs the full suite on Python 3.10–3.13 with CPU torch, plus a torch-free core job that proves the data pipeline and grammar need no heavy dependencies.

## Limitations

* **No pretrained weights** — the tiny model initialises randomly; demos exercise the full-duplex machinery (timing, events, barge-in, async thinking), not language quality. The paper's model is initialised from Qwen3-Omni and trained on ~3M conversations; that part needs the authors' released weights.
* **Mock codec/TTS** — deterministic placeholders; real Mimi / Qwen3-TTS plug in behind the `CodecBackend` / `TTSBackend` protocols.
* **Demo-grade turn-taking policy** — when to open/close an assistant turn and muting after `[CUT]` are implemented as runtime policy, not learned end-to-end as in the paper.
* **Video input** — represented as token hooks (`video_tokens`, video summary in the thinking context) but not encoded.

## Citation

If you use this implementation, please cite the original paper:

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

[MIT](LICENSE) — the implementation is independent of the paper's authors; for the paper's own weights/data releases, see the paper's repository.
