<div align="center">

# DuplexOmni

**全双工语音助手：边听边说、随时打断、任务委派与异步思考，并带可运行的浏览器网关**

[English](README.md) | 简体中文

[![CI](https://github.com/lixuanqun/DuplexOmni/actions/workflows/ci.yml/badge.svg)](https://github.com/lixuanqun/DuplexOmni/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![Paper](https://img.shields.io/badge/paper-arXiv%3A2606.09186-b31b1b.svg)](https://arxiv.org/abs/2606.09186)

*论文
["DuplexOmni: Real-Time Listening, Seeing, Thinking, and Speaking for Full-Duplex Interaction"](https://arxiv.org/abs/2606.09186)
（Huang 等，2026 —— 西安交通大学、北京大学、美团）的工程化参考实现。*

**关键词：** 全双工语音、实时语音助手、语音智能体、语音交互、打断、话轮、流式语音识别、流式语音合成、口语对话、任务委派、工具调用、Thinker-Talker、WebSocket、双工对话

</div>

---

## 这是什么？

DuplexOmni 是一个**全双工**口语对话架构：模型**同时**听、看、想、说，而不是传统半双工的"用户说完 → VAD 判停 → 模型响应"流程。论文的核心思想是把系统拆成两个异步协作的部分：

- **交互层（interaction layer）**——端到端流式语音模型（480ms 时间切片的 Thinker–Talker），负责实时对话不中断；
- **思考层（thinking layer）**——可插拔的 LLM/Agent，负责深度推理与工具调用；通过 `[THINK]` 非阻塞派发，结果以 `<...>` 片段流式回注，期间交互层照常听说。

本仓库是对该方法的**忠实、经过充分测试、完全可运行**的工程化实现：控制 token 文法、Writer–Director 数据管线、带交替优化与 KV-cache 时间切片推理的 Thinker–Talker-MTP 模型、异步交互/思考运行时，以及行为级评测基准与思考层消融实验。

也可以直接把它当成**语音助手**跑起来：`python -m duplexomni serve` 打开浏览器会话，边听边说，用户打断会停掉播音和进行中的任务，计算、时间和记事委派给后台执行，结果再送回对话。语音识别和合成是可替换的流式引擎。默认路径不需要 GPU，也不需要模型权重。

**该架构在论文中取得的结果**（DuplexBench v1.5 ToR / 延迟）：

| 模型 | ToR ↑ | 延迟 ↓ |
|---|---|---|
| **DuplexOmni（论文）** | **72.6%** | 0.506 s |
| MiniCPM-o 4.5 | 36.3% | — |
| Doubao | 27.8% | — |
| Qwen3-Omni-Realtime-Flash | 25.2% | — |
| Gemini-3.1-Flash-Live | 24.1% | — |

> **范围与如实声明。** 本仓库提供的是*方法实现*，不含论文的 7B 规模权重。附带的模型是可在 CPU 上训练的 tiny 配置（`configs/tiny.json`，随机初始化——演示验证的是全双工机制，而非语言质量）；`configs/paper_scale.json` 记录了论文的超参数。详见[局限](#局限)。

## 架构

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

两层之间通过一小组控制 token 文法协作（论文附录 A）：

| 控制符号 | 含义 |
|---|---|
| `[THINK]` | 触发思考层；等待期间照常说话（**延迟推理**） |
| `<...>` | 思考结果的一个流式片段，渐进注入 |
| `ˆ` | 重叠说话起点——助手说话期间用户开口 |
| `[CUT]` | 停止当前语音；其后文本成为 *ghost text*（保留在历史中但不合成语音） |
| `[WAIT]` | 挂起/重置当前推理请求 |
| `[PENDnS]` | `n` 秒共同沉默 |

## 特性

- **控制 token 文法**（`duplexomni/tokens.py`）——增量流式解析器；逐 token 流式解析与整段批解析语义完全一致，对非法输入有稳健处理。
- **Writer–Director 数据管线**（`duplexomni/data/`）——聊天语料 → 场景种子（论文的模式覆盖率：延迟推理 94.3%、共同沉默 68.2%、助手主动发起 50%、重叠说话 49.8%、打断重置 41.9%、backchannel 3.1%）→ 自然对话脚本 → 时序控制标注 → 一致性校验 → TTS 合成 → 双轨 480ms 切片记录（JSONL）。规则后端完全离线可跑；LLM/TTS 后端均可插拔。
- **Thinker–Talker-MTP 模型**（`duplexomni/model/`）——条件 token `c = f_text(e)+f_hidden(h)`、codec token 前缀 `(C_i, BOS, R_i, EOS)`、第 0 层 RVQ 自回归、MTP 残差码本（`r = u_0(q⁰)+Σₖuₖ(qᵏ)`）、Code2Wav；KV-cache 增量解码与全前向**数值等价**（有测试保证）；Thinker/Talker 交替优化（1:1 损失、论文学习率 1e-5/1e-4）+ 两阶段 SFT 驱动。
- **全双工运行时**（`duplexomni/runtime/`）——非阻塞 `[THINK]` 请求、渐进片段注入、打断时 `[WAIT]` 中止、地板纪律（绝不抢用户的话）、共同沉默处理、逐片 RTF 预算。
- **评测**（`duplexomni/eval/`）——脚本化行为基准（论文第 5 节方法学的系统级实现）+ none/weak/strong 思考层消融。
- **全双工网关**（`duplexomni/gateway/`、`web/`）——浏览器界面走 WebSocket，20ms PCM，服务端话轮控制，打断会取消播音和进行中的任务，ASR/TTS/VAD 可替换，任务写入 SQLite，可选 Bearer 令牌，可选 OpenAI 兼容思考模型。

## 安装

```bash
# 从 PyPI 安装 torch
pip install -e ".[torch,dev]"

# 或安装 CPU 版 torch
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e .
```

数据管线与控制 token 文法**仅依赖标准库**（无需 torch）；torch 用于 `duplexomni.model`、运行时演示与评测基准。

## 快速开始

```bash
# 1) 全双工仿真：打断、重叠、异步思考、RTF 指标
python -m duplexomni demo --wav outputs/session.wav

# 2) 用 Writer-Director 管线构建训练数据
python -m duplexomni build-data --corpus your_chat_corpus.jsonl -n 64 \
    --out data/processed/train.jsonl
python -m duplexomni check data/processed/train.jsonl

# 3) 训练 tiny 模型（两阶段 SFT、Thinker/Talker 交替优化）
python -m duplexomni train --config configs/tiny.json \
    --data data/processed/train.jsonl --steps 30

# 4) 行为基准 + 思考层消融（论文第 5 节）
python -m duplexomni bench --ablation

# 5) 全双工网关：浏览器对话、任务委派、后端工具
python -m duplexomni serve
# 打开 http://127.0.0.1:8765
```

`build-data` 支持任意 JSONL/JSON/TXT 聊天语料——UltraChat / WildChat / BELLE / COIG / no-robots / OASST2 的导出格式都可直接使用（即论文所用语料）。不传 `--corpus` 时会自动生成演示语料。

## 全双工网关

`python -m duplexomni serve` 把论文里的两层拆成可运行的服务，默认只监听 `127.0.0.1:8765`：

- **交互层**立刻接话。服务端下发 PCM，话轮按已发送的音频保持；用户开口或点打断会停掉播音和进行中的任务。
- **听和说**是可替换引擎，默认 `scripted`（不识音、用可断言的短 PCM）。键盘文本走同一条 final 路径。以后换模型只改 `DUPLEX_ASR` / `DUPLEX_TTS`。
- **委派**把计算、时间、记忆这类请求做成任务，不堵住 20ms 音频环。任务状态写入 SQLite（`DUPLEX_DB`，默认 `duplexomni.sqlite`），重连续上同一会话号可以读回。
- **执行层**跑受控工具，并把结果以片段送回交互层接着说。

不配置模型时，后端只用本地工具（安全四则运算、当前时间、本会话记忆），没有 shell，也没有外网请求。`DUPLEX_TOKEN` 有值时，`/ws` 必须带同样的 Bearer 或 `token` 查询参数；不设置时只接受本机连接。要接上思考模型：

```bash
set DUPLEX_LLM_BASE_URL=http://127.0.0.1:8000/v1
set DUPLEX_LLM_MODEL=your-model
set DUPLEX_LLM_API_KEY=your-key
python -m duplexomni serve
```

浏览器请用耳机。安装网关额外依赖：`pip install -e ".[serve]"`。

## 评测

`duplexomni/eval/` 在系统层面复现论文的评测*方法学*（附带模型未经训练，语言质量不在评测范围——但行为与时序在）：

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

消融实验复现了论文的系统级结论：替换思考层（无/弱/强）不影响全双工*行为*分数，而片段投递量随思考层强度增长。即使纯 CPU 运行，稳态 RTF 也远低于论文的 RTF < 1 预算。

## 仓库结构

```
duplexomni/
├── tokens.py            控制 token 文法 + 增量解析器（论文附录 A）
├── config.py            编解码 / 模型 / 训练 / 运行时配置
├── codecs.py            RVQ 编解码协议、mock 编解码器、标准库 WAV 读写
├── data/                Writer-Director 数据管线（论文 4.1 节）
│   ├── scenario.py        阶段 1：从聊天语料采样场景种子
│   ├── writer.py          自然脚本生成（LLM 或 mock）
│   ├── director.py        时序控制标注（规则或 LLM）
│   ├── checks.py          一致性校验 / 过滤
│   ├── synthesis.py       TTS 后端（mock 渲染真实音频）
│   ├── slicer.py          双轨时间线 + 480ms 切片记录
│   └── pipeline.py        端到端管线 → JSONL 数据集
├── model/               Thinker-Talker 模型（论文 3.2-3.3、4.1 节）
│   ├── tokenizer.py       字节级分词器 + 控制特殊符号
│   ├── transformer.py     共享因果块 + KV cache
│   ├── thinker.py         MLLM 主干 → 嵌入 E_t、隐状态 H_t
│   ├── talker.py          c = f_text(e)+f_hidden(h)；前缀 (C_i, BOS, R_i, EOS)
│   ├── mtp.py             残差码本预测头
│   ├── code2wav.py        codec 嵌入 → 480ms 波形
│   ├── collate.py         切片记录 → 训练样本
│   └── full.py            DuplexOmni、损失（1:1）、交替训练器、两阶段 SFT、
│                          KV-cache 会话生成
├── runtime/             全双工运行时（论文 3.1、3.4 节）
│   ├── thinking.py        可插拔思考层（echo / function / OpenAI 兼容）
│   ├── bridge.py          [THINK]/<...>/[WAIT] 异步协作桥
│   ├── interaction.py     480ms 切片循环（打断、重叠、静音）
│   ├── vad.py / metrics.py / session.py / events.py
├── eval/                行为基准（论文第 5 节）
│   ├── scenarios.py       脚本化 Full-DuplexBench 风格场景
│   ├── harness.py         运行器 + 分类得分（强制 RTF 预算）
│   └── ablation.py        none/weak/strong 思考层消融
├── demo.py              离线全双工仿真
└── cli.py               `python -m duplexomni ...`
configs/                 tiny.json（CPU 可跑）· paper_scale.json（论文超参数）
examples/                build_dataset · train_tiny · offline_demo · run_benchmark ·
                         websocket_demo（极简 RFC 6455 服务端 + 客户端）
tests/                   70 个单元/集成测试
docs/PAPER_ANALYSIS.md   论文深度解读（中文）
```

## 论文 → 代码映射

| 论文概念 | 代码模块 |
|---|---|
| 控制符号 `[THINK] <...> ˆ [CUT] [WAIT] [PENDNS]` | `duplexomni/tokens.py` |
| 480ms 时间切片推理、每片 6 个 Mimi 帧 | `config.py`、`runtime/interaction.py` |
| Thinker 嵌入 `E_t` / 隐状态 `H_t` → 条件 `c = f_text(e)+f_hidden(h)` | `model/thinker.py`、`model/talker.py` |
| Talker 前缀 `P_t = Σᵢ (Cᵢ, BOS, Rᵢ, EOS)`；第 0 层自回归 | `model/talker.py::build_training_sequence` |
| MTP 残差码本；`r = u₀(q⁰)+Σₖuₖ(qᵏ)` | `model/mtp.py`、`model/talker.py::sum_codec_embedding` |
| Code2Wav | `model/code2wav.py` |
| 两阶段 SFT、Thinker/Talker 交替优化（1:1 损失、LR 1e-5/1e-4） | `model/full.py::AlternateTrainer`、`train_two_stage` |
| 场景种子（62 万）→ Writer → Director → 校验 → TTS → 切片 | `data/*` |
| 携带对话/视频/任务状态的非阻塞思考请求 | `runtime/bridge.py`、`runtime/thinking.py` |
| 渐进片段注入、条件变化时中止 | `runtime/bridge.py::take_pending/halt` |
| RTF < 1 目标 | `runtime/metrics.py`、`eval/harness.py` |
| Full-DuplexBench 风格评测、思考层消融 | `eval/*` |

## 测试

```bash
pytest                       # 70 个测试：文法、管线、校验、切片、模型形状、
                             # KV-cache 等价性、交替训练、异步桥、运行时循环、
                             # 行为基准、消融
ruff check duplexomni tests examples
```

CI 在 Python 3.10–3.13（CPU torch）上运行完整测试，另设无 torch 核心任务，证明数据管线与文法不依赖任何重型依赖。

## 局限

* **不含预训练权重**——tiny 模型随机初始化；演示验证的是全双工机制（时序、事件、打断、异步思考），而非语言质量。论文模型从 Qwen3-Omni 初始化并在约 300 万轮对话上训练，这部分需要作者发布的权重。
* **mock 编解码/TTS**——确定性占位实现；真实 Mimi / Qwen3-TTS 可通过 `CodecBackend` / `TTSBackend` 协议接入。
* **演示级轮替策略**——助手话轮的开/关与 `[CUT]` 后静音是运行时策略实现，而非论文中的端到端学习。
* **视频输入**——以 token 钩子表示（`video_tokens`、思考上下文中的视频摘要），未做编码。

## 引用

如果本实现对你有帮助，请引用原论文：

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

## 许可证

[MIT](LICENSE)——本实现独立于论文作者；论文自身的权重/数据发布请见论文仓库。
