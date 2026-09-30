"""Build a full-duplex training dataset with the Writer-Director pipeline.

    python examples/build_dataset.py --corpus path/to/chat_corpus.jsonl -n 128

Any JSONL with `text`/`question` fields or OpenAI-style `messages` works
(UltraChat / WildChat / BELLE / COIG / no-robots / OASST2 exports all fit —
the corpora the paper uses).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root
import argparse
import json
import tempfile

from duplexomni.data import WriterDirectorPipeline


def make_demo_corpus(path: Path) -> None:
    records = [
        {"messages": [
            {"role": "user", "content": "帮我规划一个周末的北京周边游"},
            {"role": "assistant", "content": "好的，可以考虑古北水镇或者十渡"},
        ]},
        {"text": "how do I troubleshoot a wifi router that keeps dropping"},
        {"question": "全双工语音交互和半双工有什么区别？"},
        {"messages": [
            {"role": "user", "content": "plan a three day trip to Chengdu with friends"},
            {"role": "assistant", "content": "day one pandas, day two Leshan, day three food tour"},
        ]},
        {"text": "explain real-time speech translation pipelines in simple terms"},
    ]
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=None, help="chat corpus (.jsonl/.json/.txt)")
    ap.add_argument("-n", type=int, default=64)
    ap.add_argument("--out", default="data/processed/train.jsonl")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    corpus = args.corpus
    if corpus is None:
        corpus = str(Path(tempfile.gettempdir()) / "duplexomni_demo_corpus.jsonl")
        make_demo_corpus(Path(corpus))
        print(f"no --corpus given; wrote a demo corpus to {corpus}")

    pipe = WriterDirectorPipeline(corpus, seed=args.seed)
    records, stats = pipe.run(n=args.n, out=args.out)
    print(stats.summary())
    print(f"\nwrote {len(records)} records -> {args.out}")
    print("\nsample annotated script (first record):")
    print(records[0].annotated_script)


if __name__ == "__main__":
    main()
