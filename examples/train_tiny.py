"""Train the tiny DuplexOmni on a dataset from build_dataset.py.

    python examples/train_tiny.py --data data/processed/train.jsonl --steps 30
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root
import argparse

import torch

from duplexomni.config import tiny_config
from duplexomni.data import iter_jsonl
from duplexomni.model import AlternateTrainer, DuplexOmni, record_to_sample, train_two_stage


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--out", default="checkpoints/duplexomni_tiny.pt")
    args = ap.parse_args()

    config = tiny_config()
    torch.manual_seed(config.seed)
    model = DuplexOmni(config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model parameters: {n_params:,}")

    records = list(iter_jsonl(args.data))
    samples = [record_to_sample(r) for r in records]
    print(f"training samples: {len(samples)}")

    # paper-style two-stage SFT: stage 1 = all data, stage 2 = the same data
    # rehearsed (in a real run this is the high-quality + video-call subset)
    trainer = AlternateTrainer(model, config.training)
    split = max(1, len(samples) // 2)
    train_two_stage(
        model,
        samples[:split],
        samples[split:] or samples[:split],
        trainer=trainer,
        stage1_steps=args.steps,
        stage2_steps=max(5, args.steps // 2),
    )

    model.save_checkpoint(args.out)
    print(f"saved -> {args.out}")

    # sanity: session generation round-trip after training
    session = model.new_session()
    tk = model.tokenizer
    session.push_context([tk.bos()])
    session.assistant_token(tk.encode_control("[A]"))
    codes, pcm, wav = session.finish_slice()
    print(f"post-training slice: {len(codes)}x{len(codes[0])} codes, "
          f"{len(pcm)} pcm bytes, wav {tuple(wav.shape)}")


if __name__ == "__main__":
    main()
