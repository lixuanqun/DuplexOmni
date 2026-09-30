"""Run the behavioural benchmark + thinking-layer ablation (paper Sec. 5).

    python examples/run_benchmark.py --ablation
"""

from __future__ import annotations

import argparse

import torch

from duplexomni.config import tiny_config
from duplexomni.eval import format_ablation, format_report, run_benchmark, run_thinking_ablation
from duplexomni.model import DuplexOmni


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ablation", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    model = DuplexOmni(tiny_config())

    report = run_benchmark(model, seed=args.seed)
    print(format_report(report))

    if args.ablation:
        print()
        result = run_thinking_ablation(model, seed=args.seed)
        print(format_ablation(result))


if __name__ == "__main__":
    main()
