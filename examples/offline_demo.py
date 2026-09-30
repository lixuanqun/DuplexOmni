"""Offline full-duplex demo: barge-in, async thinking, 480 ms streaming.

    python examples/offline_demo.py --turns 3 --wav outputs/session.wav
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root
import argparse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wav", default=None)
    args = ap.parse_args()

    from duplexomni.demo import run_offline_demo

    run_offline_demo(turns=args.turns, seed=args.seed, out_wav=args.wav)


if __name__ == "__main__":
    main()
