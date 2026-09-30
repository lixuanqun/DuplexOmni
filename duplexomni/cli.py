"""Command-line interface.

    duplexomni build-data --corpus corpus.jsonl --out data/train.jsonl -n 64
    duplexomni check       data/train.jsonl
    duplexomni train       --config configs/tiny.json --data data/train.jsonl
    duplexomni demo        --turns 3            # offline full-duplex simulation
    duplexomni serve       --port 8765          # websocket demo (see examples/)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _cmd_build_data(args) -> int:
    from .data import WriterDirectorPipeline

    pipe = WriterDirectorPipeline(args.corpus, seed=args.seed)
    records, stats = pipe.run(n=args.n, out=Path(args.out))
    print(stats.summary())
    print(f"wrote {stats.n_records} records -> {args.out}")
    return 0 if records else 1


def _cmd_check(args) -> int:
    from .data import iter_jsonl
    from .data.checks import check_annotated_dialog
    from .data.director import AnnotatedDialog, AnnotatedUtterance, UtteranceKind
    from .data.scenario import Pattern, ScenarioSeed
    from .tokens import Speaker

    n = bad = 0
    for rec in iter_jsonl(args.path):
        n += 1
        utterances = []
        for u in rec.annotated_script.splitlines():
            line = u.strip()
            if not line:
                continue
            if line.startswith("<") and line.endswith(">"):
                utterances.append(AnnotatedUtterance(Speaker.ASSISTANT, line, kind=UtteranceKind.EVENT))
            elif line.startswith("[PEND") and line.endswith("S]"):
                utterances.append(AnnotatedUtterance(Speaker.ASSISTANT, line, kind=UtteranceKind.SILENCE, duration_s=float(line[5:-2])))
            elif line[:3] in ("[U]", "[A]"):
                sp = Speaker.USER if line[:3] == "[U]" else Speaker.ASSISTANT
                utterances.append(AnnotatedUtterance(sp, line[3:].strip(), duration_s=1.0, overlap="ˆ" in line))
        seed = ScenarioSeed(scenario_id=rec.id, topic="", language=rec.language, patterns=[Pattern(p) for p in rec.patterns])
        result = check_annotated_dialog(AnnotatedDialog(seed=seed, utterances=utterances))
        if not result.ok:
            bad += 1
            print(f"[FAIL] {rec.id}: {result.errors}")
    print(f"checked {n} records, {bad} failed")
    return 1 if bad else 0


def _cmd_train(args) -> int:
    from .config import DuplexOmniConfig
    from .data import iter_jsonl
    from .model import AlternateTrainer, DuplexOmni, record_to_sample

    if args.config:
        config = DuplexOmniConfig.load(args.config)
    else:
        from .config import tiny_config

        config = tiny_config()
    records = list(iter_jsonl(args.data))
    if not records:
        print("no training data", file=sys.stderr)
        return 1
    samples = [record_to_sample(r) for r in records]
    print(f"config: {json.dumps({'model_dims': config.model.thinker.d_model, 'samples': len(samples)})}")

    import torch

    torch.manual_seed(config.seed)
    model = DuplexOmni(config)
    trainer = AlternateTrainer(model, config.training)
    trainer.train(samples, max_steps=args.steps or config.training.max_steps)

    out = Path(args.out or "checkpoints/duplexomni_tiny.pt")
    model.save_checkpoint(out)
    print(f"saved checkpoint -> {out}")
    return 0


def _cmd_demo(args) -> int:
    from .demo import run_offline_demo

    run_offline_demo(turns=args.turns, seed=args.seed, out_wav=args.wav)
    return 0


def _cmd_serve(args) -> int:
    print(
        "The websocket demo server lives in examples/websocket_demo.py "
        "(python examples/websocket_demo.py --port 8765).",
        file=sys.stderr,
    )
    return 2


def _cmd_bench(args) -> int:
    import torch

    from .config import tiny_config
    from .eval import format_report, run_benchmark
    from .model import DuplexOmni

    torch.manual_seed(args.seed)
    model = DuplexOmni(tiny_config())
    report = run_benchmark(model, seed=args.seed)
    print(format_report(report))
    if args.ablation:
        from .eval import format_ablation, run_thinking_ablation

        print()
        result = run_thinking_ablation(model, seed=args.seed)
        print(format_ablation(result))
    return 0 if report.overall == 1.0 else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="duplexomni",
        description="DuplexOmni reference implementation (arXiv:2606.09186)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("build-data", help="run the Writer-Director pipeline")
    p.add_argument("--corpus", required=True, help="corpus file (.jsonl/.json/.txt)")
    p.add_argument("--out", default="data/processed/train.jsonl")
    p.add_argument("-n", type=int, default=64, help="number of scenarios")
    p.add_argument("--seed", type=int, default=None)
    p.set_defaults(func=_cmd_build_data)

    p = sub.add_parser("check", help="run consistency checks on a dataset")
    p.add_argument("path")
    p.set_defaults(func=_cmd_check)

    p = sub.add_parser("train", help="train the tiny model on a sliced dataset")
    p.add_argument("--config", default=None, help="JSON config (configs/tiny.json)")
    p.add_argument("--data", required=True, help="dataset JSONL from build-data")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--out", default=None)
    p.set_defaults(func=_cmd_train)

    p = sub.add_parser("demo", help="offline full-duplex simulation")
    p.add_argument("--turns", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--wav", default=None, help="write mixed session audio here")
    p.set_defaults(func=_cmd_demo)

    p = sub.add_parser("bench", help="run the full-duplex behavioural benchmark")
    p.add_argument("--ablation", action="store_true", help="also run the thinking-layer ablation")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=_cmd_bench)

    p = sub.add_parser("serve", help="websocket demo server (see examples/)")
    p.add_argument("--port", type=int, default=8765)
    p.set_defaults(func=_cmd_serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
