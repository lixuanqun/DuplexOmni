"""Thinking-layer ablation (paper Sec. 5, Table ablation).

The paper's finding: **full-duplex ability is independent of the thinking
layer's strength** (ToR 72.6 -> 72.1 with a weak layer) while the thinking
layer sets the reasoning ceiling.  This harness reproduces the systems-level
half of that experiment: swap the thinking layer (none / weak / strong) and
verify the *behavioural* policy score stays constant while delivered
fragment volume scales with layer strength.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .harness import run_benchmark
from .scenarios import policy_scenarios, thinking_scenarios

__all__ = ["AblationVariant", "run_thinking_ablation", "format_ablation"]

VARIANTS: dict[str, dict | None] = {
    "none": None,                       # no thinking layer at all
    "weak": {"delay_s": 0.0, "n_fragments": 1},
    "strong": {"delay_s": 0.0, "n_fragments": 6},
}


@dataclass
class AblationVariant:
    name: str
    policy_overall: float          # behavioural score over policy scenarios
    fragments: int                 # fragments delivered in thinking scenarios
    thinking_overall: float        # score over thinking scenarios
    max_rtf: float = 0.0
    detail: dict = field(default_factory=dict)


def run_thinking_ablation(model, *, seed: int = 0, verbose: bool = False) -> dict:
    policy = policy_scenarios()
    thinking_cases = thinking_scenarios()
    variants: dict[str, AblationVariant] = {}

    for name, cfg in VARIANTS.items():
        # policy scenarios run with no seeded thinking: they must behave
        # identically with any thinking layer — or with none (paper's finding)
        policy_report = run_benchmark(model, policy, seed=seed, thinking=None)

        if cfg is None:
            variants[name] = AblationVariant(
                name=name,
                policy_overall=policy_report.overall,
                fragments=0,
                thinking_overall=0.0,
                max_rtf=max(r.max_rtf for r in policy_report.results),
            )
            continue

        think_report = run_benchmark(model, thinking_cases, seed=seed, thinking=cfg)
        fragments = sum(r.fragments for r in think_report.results)
        variants[name] = AblationVariant(
            name=name,
            policy_overall=policy_report.overall,
            fragments=fragments,
            thinking_overall=think_report.overall,
            max_rtf=max(
                max(r.max_rtf for r in policy_report.results),
                max(r.max_rtf for r in think_report.results),
            ),
        )

    out = {"variants": variants}
    scores = {v.policy_overall for v in variants.values()}
    out["policy_independent_of_thinking"] = len(scores) == 1
    out["fragments_scale_with_strength"] = (
        variants["none"].fragments == 0
        and variants["weak"].fragments < variants["strong"].fragments
    )
    if verbose:
        print(format_ablation(out))
    return out


def format_ablation(result: dict) -> str:
    lines = ["thinking-layer ablation (paper Sec. 5)", "=" * 64]
    lines.append(f"{'variant':<10} {'policy score':>14} {'fragments':>11} {'max_rtf':>9}")
    for v in result["variants"].values():
        lines.append(f"{v.name:<10} {v.policy_overall:>13.0%} {v.fragments:>11} {v.max_rtf:>9.2f}")
    lines.append("-" * 64)
    lines.append(
        f"policy independent of thinking layer: {result['policy_independent_of_thinking']}"
    )
    lines.append(
        f"fragment volume scales with layer strength: {result['fragments_scale_with_strength']}"
    )
    return "\n".join(lines)
