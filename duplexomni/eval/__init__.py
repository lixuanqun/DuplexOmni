"""Evaluation: behavioural benchmark + thinking-layer ablation (paper Sec. 5)."""

from .ablation import AblationVariant, format_ablation, run_thinking_ablation
from .harness import BenchReport, CaseResult, format_report, run_benchmark, run_case
from .scenarios import (
    Scenario,
    UserTurn,
    default_scenarios,
    policy_scenarios,
    thinking_scenarios,
)

__all__ = [
    "AblationVariant",
    "BenchReport",
    "CaseResult",
    "Scenario",
    "UserTurn",
    "default_scenarios",
    "format_ablation",
    "format_report",
    "policy_scenarios",
    "run_benchmark",
    "run_case",
    "run_thinking_ablation",
    "thinking_scenarios",
]
