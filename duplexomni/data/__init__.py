"""Writer-Director data pipeline for full-duplex training data."""

from .checks import CheckResult, check_annotated_dialog, filter_dialogs
from .director import (
    AnnotatedDialog,
    AnnotatedUtterance,
    Director,
    LLMDirector,
    RuleDirector,
    UtteranceKind,
)
from .llm import LLMBackend, MockLLM, OpenAICompatLLM
from .pipeline import PipelineStats, WriterDirectorPipeline
from .scenario import (
    PAPER_PATTERN_PROBS,
    Pattern,
    ScenarioSeed,
    iter_corpus_texts,
    pattern_coverage_stats,
    sample_scenario_seeds,
)
from .slicer import DatasetRecord, Slice, iter_jsonl, slice_dialog, write_jsonl
from .synthesis import MockTTS, TTSBackend, UtteranceAudio, estimate_duration
from .writer import LLMWriter, MockWriter, Writer

__all__ = [
    "AnnotatedDialog",
    "AnnotatedUtterance",
    "CheckResult",
    "DatasetRecord",
    "Director",
    "LLMBackend",
    "LLMDirector",
    "LLMWriter",
    "MockLLM",
    "MockTTS",
    "OpenAICompatLLM",
    "MockWriter",
    "PAPER_PATTERN_PROBS",
    "Pattern",
    "PipelineStats",
    "RuleDirector",
    "ScenarioSeed",
    "Slice",
    "TTSBackend",
    "UtteranceAudio",
    "UtteranceKind",
    "Writer",
    "WriterDirectorPipeline",
    "check_annotated_dialog",
    "estimate_duration",
    "filter_dialogs",
    "iter_corpus_texts",
    "iter_jsonl",
    "pattern_coverage_stats",
    "sample_scenario_seeds",
    "slice_dialog",
    "write_jsonl",
]
