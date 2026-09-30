"""End-to-end Writer-Director pipeline (paper Sec. 4.1).

    corpus --(scenario sampling)--> seeds
           --(Writer)--------------> natural scripts
           --(Director)------------> temporally annotated dialogs
           --(consistency checks)--> filtered dialogs
           --(TTS + time-slicing)--> 480 ms slice records (JSONL)

Usage::

    from duplexomni.data import WriterDirectorPipeline, MockWriter, RuleDirector, MockTTS

    pipe = WriterDirectorPipeline(corpus="corpus.jsonl")
    records, stats = pipe.run(n=16, out="data/processed/train.jsonl")
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from ..codecs import CodecBackend, MockCodec
from ..config import RuntimeConfig
from ..tokens import Speaker, parse_script
from .checks import check_annotated_dialog
from .director import AnnotatedDialog, Director, RuleDirector
from .scenario import Pattern, ScenarioSeed, pattern_coverage_stats, sample_scenario_seeds
from .slicer import DatasetRecord, slice_dialog, write_jsonl
from .synthesis import MockTTS, TTSBackend
from .writer import MockWriter, Writer

__all__ = ["PipelineStats", "WriterDirectorPipeline"]


@dataclass
class PipelineStats:
    n_seeds: int = 0
    n_written_ok: int = 0
    n_filtered: int = 0
    n_records: int = 0
    pattern_coverage: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        cov = " ".join(f"{k}={v:.1%}" for k, v in self.pattern_coverage.items())
        return (
            f"seeds={self.n_seeds} annotated_ok={self.n_written_ok} "
            f"filtered={self.n_filtered} records={self.n_records}\npatterns: {cov}"
        )


class WriterDirectorPipeline:
    def __init__(
        self,
        corpus: str | Path,
        *,
        writer: Writer | None = None,
        director: Director | None = None,
        tts: TTSBackend | None = None,
        codec: CodecBackend | None = None,
        runtime: RuntimeConfig | None = None,
        seed: int | None = None,
    ) -> None:
        self.corpus = corpus
        self.writer = writer or MockWriter()
        self.director = director or RuleDirector()
        self.tts = tts or MockTTS()
        self.runtime = runtime or RuntimeConfig()
        self.codec = codec or MockCodec(self.runtime.codec)
        self.seed = seed

    def run(
        self,
        n: int,
        *,
        out: str | Path | None = None,
        pattern_probs: dict[Pattern, float] | None = None,
        hooks: Callable[[AnnotatedDialog], None] | None = None,
    ) -> tuple[list[DatasetRecord], PipelineStats]:
        stats = PipelineStats(n_seeds=n)
        seeds = sample_scenario_seeds(
            self.corpus, n, seed=self.seed, probs=pattern_probs
        )
        records: list[DatasetRecord] = []
        dialogs: list[AnnotatedDialog] = []
        for seed_ in seeds:
            dialog = self._annotate_one(seed_, stats)
            if dialog is None:
                continue
            dialogs.append(dialog)
            slices, _tl = slice_dialog(
                dialog, self.tts, self.codec, runtime=self.runtime
            )
            records.append(
                DatasetRecord(
                    id=seed_.scenario_id,
                    language=seed_.language,
                    patterns=[p.value for p in seed_.patterns],
                    annotated_script=dialog.to_script(),
                    slices=slices,
                    total_ms=slices[-1].end_ms if slices else 0,
                )
            )
            if hooks:
                hooks(dialog)
        stats.n_records = len(records)
        stats.pattern_coverage = pattern_coverage_stats(
            [d.seed for d in dialogs]
        )
        if out is not None and records:
            stats.n_records = write_jsonl(records, out)
        return records, stats

    # -- internals ------------------------------------------------------------ #

    def _annotate_one(self, seed: ScenarioSeed, stats: PipelineStats) -> AnnotatedDialog | None:
        try:
            script = self.writer.write(seed)
            if not _script_has_both_speakers(script):
                stats.n_filtered += 1
                stats.warnings.append(f"{seed.scenario_id}: writer script missing a speaker")
                return None
            dialog = self.director.annotate(script, seed)
        except Exception as exc:  # noqa: BLE001 - pipeline must not die on one bad seed
            stats.n_filtered += 1
            stats.warnings.append(f"{seed.scenario_id}: {type(exc).__name__}: {exc}")
            return None
        result = check_annotated_dialog(dialog)
        if not result.ok:
            stats.n_filtered += 1
            stats.warnings.append(
                f"{seed.scenario_id}: rejected: {result.errors[:3]}"
            )
            return None
        stats.n_written_ok += 1
        return dialog


def _script_has_both_speakers(script: str) -> bool:
    speakers = {e.speaker for e in parse_script(script) if hasattr(e, "speaker")}
    return Speaker.USER in speakers and Speaker.ASSISTANT in speakers
