"""Input-integrity checks shared by the judge-time guards and the ``audit`` command.

These are the checks that caught two live boards' problems on 2026-07-08: a model
whose whole column was error sentinels (issue #46) and truncation exposure on a
long-text corpus. Both the judge and the read-only ``audit`` command call the same
functions here so the pre-flight report and the guard can never disagree.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any

from ocr_bench.dataset import (
    AlignmentResult,
    _load_configs,
    check_config_alignment,
    discover_configs,
    discover_pr_configs,
    load_flat_dataset,
)
from ocr_bench.judge import MAX_OCR_TEXT_LENGTH, is_sentinel

# A model with more than this fraction of error sentinels is flagged: the run
# effectively failed on this corpus, so its outputs are not comparable.
SENTINEL_FLAG_RATE = 0.10

# Outputs shorter than this (non-empty, non-sentinel) are suspiciously terse —
# often a partial page or a refusal fragment.
SHORT_TEXT_LEN = 20


def _column_values(dataset: Any, col: str) -> list:
    """Read a column as a plain list, tolerating HF Datasets and lists of dicts.

    Column access on an HF Dataset avoids triggering per-row image decode.
    """
    if hasattr(dataset, "column_names"):
        return list(dataset[col])
    return [row.get(col) for row in dataset]


@dataclass
class ColumnStats:
    """Health statistics for one OCR output column (one model's transcriptions)."""

    name: str  # config or column name
    model: str  # resolved model id / label
    n_rows: int
    n_empty: int  # empty or whitespace-only
    n_sentinel: int  # error sentinels (see judge.is_sentinel)
    n_short: int  # non-empty, non-sentinel, < SHORT_TEXT_LEN chars
    n_over_max: int  # length > max_ocr_text_len (will be truncated by the judge)
    median_len: float
    max_len: int
    max_ocr_text_len: int

    def _rate(self, n: int) -> float:
        return n / self.n_rows if self.n_rows else 0.0

    @property
    def empty_rate(self) -> float:
        return self._rate(self.n_empty)

    @property
    def sentinel_rate(self) -> float:
        return self._rate(self.n_sentinel)

    @property
    def short_rate(self) -> float:
        return self._rate(self.n_short)

    @property
    def over_max_rate(self) -> float:
        return self._rate(self.n_over_max)

    @property
    def failed(self) -> bool:
        """True if the sentinel rate is high enough to treat the run as failed."""
        return self.sentinel_rate > SENTINEL_FLAG_RATE


def compute_column_stats(
    name: str,
    model: str,
    texts: list,
    max_ocr_text_len: int = MAX_OCR_TEXT_LENGTH,
) -> ColumnStats:
    """Compute :class:`ColumnStats` for one column's raw text values.

    Rows are partitioned into empty / sentinel / short / normal (non-overlapping),
    while length statistics are taken over every row's raw character length —
    matching how the judge truncates ``text[:max_ocr_text_len]``.
    """
    n_empty = n_sentinel = n_short = 0
    lengths: list[int] = []
    for t in texts:
        s = t or ""
        lengths.append(len(s))
        stripped = s.strip()
        if not stripped:
            n_empty += 1
        elif is_sentinel(s):
            n_sentinel += 1
        elif len(stripped) < SHORT_TEXT_LEN:
            n_short += 1

    n_over = sum(1 for length in lengths if length > max_ocr_text_len)
    return ColumnStats(
        name=name,
        model=model,
        n_rows=len(texts),
        n_empty=n_empty,
        n_sentinel=n_sentinel,
        n_short=n_short,
        n_over_max=n_over,
        median_len=statistics.median(lengths) if lengths else 0.0,
        max_len=max(lengths) if lengths else 0,
        max_ocr_text_len=max_ocr_text_len,
    )


def compute_model_stats(
    dataset: Any,
    ocr_columns: dict[str, str],
    max_ocr_text_len: int = MAX_OCR_TEXT_LENGTH,
) -> list[ColumnStats]:
    """Per-model health stats over a loaded (merged or flat) dataset.

    ``ocr_columns`` maps column/config name → model label. Used by the judge to
    derive per-model ``failed_outputs`` and the >10% sentinel-rate warning.
    """
    return [
        compute_column_stats(col, model, _column_values(dataset, col), max_ocr_text_len)
        for col, model in ocr_columns.items()
    ]


def failed_output_counts(stats: list[ColumnStats]) -> dict[str, int]:
    """Map model → sentinel count, keeping only models with at least one."""
    return {s.model: s.n_sentinel for s in stats if s.n_sentinel}


@dataclass
class ConfigAudit:
    """Audit result for a single config/column."""

    stats: ColumnStats


@dataclass
class AuditReport:
    """Read-only pre-judge health report for an output repo."""

    repo_id: str
    configs: list[ConfigAudit]
    alignment: AlignmentResult
    max_ocr_text_len: int

    @property
    def flagged_models(self) -> list[str]:
        """Models whose sentinel rate exceeds the failure threshold."""
        return [c.stats.model for c in self.configs if c.stats.failed]

    @property
    def row_count_mismatch(self) -> bool:
        """True if the configs do not all cover the same number of rows.

        A positional merge of differing-length configs pairs different pages —
        the silent-corruption path issue #5 guards against — so the judge would
        raise. The audit flags it here rather than letting judging discover it.
        """
        return len({c.stats.n_rows for c in self.configs}) > 1

    @property
    def has_problems(self) -> bool:
        """True if the repo should block an automated judge run.

        A real misalignment, a row-count mismatch, or any config over the
        sentinel threshold is a problem. ``partial``/``unverified`` alignment
        (some/all configs share no keys) is a caveat, not a failure, so it does
        not trip the exit code.
        """
        return (
            self.alignment.status == "misaligned"
            or self.row_count_mismatch
            or bool(self.flagged_models)
        )


def audit_repo(
    repo_id: str,
    split: str = "train",
    max_ocr_text_len: int = MAX_OCR_TEXT_LENGTH,
    api: Any | None = None,
) -> AuditReport:
    """Run config discovery and compute a health report — no judging, no writes.

    Mirrors the judge's discovery cascade (open PRs + main-branch configs, then
    flat fallback) so the audit sees exactly what a judge run would.
    """
    pr_configs, pr_revisions = discover_pr_configs(repo_id, api=api)
    main_configs = discover_configs(repo_id)
    config_names = list(pr_configs)
    for mc in main_configs:
        if mc not in pr_configs:
            config_names.append(mc)

    if config_names:
        loaded = _load_configs(repo_id, config_names, split, pr_revisions)
        usable = [lc for lc in loaded if lc.text_col is not None]
        alignment = check_config_alignment(usable)
        configs = []
        for lc in usable:
            text_col = lc.text_col
            if text_col is None:  # filtered into `usable` above; narrows the type
                continue
            stats = compute_column_stats(
                lc.config, lc.model_id, _column_values(lc.ds, text_col), max_ocr_text_len
            )
            configs.append(ConfigAudit(stats))
    else:
        ds, ocr_columns = load_flat_dataset(repo_id, split=split)
        alignment = AlignmentResult(status="n/a")
        configs = [
            ConfigAudit(stats)
            for stats in compute_model_stats(ds, ocr_columns, max_ocr_text_len)
        ]

    return AuditReport(
        repo_id=repo_id,
        configs=configs,
        alignment=alignment,
        max_ocr_text_len=max_ocr_text_len,
    )
