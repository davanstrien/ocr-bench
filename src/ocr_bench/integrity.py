"""Input-integrity checks shared by the judge-time guards and the ``audit`` command.

These are the checks that caught two live boards' problems on 2026-07-08: a model
whose whole column was error sentinels (issue #46) and truncation exposure on a
long-text corpus. Both the judge and the read-only ``audit`` command call the same
functions here so the pre-flight report and the guard can never disagree.
"""

from __future__ import annotations

import statistics
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from ocr_bench import __version__
from ocr_bench.dataset import (
    AlignmentResult,
    DatasetError,
    _load_configs,
    check_config_alignment,
    discover_configs,
    discover_pr_configs,
    load_flat_dataset,
    require_dataset_repo,
)
from ocr_bench.elo import ComparisonResult
from ocr_bench.judge import MAX_OCR_TEXT_LENGTH, is_sentinel, normalize_for_judge, ocr_text
from ocr_bench.screen import MIN_MODELS, ScreenReport, screen_outputs

# A model with more than this fraction of error sentinels is flagged: the run
# effectively failed on this corpus, so its outputs are not comparable.
SENTINEL_FLAG_RATE = 0.10

# judge and bench warn when --max-ocr-text-len truncates more than this share of
# the outputs in the judged sample, overall or for any one model.
TRUNCATION_WARN_RATE = 0.5

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
    normalize: bool = True,
) -> ColumnStats:
    """Compute :class:`ColumnStats` for one column's OCR values.

    Rows are partitioned into empty / sentinel / short / normal (non-overlapping).
    A value that is not a string counts as empty, as in the judge (see
    :func:`ocr_bench.judge.ocr_text`). Length statistics use the same
    normalized-or-raw representation that the judge caps, so audit truncation
    exposure matches the selected text mode.
    """
    n_empty = n_sentinel = n_short = 0
    lengths: list[int] = []
    for t in texts:
        s = ocr_text(t)
        measured = normalize_for_judge(s) if normalize else s
        lengths.append(len(measured))
        stripped = measured.strip()
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
    normalize: bool = True,
) -> list[ColumnStats]:
    """Per-model health stats over a loaded (merged or flat) dataset.

    ``ocr_columns`` maps column/config name → model label. Used by the judge to
    derive per-model ``failed_outputs`` and the >10% sentinel-rate warning.
    """
    return [
        compute_column_stats(
            col,
            model,
            _column_values(dataset, col),
            max_ocr_text_len,
            normalize=normalize,
        )
        for col, model in ocr_columns.items()
    ]


def failed_output_counts(stats: list[ColumnStats]) -> dict[str, int]:
    """Map model → sentinel count, keeping only models with at least one."""
    return {s.model: s.n_sentinel for s in stats if s.n_sentinel}


@dataclass
class ModelCoverage:
    """How many pages of the judged sample one model's rating rests on.

    - ``sampled_pages`` (M): the pages in the judged sample, that is
      ``--max-samples`` pages or every row.
    - ``ranked_pages`` (N): the sampled pages where the model took part in at
      least one comparison on the board, judged or auto-tied. Its rating uses
      only these pages.
    - ``sentinel_pages``: the sampled pages where the model's output is an
      error sentinel. Those pages are never judged for the model.

    N can also be below M minus ``sentinel_pages`` for every model. That
    happens on a page where all outputs are shorter than ``--min-chars``, or
    when adaptive stopping or ``--max-comparisons`` ends the run first.
    """

    model: str
    sampled_pages: int
    ranked_pages: int
    sentinel_pages: int

    def to_dict(self) -> dict[str, int]:
        return {
            "sampled_pages": self.sampled_pages,
            "ranked_pages": self.ranked_pages,
            "sentinel_pages": self.sentinel_pages,
        }


def model_coverage(
    dataset: Any,
    ocr_columns: dict[str, str],
    indices: list[int],
    results: Iterable[ComparisonResult],
) -> dict[str, ModelCoverage]:
    """Return the coverage of every model in ``ocr_columns`` over the judged sample.

    ``indices`` are the sampled rows and ``results`` are the comparisons on the
    board (existing and new). This only counts pages; it does not change which
    comparisons are made or how they are rated.
    """
    sample = set(indices)
    ranked: dict[str, set[int]] = {model: set() for model in ocr_columns.values()}
    for result in results:
        if result.sample_idx not in sample:
            continue
        for model in (result.model_a, result.model_b):
            if model in ranked:
                ranked[model].add(result.sample_idx)
    coverage: dict[str, ModelCoverage] = {}
    for col, model in ocr_columns.items():
        values = _column_values(dataset, col)
        sentinels = sum(1 for idx in sample if is_sentinel(values[idx]))
        coverage[model] = ModelCoverage(
            model=model,
            sampled_pages=len(sample),
            ranked_pages=len(ranked[model]),
            sentinel_pages=sentinels,
        )
    return coverage


@dataclass
class TruncationExposure:
    """How many outputs in the judged sample the judge's text cap cuts short.

    Error sentinels are left out of both counts: the judge never sees them.
    """

    max_ocr_text_len: int
    outputs: dict[str, int]  # model → outputs that are not error sentinels
    over_cap: dict[str, int]  # model → of those, outputs longer than the cap

    @property
    def total_outputs(self) -> int:
        return sum(self.outputs.values())

    @property
    def total_over_cap(self) -> int:
        return sum(self.over_cap.values())

    @property
    def overall_rate(self) -> float:
        return self.total_over_cap / self.total_outputs if self.total_outputs else 0.0

    def rate(self, model: str) -> float:
        outputs = self.outputs.get(model, 0)
        return self.over_cap.get(model, 0) / outputs if outputs else 0.0

    @property
    def models_over_warn_rate(self) -> list[str]:
        """Models with more than TRUNCATION_WARN_RATE of their outputs truncated."""
        return [model for model in self.outputs if self.rate(model) > TRUNCATION_WARN_RATE]

    @property
    def warn(self) -> bool:
        """True if the cap truncates over half the outputs, overall or for one model."""
        return self.overall_rate > TRUNCATION_WARN_RATE or bool(self.models_over_warn_rate)


def _sampled_values(dataset: Any, col: str, indices: list[int]) -> list:
    values = _column_values(dataset, col)
    return [values[idx] for idx in indices]


def truncation_exposure(
    dataset: Any,
    ocr_columns: dict[str, str],
    indices: list[int],
    max_ocr_text_len: int = MAX_OCR_TEXT_LENGTH,
    normalize: bool = True,
) -> TruncationExposure:
    """Count the sampled outputs that ``max_ocr_text_len`` truncates, per model.

    Lengths are measured as the judge caps them (after HTML normalisation when
    ``normalize`` is true), with :func:`compute_column_stats`. Error sentinels
    are removed first, so a sentinel padded with whitespace is never counted
    as a truncated output.
    """
    outputs: dict[str, int] = {}
    over_cap: dict[str, int] = {}
    for col, model in ocr_columns.items():
        values = _sampled_values(dataset, col, indices)
        judged = [value for value in values if not is_sentinel(value)]
        stats = compute_column_stats(col, model, judged, max_ocr_text_len, normalize=normalize)
        outputs[model] = stats.n_rows
        over_cap[model] = stats.n_over_max
    return TruncationExposure(max_ocr_text_len, outputs, over_cap)


def suggest_text_cap(
    dataset: Any,
    ocr_columns: dict[str, str],
    indices: list[int],
    max_ocr_text_len: int = MAX_OCR_TEXT_LENGTH,
    normalize: bool = True,
) -> ScreenReport | None:
    """Run the ``audit --screen`` checks on the judged sample, for its text-cap suggestion.

    Returns None when the screen cannot run: it needs MIN_MODELS or more
    models whose sampled outputs are not all empty or error sentinels.
    The suggestion is ``ScreenReport.suggested_max_ocr_text_len``.
    """
    texts: dict[str, list] = {}
    models: dict[str, str] = {}
    for col, model in ocr_columns.items():
        values = _sampled_values(dataset, col, indices)
        stats = compute_column_stats(col, model, values, max_ocr_text_len, normalize=normalize)
        if stats.n_empty + stats.n_sentinel == stats.n_rows:
            continue  # no output to compare, as in audit_repo
        texts[col] = values
        models[col] = model
    if len(texts) < MIN_MODELS:
        return None
    return screen_outputs(texts, models, max_ocr_text_len, normalize=normalize)


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
    judge_text_mode: str = "normalized"
    configs_without_text: list[str] = field(default_factory=list)
    auxiliary_configs: list[str] = field(default_factory=list)
    # Open PRs that discovery did not use: {config: [PR numbers]}.
    duplicate_prs: dict[str, list[int]] = field(default_factory=dict)
    superseded_prs: dict[str, list[int]] = field(default_factory=dict)
    pr_revisions: dict[str, str] = field(default_factory=dict)
    # Set by audit_repo(screen=True): the screen results, or why it did not run.
    screen: ScreenReport | None = None
    screen_skipped: str | None = None

    @property
    def unjudgeable_configs(self) -> list[str]:
        """Discovered configs that cannot contribute a single judged page."""
        no_usable_rows = [
            c.stats.name for c in self.configs if c.stats.n_rows == 0 or c.stats.empty_rate == 1.0
        ]
        return self.configs_without_text + no_usable_rows

    @property
    def too_few_models(self) -> bool:
        """True if fewer than two configs can be judged, so no pair exists."""
        unjudgeable = set(self.unjudgeable_configs)
        return sum(c.stats.name not in unjudgeable for c in self.configs) < 2

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

        A real misalignment, a row-count mismatch, any config over the sentinel
        threshold, any config with no judgeable rows, or fewer than two judgeable
        configs is a problem. ``partial``/``unverified`` alignment
        (some/all configs share no keys) is a caveat, not a failure, so it does
        not trip the exit code.
        """
        return (
            self.alignment.status == "misaligned"
            or self.row_count_mismatch
            or bool(self.flagged_models)
            or bool(self.unjudgeable_configs)
            or self.too_few_models
        )

    @property
    def problems(self) -> list[str]:
        """One line per blocking problem; empty when ``has_problems`` is False."""
        problems: list[str] = []
        if self.alignment.status == "misaligned":
            problems.append("row misalignment")
        if self.row_count_mismatch:
            counts = ", ".join(f"{c.stats.name}={c.stats.n_rows}" for c in self.configs)
            problems.append(f"row-count mismatch ({counts})")
        if self.flagged_models:
            flagged = ", ".join(self.flagged_models)
            problems.append(f">{SENTINEL_FLAG_RATE:.0%} sentinels: {flagged}")
        if self.unjudgeable_configs:
            problems.append(f"no judgeable rows: {', '.join(self.unjudgeable_configs)}")
        if self.too_few_models:
            problems.append("fewer than two judgeable configs")
        return problems

    def to_dict(self) -> dict:
        """The report as plain data, for ``audit --json``."""
        configs = []
        for cfg in self.configs:
            s = cfg.stats
            configs.append(
                {
                    "name": s.name,
                    "model": s.model,
                    "source": self.pr_revisions.get(s.name, "main"),
                    "rows": s.n_rows,
                    "empty": s.n_empty,
                    "short": s.n_short,
                    "sentinel": s.n_sentinel,
                    "over_max": s.n_over_max,
                    "median_len": s.median_len,
                    "max_len": s.max_len,
                    "alignment": self.alignment.config_status(s.name),
                }
            )
        return {
            "repo_id": self.repo_id,
            "ocr_bench_version": __version__,
            "ok": not self.has_problems,
            "problems": self.problems,
            "judge_text_mode": self.judge_text_mode,
            "max_ocr_text_len": self.max_ocr_text_len,
            "alignment": {"status": self.alignment.status, "summary": self.alignment.describe()},
            "configs": configs,
            "configs_without_text": self.configs_without_text,
            "auxiliary_configs": self.auxiliary_configs,
            "duplicate_prs": self.duplicate_prs,
            "superseded_prs": self.superseded_prs,
            "screen": self.screen.to_dict() if self.screen is not None else None,
            "screen_skipped": self.screen_skipped,
        }


def _screen_skip_reason(
    alignment: AlignmentResult, row_counts: set[int], n_models: int
) -> str | None:
    """Why the screen cannot run on these configs, or None if it can."""
    if alignment.status == "misaligned":
        return "the configs' rows are misaligned"
    if len(row_counts) > 1:
        return "the configs have different row counts"
    if n_models < MIN_MODELS:
        return (
            f"the screen needs at least {MIN_MODELS} models with outputs that are not "
            "all empty or error placeholders"
        )
    return None


def _looks_like_ocr_output(loaded_config: Any) -> bool:
    """True if a config has the page images or run metadata of an OCR output."""
    columns = set(loaded_config.ds.column_names)
    return bool(columns & {"image", "inference_info"})


def audit_repo(
    repo_id: str,
    split: str = "train",
    max_ocr_text_len: int = MAX_OCR_TEXT_LENGTH,
    judge_text_mode: str = "normalized",
    api: Any | None = None,
    configs: list[str] | None = None,
    from_prs: bool = False,
    screen: bool = False,
    columns: list[str] | None = None,
) -> AuditReport:
    """Run config discovery and compute a health report — no judging, no writes.

    Discovery follows ``ocr-bench judge`` with the same flags, so the audit sees
    exactly what a judge run would:

    - ``configs`` and ``from_prs``: the named configs, each from its newest open
      PR, or from main when it has no PR or main changed after the PR;
    - ``configs`` only: the named configs from main;
    - ``columns`` (and no ``configs``): these text columns of a flat dataset,
      one column per model, named after its model;
    - ``from_prs`` only: every config with an open PR;
    - none of these: open PRs plus main-branch configs, then flat columns.

    With ``screen``, also run the screen checks (see :mod:`ocr_bench.screen`)
    on the configs with judgeable rows.
    """
    if judge_text_mode not in {"normalized", "raw"}:
        raise ValueError("judge_text_mode must be 'normalized' or 'raw'")
    normalize = judge_text_mode == "normalized"
    require_dataset_repo(repo_id, api=api)

    duplicate_prs: dict[str, list[int]] = {}
    superseded_prs: dict[str, list[int]] = {}
    pr_revisions: dict[str, str] = {}
    if configs and from_prs:
        config_names = list(configs)
        _, pr_revisions = discover_pr_configs(
            repo_id,
            api=api,
            only=config_names,
            duplicates=duplicate_prs,
            main_configs=discover_configs(repo_id),
            superseded=superseded_prs,
        )
    elif configs:
        config_names = list(configs)
    elif columns:
        config_names = []  # a flat dataset: read the named columns below
    elif from_prs:
        pr_configs, pr_revisions = discover_pr_configs(
            repo_id,
            api=api,
            duplicates=duplicate_prs,
            main_configs=discover_configs(repo_id),
            superseded=superseded_prs,
        )
        config_names = pr_configs + list(superseded_prs)
        if not config_names:
            raise DatasetError("No configs found in open PRs")
    else:
        main_configs = discover_configs(repo_id)
        pr_configs, pr_revisions = discover_pr_configs(
            repo_id,
            api=api,
            duplicates=duplicate_prs,
            main_configs=main_configs,
            superseded=superseded_prs,
        )
        config_names = list(pr_configs)
        for mc in main_configs:
            if mc not in pr_configs:
                config_names.append(mc)

    # name -> (model label, outputs), for the screen.
    screen_inputs: dict[str, tuple[str, list]] = {}
    if config_names:
        loaded = _load_configs(repo_id, config_names, split, pr_revisions)
        usable = [lc for lc in loaded if lc.text_col is not None]
        # A config with no text is only a failed OCR run if it looks like OCR
        # output. Repositories can also hold auxiliary configs such as metadata.
        no_text = [lc for lc in loaded if lc.text_col is None]
        configs_without_text = [lc.config for lc in no_text if _looks_like_ocr_output(lc)]
        auxiliary_configs = [lc.config for lc in no_text if not _looks_like_ocr_output(lc)]
        alignment = check_config_alignment(usable)
        audits: list[ConfigAudit] = []
        for lc in usable:
            text_col = lc.text_col
            if text_col is None:  # filtered into `usable` above; narrows the type
                continue
            values = _column_values(lc.ds, text_col)
            screen_inputs[lc.config] = (lc.model_id, values)
            stats = compute_column_stats(
                lc.config,
                lc.model_id,
                values,
                max_ocr_text_len,
                normalize=normalize,
            )
            audits.append(ConfigAudit(stats))
    else:
        configs_without_text = []
        auxiliary_configs = []
        ds, ocr_columns = load_flat_dataset(repo_id, split=split, columns=columns)
        alignment = AlignmentResult(status="n/a")
        audits = []
        for col, model in ocr_columns.items():  # same order as compute_model_stats
            values = _column_values(ds, col)
            screen_inputs[col] = (model, values)
            audits.append(
                ConfigAudit(
                    compute_column_stats(col, model, values, max_ocr_text_len, normalize=normalize)
                )
            )

    report = AuditReport(
        repo_id=repo_id,
        configs=audits,
        alignment=alignment,
        max_ocr_text_len=max_ocr_text_len,
        judge_text_mode=judge_text_mode,
        configs_without_text=configs_without_text,
        auxiliary_configs=auxiliary_configs,
        duplicate_prs=duplicate_prs,
        superseded_prs=superseded_prs,
        pr_revisions=pr_revisions,
    )
    if screen:
        # Leave out configs whose every row is empty or an error sentinel: they
        # have no output to compare, and they must not count towards MIN_MODELS.
        without_output = {
            c.stats.name
            for c in report.configs
            if c.stats.n_empty + c.stats.n_sentinel == c.stats.n_rows
        }
        screened = {name: col for name, col in screen_inputs.items() if name not in without_output}
        reason = _screen_skip_reason(
            alignment, {len(values) for _, values in screened.values()}, len(screened)
        )
        if reason is None:
            report.screen = screen_outputs(
                {name: values for name, (_, values) in screened.items()},
                {name: model for name, (model, _) in screened.items()},
                max_ocr_text_len,
                normalize=normalize,
            )
        else:
            report.screen_skipped = reason
    return report
