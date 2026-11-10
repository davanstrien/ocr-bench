"""CLI entrypoint for ocr-bench."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn

import structlog
from datasets.exceptions import DatasetsError
from openai import OpenAIError
from rich.console import Console
from rich.table import Table

from ocr_bench import __version__
from ocr_bench.adaptive import (
    AdjacentPairDecision,
    classify_adjacent_pairs,
    comparison_pair_counts,
    model_parameter_counts,
    practical_preferences,
    unresolved_pairs,
)
from ocr_bench.backends import (
    DEFAULT_JUDGE,
    DEFAULT_MAX_TOKENS,
    aggregate_jury_votes,
    parse_judge_spec,
)
from ocr_bench.dataset import (
    DatasetError,
    discover_configs,
    discover_pr_configs,
    evaluation_input_fingerprints,
    load_config_dataset,
    load_flat_dataset,
    require_dataset_repo,
)
from ocr_bench.elo import ComparisonResult, Leaderboard, ShownFirst, compute_elo
from ocr_bench.integrity import (
    SENTINEL_FLAG_RATE,
    ModelCoverage,
    audit_repo,
    compute_model_stats,
    failed_output_counts,
    model_coverage,
    suggest_text_cap,
    truncation_exposure,
)
from ocr_bench.judge import (
    CRITERIA_PROFILES,
    DEFAULT_CRITERIA,
    DEFAULT_MIN_CHARS,
    MAX_IMAGE_DIM,
    MAX_OCR_TEXT_LENGTH,
    Comparison,
    _normalize_pair,
    build_comparisons,
    judging_order,
    prompt_hash,
    sample_indices,
    validate_prompt_template,
)
from ocr_bench.publish import (
    EvalMetadata,
    ensure_dataset_repo,
    evaluation_provenance_hash,
    hash_judge_specs,
    load_existing_comparisons,
    load_existing_metadata,
    publish_checkpoint,
    publish_results,
)
from ocr_bench.screen import ScreenReport

if TYPE_CHECKING:
    from ocr_bench.run import JobRun

logger = structlog.get_logger()
console = Console()


def _positive_int(value: str) -> int:
    """argparse type: integer >= 1 (rejects 0 and negatives)."""
    ivalue = int(value)
    if ivalue < 1:
        raise argparse.ArgumentTypeError(f"must be a positive integer (>= 1), got {ivalue}")
    return ivalue


def _non_negative_int(value: str) -> int:
    """argparse type: integer >= 0 (rejects negatives)."""
    ivalue = int(value)
    if ivalue < 0:
        raise argparse.ArgumentTypeError(f"must be >= 0, got {ivalue}")
    return ivalue


def _float_greater_than_one(value: str) -> float:
    """argparse type: float > 1 for meaningful size ratios."""
    fvalue = float(value)
    if not math.isfinite(fvalue) or fvalue <= 1:
        raise argparse.ArgumentTypeError(f"must be greater than 1, got {fvalue}")
    return fvalue


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ocr-bench",
        description="OCR model evaluation toolkit — VLM-as-judge with per-dataset leaderboards",
    )
    sub = parser.add_subparsers(dest="command")

    judge = sub.add_parser("judge", help="Run pairwise VLM judge on OCR outputs")

    # Dataset
    judge.add_argument("dataset", help="HF dataset repo id")
    judge.add_argument("--split", default="train", help="Dataset split (default: train)")
    judge.add_argument(
        "--columns",
        nargs="+",
        default=None,
        help=(
            "Judge these text columns of a flat dataset (one column per model, named "
            "after the model). Ignored with --configs; --from-prs is ignored with --columns."
        ),
    )
    judge.add_argument(
        "--configs",
        nargs="+",
        default=None,
        help=(
            "Config-per-model: list of config names, loaded from main. With "
            "--from-prs, each is loaded from its newest open PR, or from main if "
            "it has no open PR or it changed on main after its newest open PR."
        ),
    )
    judge.add_argument(
        "--from-prs",
        action="store_true",
        help=(
            "Force PR-based config discovery (newest open PR per config). A config "
            "that changed on main after its newest open PR is loaded from main. "
            "Combine with --configs to judge only some of them."
        ),
    )
    judge.add_argument(
        "--merge",
        action="store_true",
        help="Merge PRs to main after discovery (default: load via revision)",
    )

    # Judge
    judge.add_argument(
        "--model",
        action="append",
        dest="models",
        help=f"Judge model spec (repeatable for jury). Default: {DEFAULT_JUDGE}",
    )
    # --criteria and --criteria-file are mutually exclusive, enforced in
    # _resolve_criteria (NOT an argparse mutually-exclusive group: on Python
    # 3.13.0 such a group with a defaulted choices arg in a subparser fails to
    # detect the conflict — a real regression, fixed in 3.13.1). --criteria's
    # default is the None sentinel so "explicitly passed" is distinguishable
    # from "unset"; it resolves to DEFAULT_CRITERIA in _resolve_criteria.
    judge.add_argument(
        "--criteria",
        choices=sorted(CRITERIA_PROFILES),
        default=None,
        help=(
            "Judge criteria profile (default: default). 'table-fidelity' adds an "
            "explicit row/column cell-preservation criterion for table-dense corpora. "
            "'body-text' counts only the main text, headings, footnotes and captions "
            "towards completeness; running headers, page numbers, signature marks and "
            "catchwords are optional. Mutually exclusive with --criteria-file."
        ),
    )
    judge.add_argument(
        "--criteria-file",
        default=None,
        metavar="PATH",
        help=(
            "Path to a custom judge prompt template (mutually exclusive with "
            "--criteria). Must contain the {ocr_text_a} and {ocr_text_b} "
            "placeholders and no other format fields — see the 'default' profile "
            "for the expected shape. Recorded in metadata as custom:<filename>."
        ),
    )

    # Eval
    judge.add_argument("--max-samples", type=int, default=None, help="Max samples to evaluate")
    judge.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    judge.add_argument(
        "--max-comparisons",
        type=_positive_int,
        default=None,
        help=(
            "Hard cap on total comparisons judged this run (default: no cap; "
            "must be >= 1). Adaptive stopping bounds per-pair sampling but not "
            "the total; this makes cost and wall-clock deterministic. On reaching "
            "it, the run stops cleanly and publishes what it has."
        ),
    )
    judge.add_argument(
        "--checkpoint-every",
        type=_non_negative_int,
        default=None,
        help=(
            "Push accumulated comparisons to the results repo every N comparisons "
            "(0 = off; default: 500, or off under --full-rejudge). Cheap "
            "append-only pushes (comparisons config only, no leaderboard/README "
            "churn) so an interrupted run resumes without re-judging. Evaluated at "
            "batch boundaries, so N is a lower bound on the comparisons between "
            "checkpoints."
        ),
    )
    judge.add_argument(
        "--min-chars",
        type=int,
        default=DEFAULT_MIN_CHARS,
        help=(
            "Skip a pair when both OCR outputs are shorter than this "
            f"(default: {DEFAULT_MIN_CHARS}). Set 0 to disable."
        ),
    )
    judge.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"Max tokens for judge response (default: {DEFAULT_MAX_TOKENS})",
    )
    judge.add_argument(
        "--max-ocr-text-len",
        type=_positive_int,
        default=MAX_OCR_TEXT_LENGTH,
        help=(
            "Per-output character cap (applied after HTML normalization) for "
            f"each OCR text in the prompt (default: {MAX_OCR_TEXT_LENGTH}). "
            "Table-dense corpora may need a larger value."
        ),
    )
    judge.add_argument(
        "--judge-image-dim",
        type=_positive_int,
        default=MAX_IMAGE_DIM,
        help=(
            "Longer-side pixel cap for the judge image "
            f"(default: {MAX_IMAGE_DIM}). Dense broadsheets may need 1536-2048."
        ),
    )
    judge.add_argument(
        "--judge-text-mode",
        choices=["normalized", "raw"],
        default="normalized",
        help=(
            "How OCR text is prepared for the judge (default: normalized). "
            "'normalized' flattens HTML to bare content before the char cap so "
            "the cap is format-neutral (the default, because raw mode lets "
            "HTML-verbose outputs burn the cap on markup — the #45 bias). "
            "'raw' skips normalization and caps text as-is — use it to judge "
            "markup quality itself or to audit the harness's transformation."
        ),
    )

    # Output
    judge.add_argument(
        "--save-results",
        default=None,
        help="HF repo id to publish results to (default: {dataset}-results)",
    )
    judge.add_argument(
        "--no-publish",
        action="store_true",
        help="Don't publish results (default: publish to {dataset}-results)",
    )
    judge.add_argument(
        "--private",
        action="store_true",
        help=(
            "Require a private results repo: create it private, and stop before "
            "judging if it already exists as a public repo. This is automatic when "
            "the source dataset is private or gated."
        ),
    )
    judge.add_argument(
        "--public",
        action="store_true",
        help=(
            "Allow a public results repo even though the source dataset is private "
            "or gated. Without it, judging stops before writing to a public repo."
        ),
    )
    judge.add_argument(
        "--license",
        default=None,
        help=(
            "License tag for the published results dataset card, e.g. cc0-1.0 "
            "(default: none — the results embed source-derived text, so only "
            "the publisher knows the right license)"
        ),
    )
    judge.add_argument(
        "--full-rejudge",
        action="store_true",
        help="Re-judge all pairs, ignoring existing comparisons in --save-results repo",
    )
    judge.add_argument(
        "--no-adaptive",
        action="store_true",
        help="Disable adaptive stopping (default: adaptive is on)",
    )
    judge.add_argument(
        "--adaptive-strategy",
        choices=["balanced", "targeted"],
        default="balanced",
        help=(
            "Adaptive allocation strategy (default: balanced). 'balanced' keeps "
            "judging every model pair in each page batch; 'targeted' is opt-in and, "
            "after an initial balanced evidence threshold, tops up only adjacent pairs "
            "whose 95%% CIs remain unresolved."
        ),
    )
    judge.add_argument(
        "--size-tie-ratio",
        type=_float_greater_than_one,
        default=None,
        metavar="RATIO",
        help=(
            "Treat an unresolved adjacent pair as a practical smaller-model "
            "preference when its parameter-count ratio is at least RATIO. Applies "
            "to balanced and targeted adaptive stopping; targeted also stops topping "
            "up that pair. Does not alter ELO or claim equivalence."
        ),
    )
    judge.add_argument(
        "--size-tie-min-samples",
        type=_positive_int,
        default=10,
        metavar="N",
        help=(
            "Minimum direct comparisons before --size-tie-ratio may stop sampling "
            "a pair (default: 10)."
        ),
    )
    judge.add_argument(
        "--concurrency",
        type=_positive_int,
        default=8,
        help=(
            "Number of concurrent judge API calls (default: 8). Lower it if the "
            "provider returns 429 errors; 1 makes sequential calls."
        ),
    )

    # --- run subcommand ---
    run = sub.add_parser("run", help="Launch OCR models on a dataset via HF Jobs")
    run.add_argument(
        "input_dataset",
        nargs="?",
        help="HF dataset repo id with images (not needed with --list-models)",
    )
    run.add_argument(
        "output_repo",
        nargs="?",
        help="Output dataset repo (not needed with --list-models)",
    )
    run.add_argument(
        "--models", nargs="+", default=None, help="Model slugs to run (default: core set)"
    )
    run.add_argument("--max-samples", type=int, default=None, help="Per-model sample limit")
    run.add_argument("--split", default="train", help="Dataset split (default: train)")
    run.add_argument("--flavor", default=None, help="Override GPU flavor for all models")
    run.add_argument("--timeout", default="4h", help="Per-job timeout (default: 4h)")
    run.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    run.add_argument("--shuffle", action="store_true", help="Shuffle source dataset")
    run.add_argument("--list-models", action="store_true", help="Print available models and exit")
    run.add_argument(
        "--dry-run", action="store_true", help="Show what would launch without launching"
    )
    run.add_argument(
        "--private",
        action="store_true",
        help=(
            "Require a private output repo: create it private, and stop before "
            "launching if it already exists as a public repo. This is automatic when "
            "the input dataset is private or gated."
        ),
    )
    run.add_argument(
        "--public",
        action="store_true",
        help=(
            "Allow a public output repo even though the input dataset is private or "
            "gated. Without it, the run stops before writing to a public repo."
        ),
    )
    run.add_argument(
        "--no-wait", action="store_true", help="Launch and exit without polling (default: wait)"
    )

    # --- view subcommand ---
    view = sub.add_parser("view", help="Browse and validate results in a web UI")
    view.add_argument("results", help="HF dataset repo id with published results")
    view.add_argument("--port", type=int, default=7860, help="Port (default: 7860)")
    view.add_argument("--host", default="127.0.0.1", help="Host (default: 127.0.0.1)")
    view.add_argument("--output", default=None, help="Path to save annotations JSON")

    # --- publish subcommand ---
    publish = sub.add_parser("publish", help="Deploy results viewer as a Hugging Face Space")
    publish.add_argument("results", help="HF results dataset repo id to view in the Space")
    publish.add_argument("--space", default=None, help="Space repo id (default: {results}-viewer)")
    publish.add_argument("--private", action="store_true", help="Make the Space private")

    # --- share subcommand ---
    share = sub.add_parser(
        "share",
        help="Write a scores-only board from a results repo to a separate repo",
        description=(
            "Read a results repo written by judge (it may be private) and write "
            "ratings only to a separate dataset repo: no page text, OCR outputs or "
            "judge reasons. A missing destination is created private (--public to "
            "create it public). An existing one keeps its visibility."
        ),
    )
    share.add_argument("results", help="Results repo written by `ocr-bench judge`")
    share.add_argument(
        "dest", help="Dataset repo for the board; not the results repo or the source"
    )
    share.add_argument(
        "--alias",
        required=True,
        help="Public name of the evaluation set. The source repo id is not published.",
    )
    share.add_argument(
        "--slice-column",
        default=None,
        help=(
            "Source dataset column that labels each page (for example a document "
            "type). Adds one board per label with 10 or more judged pages. The "
            "column name and its labels are published as they are. Downloads the "
            "judged source config, images included, to check the pages."
        ),
    )
    share.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the files to a local folder (--output-dir) and push nothing",
    )
    share.add_argument(
        "--output-dir",
        default=None,
        help="Folder for --dry-run only (default: ./ocr-bench-share-<alias>)",
    )
    share.add_argument(
        "--public",
        action="store_true",
        help=(
            "Create a missing destination repo public (default: private, so you can "
            "look at the board on the Hub before you publish it)"
        ),
    )
    share.add_argument(
        "--replace",
        action="store_true",
        help="Replace a board of another alias in the destination repo",
    )

    # --- bench subcommand (run → judge → view in one shot) ---
    bench = sub.add_parser(
        "bench", help="Run → judge → view in one command (the 'just try it' path)"
    )
    bench.add_argument("input_dataset", help="HF dataset repo id with images")
    bench.add_argument("output_repo", help="Output dataset repo (OCR outputs + {repo}-results)")
    bench.add_argument(
        "--models", nargs="+", default=None, help="OCR model slugs to run (default: all core)"
    )
    bench.add_argument(
        "--judge-model",
        action="append",
        dest="judge_models",
        help=f"Judge model spec (repeatable for a jury). Default: {DEFAULT_JUDGE}",
    )
    # Mutual exclusion enforced in _resolve_criteria (see the judge parser note).
    bench.add_argument(
        "--criteria",
        choices=sorted(CRITERIA_PROFILES),
        default=None,
        help=(
            "Judge criteria profile (default: default). 'table-fidelity' adds an "
            "explicit row/column cell-preservation criterion for table-dense corpora. "
            "'body-text' counts only the main text, headings, footnotes and captions "
            "towards completeness; running headers, page numbers, signature marks and "
            "catchwords are optional. Mutually exclusive with --criteria-file."
        ),
    )
    bench.add_argument(
        "--criteria-file",
        default=None,
        metavar="PATH",
        help="Path to a custom judge prompt template (mutually exclusive with --criteria).",
    )
    bench.add_argument(
        "--max-samples", type=int, default=None, help="Per-model sample limit (also caps judging)"
    )
    bench.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    bench.add_argument(
        "--judge-text-mode",
        choices=["normalized", "raw"],
        default="normalized",
        help="OCR text prep for the judge (default: normalized; see `judge --help`).",
    )
    bench.add_argument(
        "--adaptive-strategy",
        choices=["balanced", "targeted"],
        default="balanced",
        help="Judge allocation strategy (default: balanced; see `judge --help`).",
    )
    bench.add_argument(
        "--size-tie-ratio",
        type=_float_greater_than_one,
        default=None,
        metavar="RATIO",
        help="Optional practical smaller-model stopping ratio (see `judge --help`).",
    )
    bench.add_argument(
        "--size-tie-min-samples",
        type=_positive_int,
        default=10,
        metavar="N",
        help="Minimum direct pair comparisons for the size rule (default: 10).",
    )
    bench.add_argument(
        "--concurrency",
        type=_positive_int,
        default=8,
        help=(
            "Number of concurrent judge API calls (default: 8). Lower it if the "
            "provider returns 429 errors; 1 makes sequential calls."
        ),
    )
    bench.add_argument(
        "--no-publish", action="store_true", help="Don't publish results (skips the viewer)"
    )
    bench.add_argument(
        "--private",
        action="store_true",
        help=(
            "Require private output and results repos; stop before launching if "
            "either already exists as a public repo. Automatic for a private or "
            "gated input dataset."
        ),
    )
    bench.add_argument(
        "--public",
        action="store_true",
        help="Allow public output and results repos even for a private or gated input dataset.",
    )
    bench.add_argument("--port", type=int, default=7860, help="Viewer port (default: 7860)")
    bench.add_argument("--host", default="127.0.0.1", help="Viewer host (default: 127.0.0.1)")

    # --- audit subcommand ---
    audit = sub.add_parser(
        "audit",
        help="Read-only pre-judge health check on an OCR output repo",
    )
    audit.add_argument("dataset", help="HF dataset repo id with OCR outputs")
    audit.add_argument("--split", default="train", help="Dataset split (default: train)")
    audit.add_argument(
        "--judge-text-mode",
        choices=["normalized", "raw"],
        default="normalized",
        help="Measure judge input after HTML normalization (default) or as raw OCR text.",
    )
    audit.add_argument(
        "--max-ocr-text-len",
        type=_positive_int,
        default=MAX_OCR_TEXT_LENGTH,
        help=(
            "Text length above which the judge truncates an output "
            f"(default: {MAX_OCR_TEXT_LENGTH}); reported as truncation exposure"
        ),
    )
    audit.add_argument(
        "--configs",
        nargs="+",
        default=None,
        help=(
            "Audit only these configs, loaded from main. With --from-prs, each is "
            "loaded from its newest open PR, or from main if it has no open PR or it "
            "changed on main after its newest open PR, as in 'judge'."
        ),
    )
    audit.add_argument(
        "--columns",
        nargs="+",
        default=None,
        help=(
            "Audit these text columns of a flat dataset (one column per model, named "
            "after the model), as in 'judge --columns'. Ignored with --configs; "
            "--from-prs is ignored with --columns."
        ),
    )
    audit.add_argument(
        "--from-prs",
        action="store_true",
        help="Audit the configs in open PRs (newest open PR per config), as in 'judge'.",
    )
    audit.add_argument(
        "--screen",
        action="store_true",
        help=(
            "Also run free checks that compare each model's output with the other "
            "models' outputs on the same page: loops, outputs over 5x the others' "
            "length, text on pages that most models leave blank, empty output on "
            "pages where most models have text, and peer agreement. Needs 3 or more "
            "models. Use it to find weak models before a paid judge run."
        ),
    )
    audit.add_argument(
        "--json",
        action="store_true",
        help=(
            "Print the report as JSON on stdout, with the flagged rows of each "
            "screen check. The exit codes do not change."
        ),
    )

    return parser


def _decisions_for_board(
    board: Leaderboard,
    comparisons: Iterable[object] | None,
    *,
    size_tie_ratio: float | None,
    size_tie_min_samples: int,
) -> list[AdjacentPairDecision]:
    """Classify the board's adjacent pairs under the selected stopping rule."""
    return classify_adjacent_pairs(
        board,
        comparison_pair_counts(comparisons if comparisons is not None else board.comparison_log),
        size_tie_ratio=size_tie_ratio,
        size_tie_min_samples=size_tie_min_samples,
        parameter_counts=model_parameter_counts(),
    )


def print_leaderboard(
    board: Leaderboard,
    failed_models: list[str] | None = None,
    failed_outputs: dict[str, int] | None = None,
    *,
    size_tie_ratio: float | None = None,
    size_tie_min_samples: int = 10,
    coverage: dict[str, ModelCoverage] | None = None,
) -> None:
    """Print leaderboard as a Rich table, leaving failed runs unranked.

    With ``coverage``, a Pages column shows on how many pages of the judged
    sample each model is ranked (see :class:`ocr_bench.integrity.ModelCoverage`).
    """
    from ocr_bench.publish import _get_model_sizes

    sizes = _get_model_sizes()
    preferences: dict[str, list[AdjacentPairDecision]] = {}
    if size_tie_ratio is not None:
        preferences = practical_preferences(
            _decisions_for_board(
                board,
                None,
                size_tie_ratio=size_tie_ratio,
                size_tie_min_samples=size_tie_min_samples,
            )
        )
    table = Table(title="OCR Model Leaderboard")
    table.add_column("Rank", style="bold")
    table.add_column("Model")
    table.add_column("Params", justify="right")
    has_ci = bool(board.elo_ci)
    if has_ci:
        table.add_column("ELO (95% CI)", justify="right")
    else:
        table.add_column("ELO", justify="right")
    table.add_column("Wins", justify="right")
    table.add_column("Losses", justify="right")
    table.add_column("Ties", justify="right")
    table.add_column("Win%", justify="right")
    if coverage:
        table.add_column("Pages", justify="right")

    def pages_cell(model: str) -> list[str]:
        if not coverage:
            return []
        entry = coverage.get(model)
        if entry is None:
            return ["-"]
        return [f"{entry.ranked_pages}/{entry.sampled_pages}"]

    failed = set(failed_models or [])
    rank = 0
    for model, elo in board.ranked:
        if model in failed:
            continue
        rank += 1
        pct = board.win_pct(model)
        pct_str = f"{pct:.0f}%" if pct is not None else "-"
        if has_ci and model in board.elo_ci:
            lo, hi = board.elo_ci[model]
            elo_str = f"{round(elo)} ({round(lo)}\u2013{round(hi)})"
        else:
            elo_str = str(round(elo))
        model_label = model
        if (failed_outputs or {}).get(model):
            model_label += " ⚠"
        if model in preferences:
            model_label += " ★"
        table.add_row(
            str(rank),
            model_label,
            sizes.get(model, ""),
            elo_str,
            str(board.wins[model]),
            str(board.losses[model]),
            str(board.ties[model]),
            pct_str,
            *pages_cell(model),
        )

    for model in sorted(failed):
        table.add_row(
            "-",
            f"{model} [bold red]FAILED[/bold red]",
            sizes.get(model, ""),
            "-",
            "-",
            "-",
            "-",
            "-",
            *pages_cell(model),
        )

    console.print(table)
    if coverage:
        _print_coverage_note(coverage)
    if preferences:
        console.print(
            "[dim]★ Parameter-efficient practical preference: adjacent 95% CIs overlap, "
            "but this model is sufficiently smaller under --size-tie-ratio. "
            "This does not change ELO/rank or prove equivalence.[/dim]"
        )
        for smaller, decisions in preferences.items():
            for decision in decisions:
                console.print(
                    f"  [dim]{smaller} over {decision.larger_model}: "
                    f"{decision.size_ratio:.1f}× fewer parameters after "
                    f"{decision.direct_comparisons} direct comparisons[/dim]"
                )


def _print_coverage_note(coverage: dict[str, ModelCoverage]) -> None:
    """Explain the Pages column, and name the models whose rank leaves out sentinel pages."""
    sampled = max(entry.sampled_pages for entry in coverage.values())
    console.print(
        f"[dim]Pages: ranked on N of the {sampled} sampled pages, the pages with at least "
        "one judged or auto-tied comparison for the model. Pages where all outputs are "
        "under --min-chars are not judged. Adaptive stopping or --max-comparisons can "
        "stop the run before every page is judged.[/dim]"
    )
    for model, entry in coverage.items():
        if entry.sentinel_pages:
            console.print(
                f"  {model}: ranked on {entry.ranked_pages} of {entry.sampled_pages} "
                f"pages; {entry.sentinel_pages} error sentinel(s) left out."
            )


def _warn_text_cap(
    ds: object,
    ocr_columns: dict[str, str],
    indices: list[int],
    max_ocr_text_len: int,
    normalize: bool,
) -> None:
    """Warn before judging when the text cap truncates over half the sampled outputs.

    The suggested cap comes from the ``audit --screen`` checks, run on the
    judged sample. This only prints; the run keeps ``max_ocr_text_len``.
    """
    exposure = truncation_exposure(ds, ocr_columns, indices, max_ocr_text_len, normalize=normalize)
    if not exposure.warn:
        return
    console.print(
        f"[yellow]⚠ Text cap:[/yellow] --max-ocr-text-len {max_ocr_text_len} truncates "
        f"{exposure.total_over_cap} of {exposure.total_outputs} outputs "
        f"({exposure.overall_rate:.0%}) in the judged sample. The judge sees the first "
        f"{max_ocr_text_len} characters of each, with a note that the output was cut."
    )
    over_half = exposure.models_over_warn_rate
    if over_half:
        listed = ", ".join(f"{model} ({exposure.rate(model):.0%})" for model in over_half)
        console.print(f"  Over half are truncated for: {listed}.")
    screen = suggest_text_cap(ds, ocr_columns, indices, max_ocr_text_len, normalize=normalize)
    if screen is None:
        console.print(
            "  No cap is suggested: the screen checks need outputs from 3 or more models. "
            "`ocr-bench audit` shows the output lengths."
        )
        return
    suggested = screen.suggested_max_ocr_text_len
    if suggested is None:
        # The screen's flags are leads: a flagged output can still be a complete
        # transcription (for example the one long output on a page where the
        # other models stopped early), so do not say that a higher cap cannot help.
        console.print(
            f"  --max-ocr-text-len {max_ocr_text_len} fits every output that the "
            "`audit --screen` checks do not flag. Every output over the cap is flagged "
            "(for example as a loop, or as over 5x the others' length). Inspect them "
            "before you choose a cap: `ocr-bench audit --screen --json` lists the "
            "flagged rows."
        )
        return
    console.print(
        f"  --max-ocr-text-len {suggested} fits every output that the `audit --screen` "
        "checks do not flag. A higher cap makes each judge call longer. To use it, pass "
        f"it to `ocr-bench judge`; a results repo judged with another cap needs "
        f"--full-rejudge. This run keeps {max_ocr_text_len}."
    )


def _merge_auto_ties(comparisons: list[Comparison], judged: list[dict]) -> list[dict]:
    """Interleave judge outputs with pre-decided auto-ties, in original order.

    ``judged`` holds one result per comparison whose ``auto_result`` is None
    (the pairs actually sent to the judge, in order). Auto-tie comparisons slot
    their ``auto_result`` back in place, yielding a result list parallel to
    ``comparisons`` for :func:`_convert_results`.
    """
    judged_iter = iter(judged)
    return [
        comp.auto_result if comp.auto_result is not None else next(judged_iter)
        for comp in comparisons
    ]


def _trim_to_budget(
    comparisons: list[Comparison], judge_budget: int
) -> tuple[list[Comparison], bool]:
    """Cut ``comparisons`` after the last judged pair that fits ``judge_budget``.

    ``comparisons`` is in visiting order, and a budget stops the run at one
    point in that order. Every comparison up to and including the last judged
    pair that fits the budget is kept, auto-ties included. Every comparison
    after it is dropped, auto-ties included. An auto-tie costs no judge call,
    but one from a page the run did not reach would give its pair ties, and
    its two models pages, from a part of the sample where no other pair was
    judged. When no judged pair fits the budget, nothing is kept.

    A list with at most ``judge_budget`` judged pairs is returned unchanged.
    Returns the trimmed list and whether any judged pair was dropped to fit
    the budget.
    """
    judgeable = sum(c.auto_result is None for c in comparisons)
    if judgeable <= judge_budget:
        return comparisons, False
    judged_kept = 0
    keep_count = 0  # length of the leading part of the list that is kept
    for position, comp in enumerate(comparisons):
        if comp.auto_result is not None:
            continue
        if judged_kept == judge_budget:
            break
        judged_kept += 1
        keep_count = position + 1
    return comparisons[:keep_count], True


def _shown_first(comp: Comparison) -> ShownFirst:
    """Which model's output the judge saw first: "A" (model_a), "B" (model_b), or ""."""
    if comp.auto_result is not None:
        # An auto-tie never reached a judge, so it has no display order.
        return ""
    return "B" if comp.swapped else "A"


def _convert_results(
    comparisons: list[Comparison],
    aggregated: list[dict],
    *,
    provenance_hashes: dict[tuple[str, str], str] | None = None,
) -> list[ComparisonResult]:
    """Convert judged comparisons + aggregated outputs into ComparisonResult list."""
    results: list[ComparisonResult] = []
    for comp, result in zip(comparisons, aggregated, strict=True):
        # Skip failures: empty dict (single judge failed) and 0/0 "ties"
        # (every judge in a jury failed) — neither is a real verdict.
        if not result or result.get("agreement") == "0/0":
            continue
        results.append(
            ComparisonResult(
                sample_idx=comp.sample_idx,
                model_a=comp.model_a,
                model_b=comp.model_b,
                winner=result.get("winner", "tie"),
                reason=result.get("reason", ""),
                agreement=result.get("agreement", "1/1"),
                swapped=comp.swapped,
                shown_first=_shown_first(comp),
                text_a=comp.text_a,
                text_b=comp.text_b,
                col_a=comp.col_a,
                col_b=comp.col_b,
                truncated_a=comp.truncated_a,
                truncated_b=comp.truncated_b,
                provenance_hash=(provenance_hashes or {}).get(
                    tuple(sorted((comp.col_a, comp.col_b))), ""
                ),
            )
        )
    return results


def _resolve_results_repo(dataset: str, save_results: str | None, no_publish: bool) -> str | None:
    """Derive the results repo id. Returns None if publishing is disabled."""
    if no_publish:
        return None
    if save_results:
        return save_results
    return f"{dataset}-results"


def _check_visibility_flags(args: argparse.Namespace) -> None:
    """Reject --private together with --public before anything runs."""
    if getattr(args, "private", False) and getattr(args, "public", False):
        raise DatasetError("--private and --public cannot be used together.")


def _prepare_destination(
    repo_id: str,
    *,
    private: bool,
    allow_public: bool,
    source_repo: str,
    label: str,
    create: bool = True,
) -> None:
    """Create or check ``repo_id`` before data derived from ``source_repo`` is written.

    Raises :class:`DatasetError` (via :func:`ensure_dataset_repo`) when the
    repo must be private and is public. ``create=False`` only checks an
    existing repo and prints nothing, for an early check before other writes.
    """
    visibility = ensure_dataset_repo(
        repo_id,
        private=private,
        allow_public=allow_public,
        source_repo=source_repo,
        create=create,
    )
    if not create:
        return
    if visibility.created:
        kind = "private" if visibility.private else "new"
        reason = (
            f" because [bold]{source_repo}[/bold] is private or gated"
            if visibility.inherited
            else ""
        )
        console.print(f"Created {kind} {label} repo [bold]{repo_id}[/bold]{reason}.")
    if visibility.public_with_restricted_source:
        console.print(
            f"[bold yellow]Warning:[/bold yellow] {label} repo [bold]{repo_id}[/bold] is "
            f"public although its source [bold]{source_repo}[/bold] is private or gated "
            f"(--public). The {label} will include text from the source pages."
        )


def _string_list(value: object) -> list[str] | None:
    """Decode a metadata list stored as a list or a JSON string; None if malformed."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    if not isinstance(value, list):
        return None
    strings = [v for v in value if isinstance(v, str) and v]
    if len(strings) != len(value):
        return None
    return strings


def _previous_judge_hint(existing_meta_rows: list[dict], requested_models: list[str] | None) -> str:
    """Name the earlier judge when a resume may have failed because the default judge changed.

    Returns an empty string unless the user relied on the default judge and the
    repo's latest metadata row (its last completed run) records a different
    judge or jury. Exact flags are suggested only when the recorded names hash
    to the recorded judge specs, i.e. the names are the specs that were passed;
    endpoint judges are recorded under a model name or URL digest instead.
    """
    if requested_models or not existing_meta_rows:
        return ""
    row = existing_meta_rows[-1]
    names = _string_list(row.get("judge_models"))
    if not names or names == [DEFAULT_JUDGE]:
        return ""
    hint = (
        f"\nThe last completed run in this repo used {', '.join(names)}; the default "
        f"judge is now {DEFAULT_JUDGE}. If the other settings match, "
    )
    if _string_list(row.get("judge_spec_hashes")) == hash_judge_specs(names):
        flags = " ".join(f"--model {name}" for name in names)
        bench_flags = " ".join(f"--judge-model {name}" for name in names)
        return hint + (
            f"passing [bold]{flags}[/bold] (with `ocr-bench bench`: "
            f"[bold]{bench_flags}[/bold]) may let you continue that board."
        )
    return hint + (
        "passing the same judge spec(s) as that run (`--model`, or `--judge-model` "
        "with `ocr-bench bench`) may let you continue that board."
    )


def _refresh_viewer_space(results_repo: str) -> None:
    """Keep the deployed ``{results}-viewer`` Space in sync after a judge run.

    Two layers of the issue #37 fix:

    - **Layer 0 (wiring-drift detector):** if the Space's ``REPOS`` variable is
      set but points at a *different* dataset than the one just published,
      restarting would faithfully reload the wrong data (the actual root cause
      of the Britannica incident). Warn loudly and skip.
    - **Layer 2 (freshness):** otherwise factory-reboot the Space so it reloads
      the just-published results instead of a stale in-memory snapshot.

    Wrapped so a Space hiccup (missing Space, transient Hub error) never fails
    an otherwise-successful judge run.
    """
    space_id = f"{results_repo}-viewer"
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        if not api.repo_exists(space_id, repo_type="space"):
            logger.info("no_viewer_space", space=space_id)
            return

        variables = api.get_space_variables(space_id)
        repos_var = variables.get("REPOS")
        wired_repo = repos_var.value if repos_var is not None else None

        if wired_repo and wired_repo != results_repo:
            console.print(
                f"\n[bold red]WARNING: viewer wiring drift[/bold red] — Space "
                f"[bold]{space_id}[/bold] has REPOS=[yellow]{wired_repo}[/yellow] but "
                f"results were just published to [yellow]{results_repo}[/yellow].\n"
                f"The Space is wired to a different dataset; restarting it would "
                f"reload the wrong results. [bold]Skipping restart[/bold] — fix the "
                f"Space's REPOS variable to re-sync."
            )
            logger.warning(
                "viewer_space_wiring_drift",
                space=space_id,
                wired_repo=wired_repo,
                published_repo=results_repo,
            )
            return

        api.restart_space(space_id, factory_reboot=True)
        console.print(f"Restarted viewer Space [bold]{space_id}[/bold] to load fresh results.")
        logger.info("restarted_viewer_space", space=space_id)
    except Exception as exc:
        logger.warning("viewer_space_refresh_failed", space=space_id, error=str(exc))


def _checkpoint(
    results_repo: str | None,
    results: list[ComparisonResult],
    model_names: list[str],
) -> None:
    """Push a comparisons-only checkpoint, swallowing any failure.

    Checkpointing is best-effort durability, not a correctness step: a failed
    push (transient Hub error, auth hiccup) must not abort a multi-hour judge
    run. Log a warning and continue — the next checkpoint, or the final
    publish, re-pushes the full accumulated set anyway.
    """
    if not results_repo:
        return
    try:
        publish_checkpoint(results_repo, results, model_names)
        console.print(f"  [dim]Checkpoint: pushed {len(results)} comparisons[/dim]")
    except Exception as exc:
        logger.warning("checkpoint_failed", error=str(exc))
        console.print(f"  [yellow]Checkpoint failed (continuing): {exc}[/yellow]")


def _unresolved_adjacent_pairs(
    board: Leaderboard,
    *,
    size_tie_ratio: float | None = None,
    size_tie_min_samples: int = 10,
) -> list[str]:
    """Adjacent pairs still unresolved under the selected adaptive rule."""
    decisions = _decisions_for_board(
        board,
        None,
        size_tie_ratio=size_tie_ratio,
        size_tie_min_samples=size_tie_min_samples,
    )
    return [
        f"{decision.higher_model} vs {decision.lower_model}"
        for decision in decisions
        if decision.status == "unresolved"
    ]


def _resolve_criteria(args: argparse.Namespace) -> tuple[str, str, str]:
    """Resolve (criteria_name, prompt_template, prompt_hash) from the CLI args.

    Enforces that ``--criteria`` and ``--criteria-file`` are mutually exclusive
    (``--criteria`` defaults to ``None``, so an explicit value is distinguishable
    from unset — this exclusivity is enforced here rather than via an argparse
    group, which is unreliable on Python 3.13.0). ``--criteria-file`` loads and
    validates a custom prompt template, named ``custom:<filename>``; its hash is
    computed exactly as for a built-in profile, so provenance and the mixing
    guard treat custom and built-in rubrics identically. Otherwise the named
    built-in profile (or the default when unset) is used. Raises ``DatasetError``
    on a conflict, or an unreadable/invalid custom file (clean CLI error).
    """
    criteria = getattr(args, "criteria", None)
    criteria_file = getattr(args, "criteria_file", None)
    if criteria_file and criteria is not None:
        raise DatasetError("--criteria and --criteria-file are mutually exclusive; pass only one")
    if criteria_file:
        try:
            template = Path(criteria_file).read_text(encoding="utf-8")
        except OSError as exc:
            raise DatasetError(f"could not read --criteria-file '{criteria_file}': {exc}") from exc
        try:
            validate_prompt_template(template)
        except ValueError as exc:
            raise DatasetError(f"invalid --criteria-file '{criteria_file}': {exc}") from exc
        return f"custom:{Path(criteria_file).name}", template, prompt_hash(template)
    name = criteria or DEFAULT_CRITERIA
    template = CRITERIA_PROFILES[name]
    return name, template, prompt_hash(template)


def _print_passed_over_prs(
    dataset: str,
    duplicate_prs: dict[str, list[int]],
    superseded_prs: dict[str, list[int]],
    loaded_pr_revisions: dict[str, str],
    merge: bool,
) -> None:
    """Tell the user which open PRs discovery did not use, and why."""
    discussions = f"https://huggingface.co/datasets/{dataset}/discussions"
    for config, older in sorted(duplicate_prs.items()):
        used = loaded_pr_revisions.get(config, "a merged PR").replace("refs/pr/", "#")
        ignored = ", ".join(f"#{n}" for n in sorted(older))
        console.print(
            f"[yellow]Note:[/yellow] {config} has several open PRs; using the newest "
            f"({used}) and ignoring {ignored}. Close stale ones at {discussions}"
        )
    for config, numbers in sorted(superseded_prs.items()):
        ignored = ", ".join(f"#{n}" for n in numbers)
        not_merged = " and not merging the PR" if merge else ""
        console.print(
            f"[yellow]Note:[/yellow] {config} changed on main after its open PR "
            f"({ignored}); using main{not_merged}. To use the PR instead, merge it. "
            f"Close stale PRs at {discussions}"
        )


def cmd_judge(args: argparse.Namespace) -> None:
    """Orchestrate: load → compare → judge → elo → print → publish."""
    # --- Resolve flags ---
    adaptive = not args.no_adaptive
    merge = args.merge
    results_repo = _resolve_results_repo(args.dataset, args.save_results, args.no_publish)
    from_prs = False  # track for metadata
    max_comparisons = args.max_comparisons  # global budget; None = uncapped
    adaptive_strategy = args.adaptive_strategy
    if not adaptive and adaptive_strategy != "balanced":
        raise DatasetError("--adaptive-strategy targeted requires adaptive mode")
    if adaptive_strategy == "targeted":
        console.print(
            "[yellow]Targeted adaptive allocation is opt-in:[/yellow] after the initial "
            "balanced evidence threshold, only unresolved adjacent pairs receive "
            "more samples. The allocation strategy is recorded in results metadata."
        )
    normalize = args.judge_text_mode == "normalized"
    if not normalize:
        console.print(
            "[yellow]judge-text-mode=raw:[/yellow] OCR text is capped WITHOUT HTML "
            "normalization, so HTML-verbose outputs (tables) burn the char cap on "
            "markup and get truncated mid-content while compact formats fit whole — "
            "the #45 format bias. 'normalized' is the default for exactly this "
            "reason; use raw only to judge markup quality or audit the transformation."
        )

    # Judge criteria profile: the prompt template every comparison is built with,
    # plus a stable hash of it recorded in the results metadata for provenance.
    # A built-in profile (--criteria) or a validated custom file (--criteria-file).
    criteria, prompt_template, criteria_prompt_hash = _resolve_criteria(args)

    # Resolve checkpoint cadence (args.checkpoint_every: None = unspecified).
    #
    # Checkpointing REPLACES the results repo's comparisons config with the full
    # accumulated set (existing + new). Under --full-rejudge, existing is empty,
    # so a checkpoint would overwrite the previously-complete published
    # comparisons with only THIS run's partial set — and a death between the
    # first checkpoint and the final publish would leave that partial set as the
    # published state. So default checkpointing OFF under --full-rejudge. If the
    # user explicitly opted in (N > 0), honor it but warn loudly about the risk.
    if args.checkpoint_every is None:
        if args.full_rejudge:
            checkpoint_every = 0
            logger.info("checkpoint_disabled_full_rejudge")
            console.print(
                "[dim]Checkpointing off under --full-rejudge (a mid-run checkpoint "
                "would replace the complete published comparisons with this run's "
                "partial set). Pass --checkpoint-every N to override.[/dim]"
            )
        else:
            checkpoint_every = 500
    else:
        checkpoint_every = args.checkpoint_every  # 0 = off
        if args.full_rejudge and checkpoint_every > 0:
            console.print(
                "[bold red]WARNING[/bold red]: --checkpoint-every with "
                "--full-rejudge — checkpoints REPLACE the published comparisons "
                "config with this run's partial set. If the run dies before the "
                "final publish, the previously-complete comparisons are lost."
            )
            logger.warning(
                "checkpoint_full_rejudge_clobber_risk", checkpoint_every=checkpoint_every
            )

    _check_visibility_flags(args)
    # A missing source repo stops here with a plain message, before the
    # results-repo check and PR discovery make their own Hub requests.
    require_dataset_repo(args.dataset)
    if results_repo:
        console.print(f"Results will be published to [bold]{results_repo}[/bold]")
        # Read-only check first: an existing results repo that must be private
        # but is public stops the run before anything is written, including
        # the PR merges that --merge performs during discovery. A missing repo
        # is created after the source has loaded (see below).
        _prepare_destination(
            results_repo,
            private=args.private,
            allow_public=args.public,
            source_repo=args.dataset,
            label="results",
            create=False,
        )
    elif args.private or args.public:
        console.print("[dim]--private/--public have no effect with --no-publish.[/dim]")

    # --- Load dataset (cascading auto-detection) ---
    source_configs: list[str] = []
    source_output_columns: dict[str, str] = {}
    # config -> PR revision actually loaded (absent = main), for metadata.
    loaded_pr_revisions: dict[str, str] = {}
    # config -> older open PR numbers passed over for the newest one.
    duplicate_prs: dict[str, list[int]] = {}
    # config -> open PR numbers passed over because main changed after them.
    superseded_prs: dict[str, list[int]] = {}
    if args.configs and args.from_prs:
        # Explicit configs, taken from open PRs where one exists (newest PR
        # per config) and from main otherwise.
        config_names = list(args.configs)
        found, pr_revisions = discover_pr_configs(
            args.dataset,
            merge=merge,
            only=config_names,
            duplicates=duplicate_prs,
            main_configs=discover_configs(args.dataset),
            superseded=superseded_prs,
        )
        from_prs = bool(found)
        on_main = [c for c in config_names if c not in found]
        from_pr_text = ", ".join(
            f"{c} ({pr_revisions[c].replace('refs/pr/', '#')})" if c in pr_revisions else c
            for c in found
        )
        console.print(
            f"Selected {len(config_names)} configs: [bold]{len(found)} from PRs[/bold]"
            + (f" ({from_pr_text})" if found else "")
            + (f", [bold]{len(on_main)} from main[/bold] {on_main}" if on_main else "")
        )
        if not merge:
            loaded_pr_revisions = dict(pr_revisions)
        source_configs = list(config_names)
        ds, ocr_columns = load_config_dataset(
            args.dataset,
            config_names,
            split=args.split,
            pr_revisions=pr_revisions if not merge else None,
            source_output_columns=source_output_columns,
        )
    elif args.configs:
        # Explicit configs — use them directly
        config_names = args.configs
        source_configs = list(config_names)
        ds, ocr_columns = load_config_dataset(
            args.dataset,
            config_names,
            split=args.split,
            source_output_columns=source_output_columns,
        )
    elif args.columns:
        # Explicit columns — flat loading
        ds, ocr_columns = load_flat_dataset(args.dataset, split=args.split, columns=args.columns)
    elif args.from_prs:
        # Forced PR discovery. A config that changed on main after its newest
        # open PR is still judged, from main.
        pr_configs, pr_revisions = discover_pr_configs(
            args.dataset,
            merge=merge,
            duplicates=duplicate_prs,
            main_configs=discover_configs(args.dataset),
            superseded=superseded_prs,
        )
        config_names = pr_configs + list(superseded_prs)
        if not config_names:
            raise DatasetError("No configs found in open PRs")
        from_prs = bool(pr_configs)
        if not merge:
            loaded_pr_revisions = dict(pr_revisions)
        source_configs = list(config_names)
        from_main = f", {len(superseded_prs)} from main: {list(superseded_prs)}"
        console.print(
            f"Discovered {len(pr_configs)} configs from PRs: {pr_configs}"
            + (from_main if superseded_prs else "")
        )
        ds, ocr_columns = load_config_dataset(
            args.dataset,
            config_names,
            split=args.split,
            pr_revisions=pr_revisions if not merge else None,
            source_output_columns=source_output_columns,
        )
    else:
        # Auto-detect: PRs + main branch configs combined, fall back to flat
        main_configs = discover_configs(args.dataset)
        pr_configs, pr_revisions = discover_pr_configs(
            args.dataset,
            merge=merge,
            duplicates=duplicate_prs,
            main_configs=main_configs,
            superseded=superseded_prs,
        )
        if merge:
            # The merges can add or remove configs on main: read them again.
            main_configs = discover_configs(args.dataset)
        else:
            loaded_pr_revisions = dict(pr_revisions)

        # Combine: PR configs + main configs not already in PRs
        config_names = list(pr_configs)
        for mc in main_configs:
            if mc not in pr_configs:
                config_names.append(mc)

        if config_names:
            source_configs = list(config_names)
            if pr_configs:
                from_prs = True
                console.print(f"Auto-detected {len(pr_configs)} configs from PRs: {pr_configs}")
            if main_configs:
                main_only = [c for c in main_configs if c not in pr_configs]
                if main_only:
                    console.print(f"Auto-detected {len(main_only)} configs on main: {main_only}")
            ds, ocr_columns = load_config_dataset(
                args.dataset,
                config_names,
                split=args.split,
                pr_revisions=pr_revisions if pr_configs else None,
                source_output_columns=source_output_columns,
            )
        else:
            # No configs anywhere — fall back to flat loading
            ds, ocr_columns = load_flat_dataset(args.dataset, split=args.split)

    if not source_configs:
        source_output_columns = {column: column for column in ocr_columns}
    source_revisions = {c: loaded_pr_revisions.get(c, "main") for c in source_configs}
    _print_passed_over_prs(
        args.dataset, duplicate_prs, superseded_prs, loaded_pr_revisions, merge=merge
    )

    console.print(f"Loaded {len(ds)} samples with {len(ocr_columns)} models:")
    for col, model in ocr_columns.items():
        console.print(f"  {col} → {model}")

    # --- Input integrity: sentinel outputs (issue #46) ---
    # Error sentinels (e.g. "[OCR ERROR]") are excluded from judging by
    # build_comparisons; here we count them per model for the metadata + card
    # and warn loudly when a model's run largely failed on this corpus.
    model_stats = compute_model_stats(
        ds,
        ocr_columns,
        max_ocr_text_len=args.max_ocr_text_len,
        normalize=normalize,
    )
    failed_outputs = failed_output_counts(model_stats)
    # A fully sentinel-backed run has no comparable OCR output at all. Keep it
    # visible as FAILED, but never assign it an arbitrary disconnected ELO/rank.
    # Partial failures remain rankable on their successful outputs with a
    # degraded warning, while the audit still blocks rates over its threshold.
    failed_models = sorted(
        stat.model for stat in model_stats if stat.n_rows > 0 and stat.n_sentinel == stat.n_rows
    )
    for stat in model_stats:
        if stat.sentinel_rate > SENTINEL_FLAG_RATE:
            if stat.model in failed_models:
                consequence = "this run is marked FAILED and receives no leaderboard rank"
            else:
                consequence = "its rank uses successful outputs only and is marked degraded"
            console.print(
                f"[yellow]⚠ {stat.model}: {stat.n_sentinel}/{stat.n_rows} outputs are "
                f"error sentinels ({stat.sentinel_rate:.0%}) — excluded from judging; "
                f"{consequence}.[/yellow]"
            )
        elif stat.n_sentinel:
            # Under the warning rate, but its rank still rests on fewer pages.
            console.print(
                f"Note: {stat.model}: {stat.n_sentinel}/{stat.n_rows} outputs are error "
                f"sentinels ({stat.sentinel_rate:.1%}). They are excluded from judging, "
                "so its rank leaves out those pages."
            )

    # The judged sample: the text-cap warning, the fingerprints and the coverage
    # report use it in this order. judge visits it in judging_order (below).
    judged_sample = sample_indices(len(ds), args.max_samples, args.seed)
    _warn_text_cap(ds, ocr_columns, judged_sample, args.max_ocr_text_len, normalize)

    # Stable ordering matters when ELOs tie: adjacency drives targeted sampling.
    model_names = sorted(set(ocr_columns.values()) - set(failed_models))
    current_models = set(model_names)

    model_specs = args.models or [DEFAULT_JUDGE]
    judges = [
        parse_judge_spec(spec, max_tokens=args.max_tokens, concurrency=args.concurrency)
        for spec in model_specs
    ]
    is_jury = len(judges) > 1
    judge_names = [judge.name for judge in judges]
    judge_spec_hashes = hash_judge_specs(model_specs)
    requested_max_samples = args.max_samples or len(ds)
    source_fingerprint = ""
    source_column_fingerprints: dict[str, str] = {}
    pair_provenance_hashes: dict[tuple[str, str], str] = {}
    if results_repo:
        source_fingerprint, source_column_fingerprints = evaluation_input_fingerprints(
            ds, ocr_columns, judged_sample
        )
        from itertools import combinations as _provenance_pairs

        for col_a, col_b in _provenance_pairs(ocr_columns, 2):
            pair = (min(col_a, col_b), max(col_a, col_b))
            pair_provenance_hashes[pair] = evaluation_provenance_hash(
                source_dataset=args.dataset,
                source_split=args.split,
                source_fingerprint=source_fingerprint,
                source_column_fingerprints={
                    column: source_column_fingerprints[column] for column in pair
                },
                source_columns={column: ocr_columns[column] for column in pair},
                source_output_columns={column: source_output_columns[column] for column in pair},
                judge_specs=model_specs,
                seed=args.seed,
                max_samples=requested_max_samples,
                prompt_hash=criteria_prompt_hash,
                judge_text_mode=args.judge_text_mode,
                max_ocr_text_len=args.max_ocr_text_len,
                judge_image_dim=args.judge_image_dim,
                max_tokens=args.max_tokens,
                min_chars=args.min_chars,
            )

    # Create or check the results repo after the source has loaded (a bad
    # dataset id leaves no empty repo) and before any judge call or checkpoint
    # (push_to_hub would otherwise create a missing repo with the account's
    # default visibility, usually public).
    if results_repo:
        _prepare_destination(
            results_repo,
            private=args.private,
            allow_public=args.public,
            source_repo=args.dataset,
            label="results",
        )

    # --- Incremental: load existing comparisons ---
    existing_results: list[ComparisonResult] = []
    # Comparisons outside this invocation's rankable model set (subset/retired
    # configs or currently failed models) are excluded from the current ELO fit
    # but preserved on Hub. A rerun must never erase judged history.
    preserved_out_of_grid_results: list[ComparisonResult] = []
    existing_meta_rows: list[dict] = []
    # Maps a normalized (model_a, model_b) pair to the sample indices already
    # judged for it, so build_comparisons skips only those exact (pair, sample)
    # combinations.
    skip_samples: dict[tuple[str, str], set[int]] | None = None

    if results_repo and not args.full_rejudge:
        # Resume path. Existing comparisons include any pushed by a checkpoint
        # (see --checkpoint-every), so a killed run relaunched WITHOUT
        # --full-rejudge picks up where it left off instead of re-judging.
        #
        # The skip is (pair, sample_idx)-level: a pair judged on only some
        # samples before the kill is TOPPED UP on the samples it hasn't been
        # judged on yet, rather than being dropped wholesale. This matters for
        # adaptive runs, where a checkpoint can persist a pair at (say) 3/50
        # samples — a pair-level skip would freeze it there forever.
        # --full-rejudge forces a clean re-run (ignores all existing).
        existing_results = load_existing_comparisons(results_repo)
        # Comparisons outside the rankable current grid cannot enter this ELO
        # fit, but remain part of the durable Hub history and are re-published
        # alongside current-grid rows.
        preserved_out_of_grid_results = [
            result
            for result in existing_results
            if result.model_a not in current_models or result.model_b not in current_models
        ]
        existing_results = [
            result
            for result in existing_results
            if result.model_a in current_models and result.model_b in current_models
        ]

        # The metadata config is an append-only run history.
        existing_meta_rows = load_existing_metadata(results_repo)

        if preserved_out_of_grid_results:
            console.print(
                f"[yellow]Preserving {len(preserved_out_of_grid_results)} historical "
                "comparison(s) outside the rankable current model set; "
                "they are excluded from this ELO fit.[/yellow]"
            )
            logger.info(
                "preserving_out_of_grid_comparisons",
                repo=results_repo,
                n=len(preserved_out_of_grid_results),
            )

        if existing_results or preserved_out_of_grid_results:
            # Every reused row must have been produced under the exact same
            # source/config/split/fingerprint, judge or jury, sample selection,
            # prompt, and preprocessing. Checkpoint rows carry this hash too,
            # so an interrupted first run can resume before final metadata exists.
            # Rows without a hash cannot be checked, so they need --full-rejudge.
            existing_hashes = {result.provenance_hash for result in existing_results}
            if "" in existing_hashes:
                console.print(
                    f"[red]Error:[/red] existing comparisons in "
                    f"[bold]{results_repo}[/bold] lack resume provenance. They were "
                    "written by an older ocr-bench version (or an incomplete unsafe "
                    "checkpoint), so their sample indices cannot be verified against "
                    "the current source. Use [bold]--full-rejudge[/bold] once to "
                    "replace them with provenance-stamped results."
                )
                sys.exit(1)
            mismatched = []
            for result in existing_results:
                pair = (
                    min(result.col_a, result.col_b),
                    max(result.col_a, result.col_b),
                )
                expected = pair_provenance_hashes.get(pair)
                if expected is None or result.provenance_hash != expected:
                    mismatched.append(result)
            if mismatched:
                recorded = ", ".join(sorted({result.provenance_hash[:12] for result in mismatched}))
                console.print(
                    f"[red]Error:[/red] resume provenance mismatch for "
                    f"[bold]{results_repo}[/bold] — existing comparison hash(es) "
                    f"{recorded}. The "
                    "source repo/config/split/fingerprint, judge or jury, sampling, "
                    "prompt, or preprocessing changed. Re-run with the original "
                    "settings, or use [bold]--full-rejudge[/bold]."
                    + _previous_judge_hint(existing_meta_rows, args.models)
                )
                sys.exit(1)

            if existing_results:
                skip_samples = {}
                for r in existing_results:
                    skip_samples.setdefault(_normalize_pair(r.model_a, r.model_b), set()).add(
                        r.sample_idx
                    )
                console.print(
                    f"\nIncremental mode: {len(existing_results)} reusable existing "
                    f"comparisons across {len(skip_samples)} model pairs — skipping "
                    "already-judged (pair, sample) combinations, topping up the rest."
                )
                no_display_order = sum(
                    1 for r in existing_results if r.agreement != "auto" and not r.shown_first
                )
                if no_display_order:
                    console.print(
                        f"[yellow]Note:[/yellow] {no_display_order} of them have no recorded "
                        "display order: an earlier ocr-bench version judged them. If that "
                        "run was adaptive or was resumed, a pair of models could be shown the "
                        "same way round on most of its pages. They are kept as they are; "
                        "[bold]--full-rejudge[/bold] judges them again."
                    )
            else:
                console.print(
                    "\nNo existing comparisons match the current model grid; "
                    "starting its judge run while preserving out-of-grid history."
                )
        else:
            console.print("\nNo existing comparisons found — full judge run.")

    publication_model_names = sorted(
        current_models
        | {
            model
            for result in preserved_out_of_grid_results
            for model in (result.model_a, result.model_b)
        }
    )

    console.print(f"Judge criteria: [bold]{criteria}[/bold] (prompt {criteria_prompt_hash})")
    logger.info("judge_criteria", criteria=criteria, prompt_hash=criteria_prompt_hash)

    def _judge_batch(batch_comps: list[Comparison]) -> list[ComparisonResult]:
        """Run judge(s) on a batch, returning ComparisonResults.

        Auto-tie comparisons never reach a judge; their verdict is merged back
        in :func:`_merge_auto_ties` so ELO sees them as ordinary ties.
        """
        judgeable = [c for c in batch_comps if c.auto_result is None]
        if judgeable:
            all_judge_outputs: list[list[dict]] = [judge.judge(judgeable) for judge in judges]
            if is_jury:
                judge_names = [j.name for j in judges]
                aggregated = aggregate_jury_votes(all_judge_outputs, judge_names)
            else:
                aggregated = all_judge_outputs[0]
        else:
            aggregated = []
        merged = _merge_auto_ties(batch_comps, aggregated)
        return _convert_results(
            batch_comps,
            merged,
            provenance_hashes=pair_provenance_hashes,
        )

    # Set when the run stops because it hit --max-comparisons (vs converging or
    # running out of samples). Recorded in the metadata row and drives the
    # unresolved-pairs report at the end.
    budget_exhausted = False

    if adaptive:
        # --- Adaptive stopping: batch-by-batch with convergence check ---
        from itertools import combinations as _combs

        # Visit the sample in a random order, so a budget or convergence stop
        # leaves a random subset of pages, not the end of a sorted dataset.
        all_indices = judging_order(judged_sample, args.seed)
        all_model_pairs = {
            _normalize_pair(model_a, model_b) for model_a, model_b in _combs(model_names, 2)
        }
        n_pairs = len(all_model_pairs)
        batch_samples = 5
        min_before_check = max(3 * n_pairs, 20)

        if is_jury:
            console.print(f"\nJury mode: {len(judges)} judges")
        console.print(
            f"\n[bold]Adaptive mode[/bold]: {len(all_indices)} samples, "
            f"{n_pairs} pairs, batch size {batch_samples}, "
            f"checking after {min_before_check} comparisons"
        )
        if max_comparisons is not None:
            console.print(f"Budget: stop after {max_comparisons} comparisons")

        def _stopping_decisions(
            results: list[ComparisonResult],
        ) -> tuple[Leaderboard, list[AdjacentPairDecision]]:
            """Fit the board on ``results`` and classify its adjacent pairs.

            The run stops when no pair is unresolved. The check before the
            first batch and the check after each batch both call this, so they
            apply one rule.
            """
            fitted_board = compute_elo(results, model_names)
            pair_decisions = _decisions_for_board(
                fitted_board,
                results,
                size_tie_ratio=args.size_tie_ratio,
                size_tie_min_samples=args.size_tie_min_samples,
            )
            return fitted_board, pair_decisions

        new_results: list[ComparisonResult] = []
        total_comparisons = 0  # judge calls made this run (auto-ties excluded)
        last_checkpoint = 0
        n_auto_total = 0
        # ``None`` preserves the existing balanced all-pairs behavior. Targeted
        # mode fills this only AFTER enough balanced evidence exists to compute
        # a first board, so it never starts from a disconnected sparse graph.
        active_pairs: set[tuple[str, str]] | None = None
        existing_models = {
            model for result in existing_results for model in (result.model_a, result.model_b)
        }
        existing_pair_counts = comparison_pair_counts(existing_results)
        missing_pair_evidence = {
            pair for pair in all_model_pairs if existing_pair_counts.get(pair, 0) == 0
        }
        can_resume_targeted = existing_models == current_models and not missing_pair_evidence
        if (
            adaptive_strategy == "targeted"
            and len(existing_results) >= min_before_check
            and can_resume_targeted
        ):
            # Reconstruct the allocation state on resume. The comparison log
            # records completed (pair, sample) work, not the active pair set;
            # deriving it from the current board prevents a resumed targeted run
            # from filling every pair it intentionally deprioritised previously.
            _, resumed_decisions = _stopping_decisions(existing_results)
            active_pairs = unresolved_pairs(resumed_decisions)
            if active_pairs:
                console.print(
                    f"  [dim]Resume: targeting {len(active_pairs)} unresolved "
                    f"adjacent pair(s), not all {n_pairs} pairs.[/dim]"
                )
            else:
                console.print(
                    "[green]Existing comparisons already meet the adaptive "
                    "stopping criteria.[/green]"
                )
                all_indices = []
        elif adaptive_strategy == "targeted" and existing_results and not can_resume_targeted:
            missing = sorted(current_models - existing_models)
            reason = (
                f"new models: {', '.join(missing)}"
                if missing
                else f"{len(missing_pair_evidence)} pair(s) have no direct evidence"
            )
            console.print(
                "  [dim]Model grid changed or its balanced warm-up is incomplete; "
                f"continuing balanced allocation ({reason}).[/dim]"
            )
        elif adaptive_strategy == "balanced" and len(existing_results) >= min_before_check:
            # The loop below checks the stopping rule only after it has judged
            # a batch. Check it here on the existing results, so a rerun of a
            # board that already meets the rule makes no judge call. A model
            # added since the last run has no interval, so its pairs are
            # unresolved and the run goes on.
            _, existing_decisions = _stopping_decisions(existing_results)
            if not unresolved_pairs(existing_decisions):
                console.print(
                    "[green]Existing comparisons already meet the adaptive "
                    "stopping criteria.[/green]"
                )
                all_indices = []

        for batch_num, batch_start in enumerate(range(0, len(all_indices), batch_samples)):
            # Global budget: stop before a batch we can't afford at all.
            if max_comparisons is not None and total_comparisons >= max_comparisons:
                budget_exhausted = True
                break

            batch_indices = all_indices[batch_start : batch_start + batch_samples]
            batch_comps = build_comparisons(
                ds,
                ocr_columns,
                skip_samples=skip_samples,
                include_pairs=active_pairs,
                indices=batch_indices,
                seed=args.seed,
                min_chars=args.min_chars,
                prompt_template=prompt_template,
                max_ocr_text_len=args.max_ocr_text_len,
                judge_image_dim=args.judge_image_dim,
                normalize=normalize,
            )
            if not batch_comps:
                continue

            # Trim the final batch to land exactly on the budget: the run stops
            # at the last judged pair that fits. The rest of the batch is
            # dropped, auto-ties included, so no pair gets ties from pages
            # where the other pairs were not judged.
            if max_comparisons is not None:
                batch_comps, hit_budget = _trim_to_budget(
                    batch_comps, max_comparisons - total_comparisons
                )
                if hit_budget:
                    budget_exhausted = True

            # Counted after the trim, which can drop auto-ties.
            batch_auto = sum(c.auto_result is not None for c in batch_comps)
            n_auto_total += batch_auto
            batch_results = _judge_batch(batch_comps)
            new_results.extend(batch_results)
            total_comparisons += len(batch_comps) - batch_auto
            # batch_comps goes out of scope → GC can free images

            total = len(existing_results) + len(new_results)
            console.print(f"  Batch {batch_num + 1}: {len(batch_results)} new, {total} total")

            # Periodic checkpoint: push the full accumulated comparison set,
            # including out-of-grid history excluded from this ELO fit.
            if (
                checkpoint_every
                and results_repo
                and total_comparisons - last_checkpoint >= checkpoint_every
            ):
                _checkpoint(
                    results_repo,
                    existing_results + new_results + preserved_out_of_grid_results,
                    publication_model_names,
                )
                last_checkpoint = total_comparisons

            if budget_exhausted:
                break

            if total >= min_before_check:
                board, decisions = _stopping_decisions(existing_results + new_results)

                if board.elo_ci:
                    console.print("  CI gaps:")
                    for decision in decisions:
                        hi_ci = board.elo_ci.get(decision.higher_model)
                        lo_ci = board.elo_ci.get(decision.lower_model)
                        if hi_ci is None or lo_ci is None:
                            continue
                        gap = hi_ci[0] - lo_ci[1]  # positive = statistically resolved
                        if decision.status == "resolved":
                            status = "[green]ok[/green]"
                        elif decision.status == "prefer-smaller":
                            status = (
                                f"[cyan]prefer {decision.smaller_model} "
                                f"({decision.size_ratio:.1f}× smaller)[/cyan]"
                            )
                        else:
                            status = f"[yellow]overlap {-gap:.0f}[/yellow]"
                        console.print(
                            f"    {decision.higher_model} vs {decision.lower_model}: "
                            f"gap={gap:+.0f} {status}"
                        )

                next_pairs = unresolved_pairs(decisions)
                if not next_pairs:
                    remaining = len(all_indices) - batch_start - len(batch_indices)
                    skipped_pairs = (
                        n_pairs
                        if adaptive_strategy == "balanced" or active_pairs is None
                        else len(active_pairs)
                    )
                    console.print(
                        f"[green]Adaptive stopping criteria met after {total} comparisons! "
                        f"Skipped up to ~{remaining * skipped_pairs} remaining.[/green]"
                    )
                    break

                if adaptive_strategy == "targeted":
                    active_pairs = next_pairs
                    console.print(
                        f"  [dim]Next batch targets {len(active_pairs)} unresolved "
                        f"adjacent pair(s), not all {n_pairs} pairs.[/dim]"
                    )

        if budget_exhausted:
            console.print(
                f"\n[yellow]Budget reached ({total_comparisons} comparisons) — "
                f"stopping and publishing.[/yellow]"
            )
        judged_valid = len(new_results) - n_auto_total
        console.print(f"\n{judged_valid}/{total_comparisons} valid comparisons")
        if n_auto_total:
            console.print(f"[dim]{n_auto_total} identical outputs auto-tied (no judge call)[/dim]")
    else:
        # --- Standard single-pass flow ---
        # Random visiting order: a --max-comparisons trim keeps the first pairs
        # built, so in dataset order it would drop the end of a sorted dataset.
        comparisons = build_comparisons(
            ds,
            ocr_columns,
            indices=judging_order(judged_sample, args.seed),
            seed=args.seed,
            skip_samples=skip_samples,
            min_chars=args.min_chars,
            prompt_template=prompt_template,
            max_ocr_text_len=args.max_ocr_text_len,
            judge_image_dim=args.judge_image_dim,
            normalize=normalize,
        )

        # Global budget: judge at most N pairs. The run stops at the last
        # judged pair that fits; the comparisons built after it are dropped,
        # auto-ties included. With the random visiting order above, they come
        # from random pages.
        if max_comparisons is not None:
            comparisons, hit_budget = _trim_to_budget(comparisons, max_comparisons)
            if hit_budget:
                budget_exhausted = True

        n_auto_total = sum(c.auto_result is not None for c in comparisons)
        n_judge_calls = len(comparisons) - n_auto_total
        if n_auto_total:
            console.print(
                f"\nBuilt {len(comparisons)} new pairwise comparisons "
                f"({n_judge_calls} to judge, {n_auto_total} auto-tied)"
            )
        else:
            console.print(f"\nBuilt {len(comparisons)} new pairwise comparisons")

        if not comparisons and not existing_results:
            console.print(
                "[yellow]No valid comparisons — check that OCR columns have text.[/yellow]"
            )
            return

        if not comparisons:
            console.print("[green]All pairs already judged — refitting leaderboard.[/green]")
            board = compute_elo(existing_results, model_names)
            coverage = model_coverage(ds, ocr_columns, judged_sample, existing_results)
            console.print()
            print_leaderboard(
                board,
                failed_models,
                failed_outputs,
                size_tie_ratio=args.size_tie_ratio,
                size_tie_min_samples=args.size_tie_min_samples,
                coverage=coverage,
            )
            if results_repo:
                metadata = EvalMetadata(
                    source_dataset=args.dataset,
                    source_split=args.split,
                    source_fingerprint=source_fingerprint,
                    source_column_fingerprints=source_column_fingerprints,
                    source_configs=source_configs,
                    source_revisions=source_revisions,
                    source_columns=ocr_columns,
                    source_output_columns=source_output_columns,
                    judge_models=judge_names,
                    judge_spec_hashes=judge_spec_hashes,
                    seed=args.seed,
                    max_samples=args.max_samples or len(ds),
                    total_comparisons=0,
                    valid_comparisons=0,
                    max_comparisons=max_comparisons,
                    budget_exhausted=budget_exhausted,
                    from_prs=from_prs,
                    failed_outputs=failed_outputs,
                    failed_models=failed_models,
                    model_coverage={m: c.to_dict() for m, c in coverage.items()},
                    criteria=criteria,
                    prompt_hash=criteria_prompt_hash,
                    max_ocr_text_len=args.max_ocr_text_len,
                    judge_image_dim=args.judge_image_dim,
                    judge_text_mode=args.judge_text_mode,
                    min_chars=args.min_chars,
                    adaptive_strategy=adaptive_strategy,
                    adaptive=adaptive,
                    ocr_bench_version=__version__,
                    size_tie_ratio=args.size_tie_ratio,
                    size_tie_min_samples=args.size_tie_min_samples,
                )
                publish_results(
                    results_repo,
                    board,
                    metadata,
                    existing_metadata=existing_meta_rows,
                    license_id=args.license,
                    preserved_comparisons=preserved_out_of_grid_results,
                )
                console.print(f"\nResults published to [bold]{results_repo}[/bold]")
                _refresh_viewer_space(results_repo)
            return

        if is_jury:
            console.print(f"\nJury mode: {len(judges)} judges")

        for judge in judges:
            console.print(f"\nRunning judge: {judge.name}")

        # Judge in chunks of checkpoint_every so checkpoints can fire mid-run;
        # a single _judge_batch over everything would only checkpoint at the end.
        # checkpoint_every == 0 → one chunk (no checkpointing).
        chunk = checkpoint_every if checkpoint_every else len(comparisons)
        new_results = []
        total_comparisons = 0  # judge calls made (auto-ties excluded)
        last_checkpoint = 0
        for start in range(0, len(comparisons), chunk):
            sub = comparisons[start : start + chunk]
            new_results.extend(_judge_batch(sub))
            # Auto-ties cost no judge call, so they neither advance the budget
            # nor trigger checkpoints; the pushed results still include them.
            total_comparisons += sum(c.auto_result is None for c in sub)
            if (
                checkpoint_every
                and results_repo
                and total_comparisons - last_checkpoint >= checkpoint_every
            ):
                _checkpoint(
                    results_repo,
                    existing_results + new_results + preserved_out_of_grid_results,
                    publication_model_names,
                )
                last_checkpoint = total_comparisons

        if budget_exhausted:
            console.print(
                f"\n[yellow]Budget reached ({total_comparisons} comparisons) — "
                f"publishing what was judged.[/yellow]"
            )
        judged_valid = len(new_results) - n_auto_total
        console.print(f"\n{judged_valid}/{total_comparisons} valid comparisons")
        if n_auto_total:
            console.print(f"[dim]{n_auto_total} identical outputs auto-tied (no judge call)[/dim]")

    # No new verdicts and nothing already on file → don't publish an empty
    # (all-1500 ELO) board. The standard path guards before judging; this also
    # catches the adaptive path when every batch was filtered out (e.g. an
    # aggressive --min-chars on a blank-heavy dataset) or a budget so small
    # everything was trimmed. Budget-exhausted WITH results still publishes.
    if not new_results and not existing_results:
        console.print("[yellow]No valid comparisons — check that OCR columns have text.[/yellow]")
        return

    # A run that lands EXACTLY on the budget (e.g. the final batch fills it, or a
    # non-adaptive build produces exactly N comparisons) never trips the mid-loop
    # trim/break, so catch the exhausted state here for both paths.
    if max_comparisons and total_comparisons >= max_comparisons:
        budget_exhausted = True

    # --- Merge existing + new, compute ELO ---
    all_results = existing_results + new_results
    board = compute_elo(all_results, model_names)
    coverage = model_coverage(ds, ocr_columns, judged_sample, all_results)
    console.print()
    print_leaderboard(
        board,
        failed_models,
        failed_outputs,
        size_tie_ratio=args.size_tie_ratio,
        size_tie_min_samples=args.size_tie_min_samples,
        coverage=coverage,
    )

    # A budget-capped run may stop before the ranking is statistically settled —
    # surface which adjacent pairs are still unresolved so the operator knows
    # what a top-up run should target.
    if budget_exhausted:
        unresolved = _unresolved_adjacent_pairs(
            board,
            size_tie_ratio=args.size_tie_ratio,
            size_tie_min_samples=args.size_tie_min_samples,
        )
        logger.warning(
            "budget_exhausted",
            limit=max_comparisons,
            judged=total_comparisons,
            unresolved_pairs=unresolved,
        )
        if unresolved:
            console.print("[yellow]Unresolved rankings at budget stop:[/yellow]")
            for pair in unresolved:
                console.print(f"  [yellow]{pair}[/yellow]")

    # --- Publish ---
    if results_repo:
        metadata = EvalMetadata(
            source_dataset=args.dataset,
            source_split=args.split,
            source_fingerprint=source_fingerprint,
            source_column_fingerprints=source_column_fingerprints,
            source_configs=source_configs,
            source_revisions=source_revisions,
            source_columns=ocr_columns,
            source_output_columns=source_output_columns,
            judge_models=judge_names,
            judge_spec_hashes=judge_spec_hashes,
            seed=args.seed,
            max_samples=args.max_samples or len(ds),
            total_comparisons=total_comparisons,
            valid_comparisons=len(new_results) - n_auto_total,
            max_comparisons=max_comparisons,
            budget_exhausted=budget_exhausted,
            auto_tied=n_auto_total,
            from_prs=from_prs,
            failed_outputs=failed_outputs,
            failed_models=failed_models,
            model_coverage={m: c.to_dict() for m, c in coverage.items()},
            criteria=criteria,
            prompt_hash=criteria_prompt_hash,
            max_ocr_text_len=args.max_ocr_text_len,
            judge_image_dim=args.judge_image_dim,
            judge_text_mode=args.judge_text_mode,
            min_chars=args.min_chars,
            adaptive_strategy=adaptive_strategy,
            adaptive=adaptive,
            ocr_bench_version=__version__,
            size_tie_ratio=args.size_tie_ratio,
            size_tie_min_samples=args.size_tie_min_samples,
        )
        publish_results(
            results_repo,
            board,
            metadata,
            existing_metadata=existing_meta_rows,
            license_id=args.license,
            preserved_comparisons=preserved_out_of_grid_results,
        )
        console.print(f"\nResults published to [bold]{results_repo}[/bold]")
        _refresh_viewer_space(results_repo)


def _print_job_summary(jobs: list[JobRun]) -> list[JobRun]:
    """Print a per-job status line after a run (failures in red).

    Returns the jobs that did not complete, so callers can decide what to do.
    """
    from ocr_bench.run import failed_jobs

    for job in jobs:
        if job.status == "completed":
            console.print(f"  [green]✓[/green] {job.model_slug}: {job.status}")
        else:
            console.print(f"  [red]✗ {job.model_slug}: {job.status}[/red]")
    failed = failed_jobs(jobs)
    if failed:
        console.print(
            f"\n[bold red]{len(failed)} of {len(jobs)} job(s) did not complete.[/bold red]"
        )
    else:
        console.print(f"\n[bold green]All {len(jobs)} job(s) completed.[/bold green]")
    return failed


def cmd_run(args: argparse.Namespace) -> list[JobRun]:
    """Launch OCR models on a dataset via HF Jobs. Returns the launched jobs."""
    from ocr_bench.run import (
        DEFAULT_MODELS,
        MODEL_REGISTRY,
        build_script_args,
        launch_ocr_jobs,
        poll_jobs,
        resolve_launch_settings,
    )

    # --list-models
    if args.list_models:
        table = Table(title="Available OCR Models", show_lines=True)
        table.add_column("Slug", style="cyan bold")
        table.add_column("Model ID")
        table.add_column("Size", justify="right")
        table.add_column("Default GPU", justify="center")

        for slug in sorted(MODEL_REGISTRY):
            cfg = MODEL_REGISTRY[slug]
            default = " (default)" if slug in DEFAULT_MODELS else ""
            gpu = cfg.default_flavor + (" (image-mode)" if cfg.image else "")
            table.add_row(slug + default, cfg.model_id, cfg.size, gpu)

        console.print(table)
        console.print(f"\nDefault set: {', '.join(DEFAULT_MODELS)}")
        return []

    if not args.input_dataset or not args.output_repo:
        console.print(
            "[red]Error:[/red] run requires INPUT_DATASET and OUTPUT_REPO "
            "unless --list-models is used."
        )
        raise SystemExit(2)

    selected = args.models or DEFAULT_MODELS
    for slug in selected:
        if slug not in MODEL_REGISTRY:
            console.print(f"[red]Unknown model: {slug}[/red]")
            console.print(f"Available: {', '.join(MODEL_REGISTRY.keys())}")
            sys.exit(1)

    console.print("\n[bold]OCR Benchmark Run[/bold]")
    console.print(f"  Source:  {args.input_dataset}")
    console.print(f"  Output:  {args.output_repo}")
    console.print(f"  Models:  {', '.join(selected)}")
    if args.max_samples:
        console.print(f"  Samples: {args.max_samples} per model")
    console.print()

    # Dry run
    if args.dry_run:
        console.print("[bold yellow]DRY RUN[/bold yellow] — no jobs will be launched\n")
        for slug in selected:
            cfg = MODEL_REGISTRY[slug]
            settings = resolve_launch_settings(cfg, flavor_override=args.flavor)
            script_args = build_script_args(
                args.input_dataset,
                args.output_repo,
                slug,
                split=args.split,
                max_samples=args.max_samples,
                shuffle=args.shuffle,
                seed=args.seed,
                extra_args=cfg.default_args or None,
            )
            console.print(f"[cyan]{slug}[/cyan] ({cfg.model_id})")
            console.print(f"  Flavor:  {settings['flavor']}")
            console.print(f"  Timeout: {args.timeout}")
            if "image" in settings:
                source = "registry" if cfg.image else "script header"
                console.print(f"  Image:   {settings['image']} ({source})")
            if "python" in settings:
                console.print(f"  Python:  {settings['python']}")
            if "env" in settings:
                console.print(f"  Env:     {settings['env']}")
            console.print(f"  Script:  {cfg.script}")
            console.print(f"  Args:    {' '.join(script_args)}")
            console.print()
        console.print("Remove --dry-run to launch these jobs.")
        return []

    # The recipes push their outputs with push_to_hub, which would create a
    # missing output repo with the account's default visibility (usually
    # public), so create or check it before launching.
    _check_visibility_flags(args)
    _prepare_destination(
        args.output_repo,
        private=args.private,
        allow_public=args.public,
        source_repo=args.input_dataset,
        label="output",
    )

    # Launch
    jobs = launch_ocr_jobs(
        args.input_dataset,
        args.output_repo,
        models=selected,
        max_samples=args.max_samples,
        split=args.split,
        shuffle=args.shuffle,
        seed=args.seed,
        flavor_override=args.flavor,
        timeout=args.timeout,
    )

    console.print(f"\n[green]{len(jobs)} jobs launched.[/green]")
    for job in jobs:
        console.print(f"  [cyan]{job.model_slug}[/cyan]: {job.job_url}")

    if not args.no_wait:
        console.print("\n[bold]Waiting for jobs to complete...[/bold]")
        poll_jobs(jobs)
        console.print("\n[bold]Job results:[/bold]")
        failed = _print_job_summary(jobs)
        if not failed:
            console.print("\nEvaluate:")
            console.print(f"  ocr-bench judge {args.output_repo}")
    else:
        console.print("\nJobs running in background.")
        console.print("Check status at: https://huggingface.co/settings/jobs")
        console.print(f"When complete: ocr-bench judge {args.output_repo}")

    return jobs


def cmd_view(args: argparse.Namespace) -> None:
    """Launch the FastAPI + HTMX results viewer."""
    try:
        import uvicorn

        from ocr_bench.web import create_app
    except ImportError:
        console.print(
            "[red]Error:[/red] FastAPI/uvicorn not installed. "
            "Install the viewer extra: [bold]pip install ocr-bench\\[viewer][/bold]"
        )
        sys.exit(1)

    console.print(f"Loading results from [bold]{args.results}[/bold]...")
    app = create_app(args.results, output_path=args.output)
    console.print(f"Starting viewer at [bold]http://{args.host}:{args.port}[/bold]")
    uvicorn.run(app, host=args.host, port=args.port)


def _print_share_board(title: str, rows: list[dict]) -> None:
    table = Table(title=title)
    for column in ("Rank", "Model", "ELO", "95% CI", "Pages", "Status"):
        table.add_column(column)
    for row in rows:
        ci = f"{row['elo_low']}–{row['elo_high']}" if row.get("elo_low") is not None else "—"
        table.add_row(
            str(row["rank"] or "—"),
            row["model"],
            str(row["elo"] if row["elo"] is not None else "—"),
            ci,
            str(row["pages"]),
            row.get("status", ""),
        )
    console.print(table)


def cmd_share(args: argparse.Namespace) -> None:
    """Write a scores-only board from a results repo to a separate repo."""
    from ocr_bench.share import (
        hub_model_sizes,
        load_private_evaluation,
        plan_share,
        push_share,
        write_local,
    )

    console.print(f"Reading [bold]{args.results}[/bold]...")
    evaluation = load_private_evaluation(args.results)
    plan = plan_share(
        evaluation,
        results_repo=args.results,
        alias=args.alias,
        slice_column=args.slice_column,
        board_repo=args.dest,
        model_sizes=hub_model_sizes,
    )
    for private in (args.results, plan.source_dataset):
        if (
            private
            and private.lower() in args.dest.lower()
            and private.lower() != args.dest.lower()
        ):
            console.print(
                f"[yellow]Note:[/yellow] the board repo name contains {private}, and repo names "
                "are public with the board."
            )
    _print_share_board(f"{plan.alias}: overall ({plan.settings['pages']} pages)", plan.overall_rows)
    for name in plan.settings["slices"]:
        _print_share_board(
            f"{plan.alias}: {name}", [row for row in plan.slice_rows if row["slice"] == name]
        )
    if args.slice_column:
        console.print(
            f"{len(plan.settings['slices'])} slice boards; "
            f"{plan.settings['pages_not_in_a_slice_board']} judged pages are in none."
        )
    console.print(
        "[green]Privacy checks passed[/green]: allowed columns and values only, and no "
        "copied fragment of 63 or more characters from a compared output or judge "
        "reason (text with little variety, such as a repeated character, is not "
        "checked). The alias, slice labels and model names are published as they "
        "are: check them."
    )

    if args.dry_run:
        slug = "".join(c if c.isalnum() or c in "._-" else "-" for c in plan.alias).strip(".")
        out_dir = Path(args.output_dir or f"ocr-bench-share-{slug or 'board'}")
        written = write_local(plan, out_dir)
        console.print(
            f"Dry run: wrote {len(written)} files to [bold]{out_dir}[/bold]. Nothing was pushed."
        )
        return

    result = push_share(
        plan, args.dest, results_repo=args.results, public=args.public, replace=args.replace
    )
    state = "created" if result.created else "updated"
    visibility = "private" if result.private else "public"
    console.print(f"Board {state}: https://huggingface.co/datasets/{result.repo_id} ({visibility})")
    if result.private:
        # The settings page works for every client version. The hf command needs
        # huggingface_hub 2.x: 1.x has no --public on `hf repos settings`.
        console.print(
            "The board is private. Look at it on the Hub, then publish it under "
            "Settings → Change visibility:\n"
            f"  https://huggingface.co/datasets/{result.repo_id}/settings\n"
            "or, with the hf CLI 2.0 or later:\n"
            f"  hf repos settings {result.repo_id} --repo-type dataset --public"
        )


SPACE_TEMPLATE = "davanstrien/ocr-bench-space-template"


def cmd_publish(args: argparse.Namespace) -> None:
    """Deploy results viewer as a Hugging Face Space."""
    from huggingface_hub import HfApi, SpaceHardware

    api = HfApi()
    results = args.results
    space_id = args.space or f"{results}-viewer"

    console.print(f"Deploying viewer for [bold]{results}[/bold] to [bold]{space_id}[/bold]...")

    # A viewer for private results defaults to a private Space. An existing
    # Space keeps its visibility (duplicate_space with exist_ok leaves it as is).
    private = args.private
    results_private = api.repo_info(results, repo_type="dataset").private is True
    if results_private and not private:
        private = True
        console.print(f"[bold]{results}[/bold] is private, so a new Space will be private too.")
    if results_private:
        # The template reads REPOS with load_dataset; without a token it cannot
        # see a private repo. Never copy the user's token into a Space silently.
        console.print(
            f"[yellow]The viewer needs read access to {results}:[/yellow] add an "
            f"HF_TOKEN secret with read access in the Space settings of {space_id}."
        )

    api.duplicate_space(
        from_id=SPACE_TEMPLATE,
        to_id=space_id,
        private=True if private else None,
        hardware=SpaceHardware.CPU_BASIC,
        exist_ok=True,
        variables=[{"key": "REPOS", "value": results}],
    )

    api.add_space_variable(repo_id=space_id, key="REPOS", value=results)

    # Resolve the source dataset from the results repo's metadata config so the
    # deployed Space card cross-links to both the results and the source data
    # (issue #38). A results repo always exists at this point; source may not
    # resolve (older repos, private, transient error) — filter it out if so.
    source_dataset = None
    meta_rows = load_existing_metadata(results)
    if meta_rows:
        source_dataset = meta_rows[-1].get("source_dataset") or None

    # De-dup while preserving order; drop Nones.
    datasets_links: list[str] = []
    for repo in (results, source_dataset):
        if repo and repo not in datasets_links:
            datasets_links.append(repo)

    title = f"OCR Bench — {results.split('/')[-1]}"

    # Update Space metadata to cross-link the results (and source) datasets.
    try:
        from huggingface_hub import metadata_update

        metadata_update(
            space_id,
            {"datasets": datasets_links, "tags": ["ocr-bench"], "title": title},
            repo_type="space",
            overwrite=True,
        )
    except Exception as exc:
        logger.warning("space_metadata_update_failed", error=str(exc))

    url = f"https://huggingface.co/spaces/{space_id}"
    console.print(f"[green]Space published![/green] {url}")


def cmd_bench(args: argparse.Namespace) -> None:
    """One command: run OCR models, judge them, then open the viewer.

    Chains ``run`` → ``judge`` → ``view``, threading the shared flags through
    each phase. Sub-namespaces are built through :func:`build_parser` so the
    per-subcommand defaults live in one place. The individual subcommands give
    finer control over each stage.
    """
    parser = build_parser()

    # Validate the criteria selection BEFORE launching OCR jobs: a bad
    # --criteria-file or a --criteria/--criteria-file conflict should fail fast,
    # not after a full (paid) run. Raises DatasetError, caught by main().
    _resolve_criteria(args)

    # Prepare both destinations BEFORE launching OCR jobs, in the same chain
    # the phases use (input -> output repo -> results repo), so a results repo
    # that judge would refuse stops the run here, not after a full (paid) OCR
    # run. The run and judge phases repeat these checks; both are idempotent.
    _check_visibility_flags(args)
    _prepare_destination(
        args.output_repo,
        private=args.private,
        allow_public=args.public,
        source_repo=args.input_dataset,
        label="output",
    )
    if not args.no_publish:
        results_repo = _resolve_results_repo(args.output_repo, None, False)
        assert results_repo is not None
        _prepare_destination(
            results_repo,
            private=args.private,
            allow_public=args.public,
            source_repo=args.output_repo,
            label="results",
        )

    # --- Phase 1: run OCR models (waits for jobs to finish by default) ---
    from ocr_bench.run import failed_jobs

    run_argv = ["run", args.input_dataset, args.output_repo, "--seed", str(args.seed)]
    if args.models:
        run_argv += ["--models", *args.models]
    if args.max_samples is not None:
        run_argv += ["--max-samples", str(args.max_samples)]
    if args.private:
        run_argv.append("--private")
    if args.public:
        run_argv.append("--public")
    console.rule("[bold]1/3 Run[/bold]")
    jobs = cmd_run(parser.parse_args(run_argv)) or []

    # Abort before judging if any model failed — a silently incomplete
    # leaderboard is the worst outcome for the "just try it" path.
    failed = failed_jobs(jobs)
    if failed:
        slugs = ", ".join(j.model_slug for j in failed)
        console.print(
            f"\n[bold red]Aborting: {len(failed)} of {len(jobs)} OCR job(s) did not "
            f"complete ({slugs}).[/bold red] A partial run would produce a "
            "misleading leaderboard, so bench stops here."
        )
        retry = " ".join(j.model_slug for j in failed)
        console.print(
            "\nRe-run just the failed models:\n"
            f"  ocr-bench run {args.input_dataset} {args.output_repo} --models {retry}\n"
            "or judge the models that did succeed:\n"
            f"  ocr-bench judge {args.output_repo} --from-prs"
        )
        return

    # --- Phase 2: judge the OCR outputs (from the PRs the run just opened) ---
    judge_argv = [
        "judge",
        args.output_repo,
        "--from-prs",
        "--seed",
        str(args.seed),
        "--judge-text-mode",
        args.judge_text_mode,
        "--adaptive-strategy",
        args.adaptive_strategy,
        "--size-tie-min-samples",
        str(args.size_tie_min_samples),
        "--concurrency",
        str(args.concurrency),
    ]
    if args.size_tie_ratio is not None:
        judge_argv += ["--size-tie-ratio", str(args.size_tie_ratio)]
    for model in args.judge_models or []:
        judge_argv += ["--model", model]
    # Forward whichever criteria flag was set (at most one — already validated
    # above). Neither → the judge phase uses its own default.
    if args.criteria_file:
        judge_argv += ["--criteria-file", args.criteria_file]
    elif args.criteria is not None:
        judge_argv += ["--criteria", args.criteria]
    if args.max_samples is not None:
        judge_argv += ["--max-samples", str(args.max_samples)]
    if args.no_publish:
        judge_argv.append("--no-publish")
    elif args.private:
        judge_argv.append("--private")
    elif args.public:
        judge_argv.append("--public")
    console.rule("[bold]2/3 Judge[/bold]")
    cmd_judge(parser.parse_args(judge_argv))

    # --- Phase 3: view the results ---
    if args.no_publish:
        console.print(
            "\n[yellow]--no-publish set: nothing was published, so there is "
            "nothing to view. Skipping the viewer.[/yellow]"
        )
        return
    results_repo = _resolve_results_repo(args.output_repo, None, False)
    assert results_repo is not None  # no_publish is False past the guard above
    view_argv = ["view", results_repo, "--port", str(args.port), "--host", args.host]
    console.rule("[bold]3/3 View[/bold]")
    cmd_view(parser.parse_args(view_argv))


# Audit exit codes, so CI can tell a bad *repo* from a broken *run*.
_AUDIT_EXIT_INTEGRITY = 1  # audit ran and found blocking problems
_AUDIT_EXIT_OPERATIONAL = 2  # audit could not complete (network/Hub/load)
# Screen counts above this rate are shown in red, other non-zero counts in yellow.
SCREEN_WARN_RATE = 0.10
# Rates on fewer pages than this are too noisy to colour.
SCREEN_MIN_COLOUR_PAGES = 10


def _pct(rate: float) -> str:
    """Format a rate as a percentage, colouring anything non-zero for attention."""
    if rate <= 0:
        return "0%"
    colour = "red" if rate > SENTINEL_FLAG_RATE else "yellow"
    return f"[{colour}]{rate:.0%}[/{colour}]"


def _align_cell(status: str) -> str:
    """Colour a per-config alignment status for the audit table."""
    colours = {"misaligned": "red", "unverified": "yellow", "ok": "green"}
    colour = colours.get(status)
    return f"[{colour}]{status}[/{colour}]" if colour else status


def cmd_audit(args: argparse.Namespace) -> None:
    """Read-only pre-judge health check.

    Exit codes: 0 = clean, 1 = the repo would poison a judge run (integrity
    failure), 2 = the audit could not complete (network / Hub / dataset load).
    The screen checks never change the exit code.
    """
    # With --json, stdout holds only the JSON. Log lines (structlog prints to
    # stdout by default) and any other output during the audit go to stderr.
    output = contextlib.redirect_stdout(sys.stderr) if args.json else contextlib.nullcontext()
    try:
        with output:
            report = audit_repo(
                args.dataset,
                split=args.split,
                max_ocr_text_len=args.max_ocr_text_len,
                judge_text_mode=args.judge_text_mode,
                configs=args.configs,
                from_prs=args.from_prs,
                screen=args.screen,
                columns=args.columns,
            )
    except (DatasetError, DatasetsError, OSError, OpenAIError, ValueError) as exc:
        # Couldn't even run the check (repo missing, no OCR columns, network /
        # Hub outage, a config the datasets library cannot build). Distinct from
        # an integrity failure so automation can tell "broken run" from "bad data".
        _audit_error(args, f"Audit could not complete: {exc}")

    if not report.configs and not report.configs_without_text and not report.auxiliary_configs:
        _audit_error(args, "No OCR configs/columns found to audit.")

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
        if report.has_problems:
            sys.exit(_AUDIT_EXIT_INTEGRITY)
        return

    align = report.alignment
    table = Table(
        title=f"OCR input audit — {args.dataset} ({args.judge_text_mode} judge text)",
        show_lines=False,
    )
    table.add_column("Config", style="cyan")
    table.add_column("Model")
    table.add_column("Rows", justify="right")
    table.add_column("Empty", justify="right")
    table.add_column("<20ch", justify="right")
    table.add_column("Sentinel", justify="right")
    table.add_column("Median len", justify="right")
    table.add_column("Max len", justify="right")
    table.add_column(f">{args.max_ocr_text_len}", justify="right")
    table.add_column("Align")

    for cfg in report.configs:
        s = cfg.stats
        table.add_row(
            s.name,
            s.model,
            str(s.n_rows),
            _pct(s.empty_rate),
            _pct(s.short_rate),
            _pct(s.sentinel_rate),
            f"{s.median_len:.0f}",
            str(s.max_len),
            _pct(s.over_max_rate),
            _align_cell(align.config_status(s.name)),
        )
    for config in report.configs_without_text:
        table.add_row(config, "[red]no OCR text column[/red]", *["-"] * 8)
    for config in report.auxiliary_configs:
        table.add_row(config, "[dim]skipped: not OCR output[/dim]", *["-"] * 8)

    console.print(table)
    _print_passed_over_prs(
        args.dataset,
        report.duplicate_prs,
        report.superseded_prs,
        report.pr_revisions,
        merge=False,
    )

    # Overall alignment line
    if align.status == "misaligned":
        console.print(f"[red]Alignment: {align.describe()}[/red]")
    elif align.status in ("unverified", "partial"):
        console.print(f"[yellow]Alignment: {align.describe()}[/yellow]")
    else:
        console.print(f"Alignment: {align.describe()}")

    if report.screen is not None:
        _print_screen(report.screen)
    elif report.screen_skipped:
        console.print(f"\n[yellow]Screen skipped:[/yellow] {report.screen_skipped}.")

    # Verdict + exit code (usable in automation)
    if report.has_problems:
        console.print(f"\n[red]FAIL[/red] — {'; '.join(report.problems)}")
        sys.exit(_AUDIT_EXIT_INTEGRITY)
    console.print("\n[green]OK[/green] — no blocking issues found")


def _audit_error(args: argparse.Namespace, message: str) -> NoReturn:
    """Report an audit that could not complete, and exit with the operational code."""
    if args.json:
        print(json.dumps({"error": message}))
    else:
        console.print(f"[red]{message}[/red]")
    sys.exit(_AUDIT_EXIT_OPERATIONAL)


def _count_cell(count: int, total: int) -> str:
    """Format 'count/total'; colour it only when total is large enough to judge a rate."""
    if total == 0:
        return "-"
    cell = f"{count}/{total}"
    if count == 0 or total < SCREEN_MIN_COLOUR_PAGES:
        return cell
    colour = "red" if count / total > SCREEN_WARN_RATE else "yellow"
    return f"[{colour}]{cell}[/{colour}]"


def _print_screen(screen: ScreenReport) -> None:
    """Print the screen table and the text-cap line."""
    table = Table(title=f"Screen — {screen.rows} rows, no judge", show_lines=False)
    table.add_column("Config", style="cyan")
    table.add_column("Model")
    table.add_column("Pages", justify="right")
    table.add_column("Loops", justify="right")
    table.add_column(">5x length", justify="right")
    table.add_column("Text on blank", justify="right")
    table.add_column("Empty on text", justify="right")
    table.add_column("Peer agreement", justify="right")
    for m in screen.models:
        peer = "-" if m.peer_agreement is None else f"{m.peer_agreement:.2f}"
        table.add_row(
            m.name,
            m.model,
            str(m.pages),
            _count_cell(len(m.loop_rows), m.pages),
            _count_cell(len(m.long_rows), m.checked_pages),
            _count_cell(len(m.text_on_blank_rows), m.blank_pages),
            _count_cell(len(m.empty_on_text_rows), m.text_pages),
            peer,
        )
    console.print()
    console.print(table)
    console.print(
        "[dim]'Pages' leaves out error placeholders. A blank page is one where most "
        "outputs have under 20 characters; a text page is one where most have 200 or "
        "more. Peer agreement is the mean, over pages, of the median similarity to the "
        "other models' outputs, with loops left out. It is not a ranking. Counts on "
        f"fewer than {SCREEN_MIN_COLOUR_PAGES} pages are not coloured. These checks are "
        "leads, not verdicts: look at the flagged rows (--json lists them) before you "
        "drop a model.[/dim]"
    )
    console.print(
        f"Outputs that no check flagged: {screen.unflagged_outputs}; judge-input length "
        f"median {screen.unflagged_median}, 95th percentile {screen.unflagged_p95}, "
        f"longest {screen.longest_unflagged} characters."
    )
    suggested = screen.suggested_max_ocr_text_len
    if suggested is None:
        console.print(f"Text cap: --max-ocr-text-len {screen.max_ocr_text_len} fits all of them.")
    else:
        console.print(
            f"[yellow]Text cap:[/yellow] --max-ocr-text-len {screen.max_ocr_text_len} "
            f"truncates {screen.unflagged_over_cap} of them. --max-ocr-text-len "
            f"{suggested} fits them all; a higher cap makes each judge call longer."
        )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    try:
        if args.command == "judge":
            cmd_judge(args)
        elif args.command == "run":
            cmd_run(args)
        elif args.command == "view":
            cmd_view(args)
        elif args.command == "publish":
            cmd_publish(args)
        elif args.command == "bench":
            cmd_bench(args)
        elif args.command == "audit":
            cmd_audit(args)
        elif args.command == "share":
            cmd_share(args)
    except DatasetError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)
    except (OpenAIError, OSError) as exc:
        # Judge or Hub request failed (bad/expired token, unknown model id,
        # rate limit, provider/network outage) — every requests/HfHubHTTPError
        # subclasses OSError — or another OS-level error. Fail with a clean
        # message instead of dumping a traceback on the user.
        console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)
