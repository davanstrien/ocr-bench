"""CLI entrypoint for ocr-bench."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import structlog
from openai import OpenAIError
from rich.console import Console
from rich.table import Table

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
    load_config_dataset,
    load_flat_dataset,
)
from ocr_bench.elo import ComparisonResult, Leaderboard, compute_elo, rankings_resolved
from ocr_bench.integrity import (
    SENTINEL_FLAG_RATE,
    audit_repo,
    compute_model_stats,
    failed_output_counts,
)
from ocr_bench.judge import (
    CRITERIA_PROFILES,
    DEFAULT_CRITERIA,
    DEFAULT_MIN_CHARS,
    MAX_OCR_TEXT_LENGTH,
    Comparison,
    _normalize_pair,
    build_comparisons,
    is_sentinel,
    prompt_hash,
    sample_indices,
    validate_prompt_template,
)
from ocr_bench.publish import (
    EvalMetadata,
    load_existing_comparisons,
    load_existing_metadata,
    publish_checkpoint,
    publish_results,
)

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
    judge.add_argument("--columns", nargs="+", default=None, help="Explicit OCR column names")
    judge.add_argument(
        "--configs", nargs="+", default=None, help="Config-per-model: list of config names"
    )
    judge.add_argument("--from-prs", action="store_true", help="Force PR-based config discovery")
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
            "Mutually exclusive with --criteria-file."
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
        "--concurrency",
        type=int,
        default=1,
        help="Number of concurrent judge API calls (default: 1)",
    )

    # --- run subcommand ---
    run = sub.add_parser("run", help="Launch OCR models on a dataset via HF Jobs")
    run.add_argument("input_dataset", help="HF dataset repo id with images")
    run.add_argument("output_repo", help="Output dataset repo (all models push here)")
    run.add_argument(
        "--models", nargs="+", default=None, help="Model slugs to run (default: all 4 core)"
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
    publish.add_argument(
        "--space", default=None, help="Space repo id (default: {results}-viewer)"
    )
    publish.add_argument("--private", action="store_true", help="Make the Space private")

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
            "Mutually exclusive with --criteria-file."
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
        "--no-publish", action="store_true", help="Don't publish results (skips the viewer)"
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
        "--max-ocr-text-len",
        type=int,
        default=MAX_OCR_TEXT_LENGTH,
        help=(
            "Text length above which the judge truncates an output "
            f"(default: {MAX_OCR_TEXT_LENGTH}); reported as truncation exposure"
        ),
    )

    return parser


def print_leaderboard(
    board: Leaderboard,
    failed_models: list[str] | None = None,
    failed_outputs: dict[str, int] | None = None,
) -> None:
    """Print leaderboard as a Rich table, leaving failed runs unranked."""
    from ocr_bench.publish import _get_model_sizes

    sizes = _get_model_sizes()
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
        model_label = f"{model} ⚠" if (failed_outputs or {}).get(model) else model
        table.add_row(
            str(rank),
            model_label,
            sizes.get(model, ""),
            elo_str,
            str(board.wins[model]),
            str(board.losses[model]),
            str(board.ties[model]),
            pct_str,
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
        )

    console.print(table)


def _merge_auto_ties(
    comparisons: list[Comparison], judged: list[dict]
) -> list[dict]:
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
    """Cap judged pairs at ``judge_budget`` while keeping every auto-tie.

    Auto-ties cost no judge call, so they never count against the budget and
    are always retained (they are real results that must still reach the
    leaderboard). Returns the trimmed list and whether any judged pair was
    dropped to fit the budget.
    """
    judgeable = sum(c.auto_result is None for c in comparisons)
    if judgeable <= judge_budget:
        return comparisons, False
    kept: list[Comparison] = []
    remaining = judge_budget
    for comp in comparisons:
        if comp.auto_result is not None:
            kept.append(comp)
        elif remaining > 0:
            kept.append(comp)
            remaining -= 1
    return kept, True


def _convert_results(
    comparisons: list[Comparison], aggregated: list[dict]
) -> list[ComparisonResult]:
    """Convert judged comparisons + aggregated outputs into ComparisonResult list."""
    results: list[ComparisonResult] = []
    for comp, result in zip(comparisons, aggregated):
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
                text_a=comp.text_a,
                text_b=comp.text_b,
                col_a=comp.col_a,
                col_b=comp.col_b,
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


def _filter_existing_sentinel_comparisons(
    results: list[ComparisonResult],
) -> tuple[list[ComparisonResult], int]:
    """Remove historical comparisons where an error sentinel competed as OCR.

    Results repos created before issue #46 was fixed can contain verdicts for
    strings such as ``[OCR ERROR]``. Reusing those rows would preserve the
    poisoned ELO and add their pair/sample keys to the resume skip map, so the
    corrected comparison builder would never get a chance to reconsider them.
    Filter them before either operation and report how many were discarded.
    """
    kept = [r for r in results if not (is_sentinel(r.text_a) or is_sentinel(r.text_b))]
    return kept, len(results) - len(kept)


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


def _unresolved_adjacent_pairs(board: Leaderboard) -> list[str]:
    """Adjacent-rank model pairs whose 95% CIs still overlap.

    Used to report what a budget-capped run left statistically unresolved.
    Empty when CIs are unavailable or every adjacent pair is separated.
    """
    if not board.elo_ci:
        return []
    ranked = board.ranked
    pairs: list[str] = []
    for i in range(len(ranked) - 1):
        hi_model, _ = ranked[i]
        lo_model, _ = ranked[i + 1]
        hi_ci = board.elo_ci.get(hi_model)
        lo_ci = board.elo_ci.get(lo_model)
        if hi_ci and lo_ci and lo_ci[1] >= hi_ci[0]:
            pairs.append(f"{hi_model} vs {lo_model}")
    return pairs


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
        raise DatasetError(
            "--criteria and --criteria-file are mutually exclusive; pass only one"
        )
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


def _existing_criteria_provenance(meta_rows: list[dict]) -> tuple[str, str]:
    """The (criteria profile, prompt hash) the existing comparisons were judged under.

    Reads the LAST metadata row. Pre-#44 rows lack these columns (or carry None
    after schema alignment); treat them as the ``default`` profile — historically
    accurate, since the default profile's prompt is byte-identical to the
    pre-#44 hardcoded prompt. Used to block mixing criteria rubrics on one board.
    """
    default_hash = prompt_hash(CRITERIA_PROFILES[DEFAULT_CRITERIA])
    if not meta_rows:
        return DEFAULT_CRITERIA, default_hash
    last = meta_rows[-1]
    return (last.get("criteria") or DEFAULT_CRITERIA, last.get("prompt_hash") or default_hash)


def cmd_judge(args: argparse.Namespace) -> None:
    """Orchestrate: load → compare → judge → elo → print → publish."""
    # --- Resolve flags ---
    adaptive = not args.no_adaptive
    merge = args.merge
    results_repo = _resolve_results_repo(args.dataset, args.save_results, args.no_publish)
    from_prs = False  # track for metadata
    max_comparisons = args.max_comparisons  # global budget; None = uncapped

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

    if results_repo:
        console.print(f"Results will be published to [bold]{results_repo}[/bold]")

    # --- Load dataset (cascading auto-detection) ---
    if args.configs:
        # Explicit configs — use them directly
        config_names = args.configs
        ds, ocr_columns = load_config_dataset(args.dataset, config_names, split=args.split)
    elif args.columns:
        # Explicit columns — flat loading
        ds, ocr_columns = load_flat_dataset(args.dataset, split=args.split, columns=args.columns)
    elif args.from_prs:
        # Forced PR discovery
        config_names, pr_revisions = discover_pr_configs(args.dataset, merge=merge)
        if not config_names:
            raise DatasetError("No configs found in open PRs")
        from_prs = True
        console.print(f"Discovered {len(config_names)} configs from PRs: {config_names}")
        ds, ocr_columns = load_config_dataset(
            args.dataset,
            config_names,
            split=args.split,
            pr_revisions=pr_revisions if not merge else None,
        )
    else:
        # Auto-detect: PRs + main branch configs combined, fall back to flat
        pr_configs, pr_revisions = discover_pr_configs(args.dataset, merge=merge)
        main_configs = discover_configs(args.dataset)

        # Combine: PR configs + main configs not already in PRs
        config_names = list(pr_configs)
        for mc in main_configs:
            if mc not in pr_configs:
                config_names.append(mc)

        if config_names:
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
            )
        else:
            # No configs anywhere — fall back to flat loading
            ds, ocr_columns = load_flat_dataset(args.dataset, split=args.split)

    console.print(f"Loaded {len(ds)} samples with {len(ocr_columns)} models:")
    for col, model in ocr_columns.items():
        console.print(f"  {col} → {model}")

    # --- Input integrity: sentinel outputs (issue #46) ---
    # Error sentinels (e.g. "[OCR ERROR]") are excluded from judging by
    # build_comparisons; here we count them per model for the metadata + card
    # and warn loudly when a model's run largely failed on this corpus.
    model_stats = compute_model_stats(ds, ocr_columns, max_ocr_text_len=MAX_OCR_TEXT_LENGTH)
    failed_outputs = failed_output_counts(model_stats)
    # A fully sentinel-backed run has no comparable OCR output at all. Keep it
    # visible as FAILED, but never assign it an arbitrary disconnected ELO/rank.
    # Partial failures remain rankable on their successful outputs with a
    # degraded warning, while the audit still blocks rates over its threshold.
    failed_models = sorted(
        stat.model
        for stat in model_stats
        if stat.n_rows > 0 and stat.n_sentinel == stat.n_rows
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

    # --- Incremental: load existing comparisons ---
    existing_results: list[ComparisonResult] = []
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
        loaded_existing = load_existing_comparisons(results_repo)
        existing_results, discarded_sentinels = _filter_existing_sentinel_comparisons(
            loaded_existing
        )
        # A currently all-sentinel model is intentionally absent from the ELO
        # graph. Historical non-sentinel verdicts for that model must not reconnect
        # it or influence the remaining rankings.
        failed_set = set(failed_models)
        before_failed_filter = len(existing_results)
        existing_results = [
            result
            for result in existing_results
            if result.model_a not in failed_set and result.model_b not in failed_set
        ]
        discarded_failed_model = before_failed_filter - len(existing_results)

        # Preserve the append-only metadata history even when integrity filtering
        # removes every old comparison.
        existing_meta_rows = load_existing_metadata(results_repo)

        if discarded_sentinels:
            console.print(
                f"\n[bold yellow]Discarded {discarded_sentinels} existing comparison(s) "
                "containing OCR error sentinels.[/bold yellow] They will not affect "
                "ELO or the resume skip map."
            )
            logger.warning(
                "discarded_existing_sentinel_comparisons",
                repo=results_repo,
                n=discarded_sentinels,
            )
        if discarded_failed_model:
            console.print(
                f"[bold yellow]Discarded {discarded_failed_model} historical comparison(s) "
                "involving a currently failed model.[/bold yellow]"
            )
            logger.warning(
                "discarded_failed_model_comparisons",
                repo=results_repo,
                n=discarded_failed_model,
                models=failed_models,
            )

        if existing_results:
            # Provenance guard: NEVER mix criteria rubrics on one board. The
            # existing comparisons were judged under the profile recorded in the
            # last metadata row (pre-#44 rows = the default profile, whose prompt
            # is byte-identical to the old hardcoded one). Judging the rest — or
            # even just refitting — under a different --criteria would merge
            # incompatible verdicts into one ELO board AND republish the metadata
            # mislabeled with the current run's criteria. Refuse and exit; the
            # only safe way to change rubric is --full-rejudge (discards existing).
            # Compare on the prompt HASH (the true rubric identity): this catches
            # a built-in prompt edited across code versions, matches the same
            # custom file re-run, and blocks two different custom files even if
            # they share a basename.
            prev_criteria, prev_hash = _existing_criteria_provenance(existing_meta_rows)
            if prev_hash != criteria_prompt_hash:
                if prev_criteria.startswith("custom:"):
                    match_hint = (
                        f"re-supply the same custom prompt file (recorded as "
                        f"'{prev_criteria}', prompt {prev_hash})"
                    )
                else:
                    match_hint = f"re-run with [bold]--criteria {prev_criteria}[/bold]"
                console.print(
                    f"[red]Error:[/red] criteria mismatch — [bold]{results_repo}[/bold] "
                    f"was judged under criteria '[bold]{prev_criteria}[/bold]' "
                    f"(prompt {prev_hash}), but this run requested "
                    f"'[bold]{criteria}[/bold]' (prompt {criteria_prompt_hash}). "
                    f"Mixing rubrics on one leaderboard would produce meaningless "
                    f"rankings and mislabeled metadata.\n"
                    f"To match the existing results, {match_hint}; or use "
                    f"[bold]--full-rejudge[/bold] to discard them and re-judge "
                    f"everything under '{criteria}'."
                )
                sys.exit(1)
            skip_samples = {}
            for r in existing_results:
                skip_samples.setdefault(_normalize_pair(r.model_a, r.model_b), set()).add(
                    r.sample_idx
                )
            console.print(
                f"\nIncremental mode: {len(existing_results)} reusable existing comparisons "
                f"across {len(skip_samples)} model pairs — skipping already-judged "
                f"(pair, sample) combinations, topping up the rest."
            )
        elif discarded_sentinels or discarded_failed_model:
            console.print(
                "\nNo reusable existing comparisons remain after integrity filtering — "
                "rebuilding from the current OCR outputs."
            )
        else:
            console.print("\nNo existing comparisons found — full judge run.")

    model_names = list(set(ocr_columns.values()) - set(failed_models))

    # --- Judge setup (shared by both paths) ---
    model_specs = args.models or [DEFAULT_JUDGE]
    judges = [
        parse_judge_spec(spec, max_tokens=args.max_tokens, concurrency=args.concurrency)
        for spec in model_specs
    ]
    is_jury = len(judges) > 1

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
        return _convert_results(batch_comps, merged)

    # Set when the run stops because it hit --max-comparisons (vs converging or
    # running out of samples). Recorded in the metadata row and drives the
    # unresolved-pairs report at the end.
    budget_exhausted = False

    if adaptive:
        # --- Adaptive stopping: batch-by-batch with convergence check ---
        from itertools import combinations as _combs

        all_indices = sample_indices(len(ds), args.max_samples, args.seed)
        n_pairs = len(list(_combs(model_names, 2)))
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

        new_results: list[ComparisonResult] = []
        total_comparisons = 0  # judge calls made this run (auto-ties excluded)
        last_checkpoint = 0
        n_auto_total = 0
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
                indices=batch_indices,
                seed=args.seed,
                min_chars=args.min_chars,
                prompt_template=prompt_template,
            )
            if not batch_comps:
                continue

            batch_auto = sum(c.auto_result is not None for c in batch_comps)

            # Trim the final batch to land exactly on the budget: cap judged
            # pairs at the remaining budget but keep auto-ties (they cost no
            # judge call), so the run stops cleanly without dropping real ties.
            if max_comparisons is not None:
                batch_comps, hit_budget = _trim_to_budget(
                    batch_comps, max_comparisons - total_comparisons
                )
                if hit_budget:
                    budget_exhausted = True

            n_auto_total += batch_auto
            batch_results = _judge_batch(batch_comps)
            new_results.extend(batch_results)
            total_comparisons += len(batch_comps) - batch_auto
            # batch_comps goes out of scope → GC can free images

            total = len(existing_results) + len(new_results)
            console.print(f"  Batch {batch_num + 1}: {len(batch_results)} new, {total} total")

            # Periodic checkpoint: push the full accumulated comparison set
            # (append-only, comparisons config only). Fires at batch boundaries,
            # so checkpoint_every is a lower bound on comparisons between pushes.
            if (
                checkpoint_every
                and results_repo
                and total_comparisons - last_checkpoint >= checkpoint_every
            ):
                _checkpoint(results_repo, existing_results + new_results, model_names)
                last_checkpoint = total_comparisons

            if budget_exhausted:
                break

            if total >= min_before_check:
                board = compute_elo(existing_results + new_results, model_names)
                # Show CI gaps for each adjacent pair
                ranked = board.ranked
                if board.elo_ci:
                    gaps: list[str] = []
                    for i in range(len(ranked) - 1):
                        hi_model, _ = ranked[i]
                        lo_model, _ = ranked[i + 1]
                        hi_ci = board.elo_ci.get(hi_model)
                        lo_ci = board.elo_ci.get(lo_model)
                        if hi_ci and lo_ci:
                            gap = hi_ci[0] - lo_ci[1]  # positive = resolved
                            if gap > 0:
                                status = "[green]ok[/green]"
                            else:
                                status = f"[yellow]overlap {-gap:.0f}[/yellow]"
                            gaps.append(f"    {hi_model} vs {lo_model}: gap={gap:+.0f} {status}")
                    if gaps:
                        console.print("  CI gaps:")
                        for g in gaps:
                            console.print(g)

                if rankings_resolved(board):
                    remaining = len(all_indices) - batch_start - len(batch_indices)
                    console.print(
                        f"[green]Rankings converged after {total} comparisons! "
                        f"Skipped ~{remaining * n_pairs} remaining.[/green]"
                    )
                    break

        if budget_exhausted:
            console.print(
                f"\n[yellow]Budget reached ({total_comparisons} comparisons) — "
                f"stopping and publishing.[/yellow]"
            )
        judged_valid = len(new_results) - n_auto_total
        console.print(f"\n{judged_valid}/{total_comparisons} valid comparisons")
        if n_auto_total:
            console.print(
                f"[dim]{n_auto_total} identical outputs auto-tied (no judge call)[/dim]"
            )
    else:
        # --- Standard single-pass flow ---
        comparisons = build_comparisons(
            ds,
            ocr_columns,
            max_samples=args.max_samples,
            seed=args.seed,
            skip_samples=skip_samples,
            min_chars=args.min_chars,
            prompt_template=prompt_template,
        )

        # Global budget: judge at most N pairs, keeping every auto-tie (they
        # cost no judge call). Which judged pairs get dropped depends on build
        # order — acceptable for a blunt operational cap.
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
            console.print()
            print_leaderboard(board, failed_models, failed_outputs)
            if results_repo:
                metadata = EvalMetadata(
                    source_dataset=args.dataset,
                    judge_models=[],
                    seed=args.seed,
                    max_samples=args.max_samples or len(ds),
                    total_comparisons=0,
                    valid_comparisons=0,
                    max_comparisons=max_comparisons,
                    budget_exhausted=budget_exhausted,
                    from_prs=from_prs,
                    failed_outputs=failed_outputs,
                    failed_models=failed_models,
                    criteria=criteria,
                    prompt_hash=criteria_prompt_hash,
                )
                publish_results(
                    results_repo,
                    board,
                    metadata,
                    existing_metadata=existing_meta_rows,
                    license_id=args.license,
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
                _checkpoint(results_repo, existing_results + new_results, model_names)
                last_checkpoint = total_comparisons

        if budget_exhausted:
            console.print(
                f"\n[yellow]Budget reached ({total_comparisons} comparisons) — "
                f"publishing what was judged.[/yellow]"
            )
        judged_valid = len(new_results) - n_auto_total
        console.print(f"\n{judged_valid}/{total_comparisons} valid comparisons")
        if n_auto_total:
            console.print(
                f"[dim]{n_auto_total} identical outputs auto-tied (no judge call)[/dim]"
            )

    # No new verdicts and nothing already on file → don't publish an empty
    # (all-1500 ELO) board. The standard path guards before judging; this also
    # catches the adaptive path when every batch was filtered out (e.g. an
    # aggressive --min-chars on a blank-heavy dataset) or a budget so small
    # everything was trimmed. Budget-exhausted WITH results still publishes.
    if not new_results and not existing_results:
        console.print(
            "[yellow]No valid comparisons — check that OCR columns have text.[/yellow]"
        )
        return

    # A run that lands EXACTLY on the budget (e.g. the final batch fills it, or a
    # non-adaptive build produces exactly N comparisons) never trips the mid-loop
    # trim/break, so catch the exhausted state here for both paths.
    if max_comparisons and total_comparisons >= max_comparisons:
        budget_exhausted = True

    # --- Merge existing + new, compute ELO ---
    all_results = existing_results + new_results
    board = compute_elo(all_results, model_names)
    console.print()
    print_leaderboard(board, failed_models, failed_outputs)

    # A budget-capped run may stop before the ranking is statistically settled —
    # surface which adjacent pairs are still unresolved so the operator knows
    # what a top-up run should target.
    if budget_exhausted:
        unresolved = _unresolved_adjacent_pairs(board)
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
            judge_models=[j.name for j in judges],
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
            criteria=criteria,
            prompt_hash=criteria_prompt_hash,
        )
        publish_results(
            results_repo,
            board,
            metadata,
            existing_metadata=existing_meta_rows,
            license_id=args.license,
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
            flavor = args.flavor or cfg.default_flavor
            script_args = build_script_args(
                args.input_dataset,
                args.output_repo,
                slug,
                max_samples=args.max_samples,
                shuffle=args.shuffle,
                seed=args.seed,
                extra_args=cfg.default_args or None,
            )
            console.print(f"[cyan]{slug}[/cyan] ({cfg.model_id})")
            console.print(f"  Flavor:  {flavor}")
            console.print(f"  Timeout: {args.timeout}")
            if cfg.image:
                console.print(f"  Image:   {cfg.image}")
                console.print(f"  Python:  {cfg.python}")
                console.print(f"  Env:     {cfg.env}")
            console.print(f"  Script:  {cfg.script}")
            console.print(f"  Args:    {' '.join(script_args)}")
            console.print()
        console.print("Remove --dry-run to launch these jobs.")
        return []

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


SPACE_TEMPLATE = "davanstrien/ocr-bench-space-template"


def cmd_publish(args: argparse.Namespace) -> None:
    """Deploy results viewer as a Hugging Face Space."""
    from huggingface_hub import HfApi, SpaceHardware

    api = HfApi()
    results = args.results
    space_id = args.space or f"{results}-viewer"

    console.print(f"Deploying viewer for [bold]{results}[/bold] to [bold]{space_id}[/bold]...")

    api.duplicate_space(
        from_id=SPACE_TEMPLATE,
        to_id=space_id,
        private=args.private if args.private else None,
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

    # --- Phase 1: run OCR models (waits for jobs to finish by default) ---
    from ocr_bench.run import failed_jobs

    run_argv = ["run", args.input_dataset, args.output_repo, "--seed", str(args.seed)]
    if args.models:
        run_argv += ["--models", *args.models]
    if args.max_samples is not None:
        run_argv += ["--max-samples", str(args.max_samples)]
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
        "judge", args.output_repo, "--from-prs", "--seed", str(args.seed)
    ]
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
    """
    try:
        report = audit_repo(
            args.dataset,
            split=args.split,
            max_ocr_text_len=args.max_ocr_text_len,
        )
    except (DatasetError, OSError, OpenAIError, ValueError) as exc:
        # Couldn't even run the check (repo missing, no OCR columns, network /
        # Hub outage). Distinct from an integrity failure so automation can tell
        # "broken run" from "bad data".
        console.print(f"[red]Audit could not complete:[/red] {exc}")
        sys.exit(_AUDIT_EXIT_OPERATIONAL)

    if not report.configs:
        console.print("[red]No OCR configs/columns found to audit.[/red]")
        sys.exit(_AUDIT_EXIT_OPERATIONAL)

    align = report.alignment
    table = Table(title=f"OCR input audit — {args.dataset}", show_lines=False)
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

    console.print(table)

    # Overall alignment line
    if align.status == "misaligned":
        console.print(f"[red]Alignment: {align.describe()}[/red]")
    elif align.status in ("unverified", "partial"):
        console.print(f"[yellow]Alignment: {align.describe()}[/yellow]")
    else:
        console.print(f"Alignment: {align.describe()}")

    # Verdict + exit code (usable in automation)
    if report.has_problems:
        problems: list[str] = []
        if align.status == "misaligned":
            problems.append("row misalignment")
        if report.row_count_mismatch:
            counts = ", ".join(f"{c.stats.name}={c.stats.n_rows}" for c in report.configs)
            problems.append(f"row-count mismatch ({counts})")
        if report.flagged_models:
            flagged = ", ".join(report.flagged_models)
            problems.append(f">{SENTINEL_FLAG_RATE:.0%} sentinels: {flagged}")
        console.print(f"\n[red]FAIL[/red] — {'; '.join(problems)}")
        sys.exit(_AUDIT_EXIT_INTEGRITY)
    console.print("\n[green]OK[/green] — no blocking issues found")


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
