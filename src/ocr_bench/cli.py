"""CLI entrypoint for ocr-bench."""

from __future__ import annotations

import argparse
import sys

import structlog
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
from ocr_bench.judge import Comparison, _normalize_pair, build_comparisons, sample_indices
from ocr_bench.publish import (
    EvalMetadata,
    load_existing_comparisons,
    load_existing_metadata,
    publish_results,
)
from ocr_bench.standard_eval import evaluate_against_ground_truth
from ocr_bench.task_config import DEFAULT_GROUND_TRUTH_COLUMN

logger = structlog.get_logger()
console = Console()


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
        "--ground-truth-column",
        default=DEFAULT_GROUND_TRUTH_COLUMN,
        help=f"Ground truth column for standard metrics (default: {DEFAULT_GROUND_TRUTH_COLUMN})",
    )
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

    # Eval
    judge.add_argument("--max-samples", type=int, default=None, help="Max samples to evaluate")
    judge.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
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

    return parser


def print_leaderboard(board: Leaderboard) -> None:
    """Print leaderboard as a Rich table."""
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

    for rank, (model, elo) in enumerate(board.ranked, 1):
        pct = board.win_pct(model)
        pct_str = f"{pct:.0f}%" if pct is not None else "-"
        if has_ci and model in board.elo_ci:
            lo, hi = board.elo_ci[model]
            elo_str = f"{round(elo)} ({round(lo)}\u2013{round(hi)})"
        else:
            elo_str = str(round(elo))
        table.add_row(
            str(rank),
            model,
            sizes.get(model, ""),
            elo_str,
            str(board.wins[model]),
            str(board.losses[model]),
            str(board.ties[model]),
            pct_str,
        )

    console.print(table)


def _convert_results(
    comparisons: list[Comparison], aggregated: list[dict]
) -> list[ComparisonResult]:
    """Convert judged comparisons + aggregated outputs into ComparisonResult list."""
    results: list[ComparisonResult] = []
    for comp, result in zip(comparisons, aggregated):
        if not result:
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


def cmd_judge(args: argparse.Namespace) -> None:
    """Orchestrate: load → compare → judge → elo → print → publish."""
    # --- Resolve flags ---
    adaptive = not args.no_adaptive
    merge = args.merge
    results_repo = _resolve_results_repo(args.dataset, args.save_results, args.no_publish)
    from_prs = False  # track for metadata

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

    standard_metrics = evaluate_against_ground_truth(
        ds,
        ocr_columns,
        ground_truth_column=args.ground_truth_column,
    )
    if standard_metrics:
        metrics_table = Table(title="Standard Evaluation (Dummy Scaffold)")
        metrics_table.add_column("Model")
        metrics_table.add_column("Samples", justify="right")
        metrics_table.add_column("Global F1", justify="right")
        metrics_table.add_column("Jury global F1", justify="right")
        for metric in standard_metrics:
            metrics_table.add_row(
                metric.model,
                str(metric.samples),
                f"{metric.global_f1:.3f}",
                f"{metric.jury_global_f1:.3f}",
            )
        console.print()
        console.print(metrics_table)
    else:
        console.print(
            f"[yellow]Standard evaluation skipped:[/yellow] "
            f"missing or incompatible '{args.ground_truth_column}' column."
        )

    # --- Incremental: load existing comparisons ---
    existing_results: list[ComparisonResult] = []
    existing_meta_rows: list[dict] = []
    skip_pairs: set[tuple[str, str]] | None = None

    if results_repo and not args.full_rejudge:
        existing_results = load_existing_comparisons(results_repo)
        if existing_results:
            judged_pairs = {_normalize_pair(r.model_a, r.model_b) for r in existing_results}
            skip_pairs = judged_pairs
            console.print(
                f"\nIncremental mode: {len(existing_results)} existing comparisons "
                f"across {len(judged_pairs)} model pairs — skipping those."
            )
            existing_meta_rows = load_existing_metadata(results_repo)
        else:
            console.print("\nNo existing comparisons found — full judge run.")

    model_names = list(set(ocr_columns.values()))

    # --- Judge setup (shared by both paths) ---
    model_specs = args.models or [DEFAULT_JUDGE]
    judges = [
        parse_judge_spec(spec, max_tokens=args.max_tokens, concurrency=args.concurrency)
        for spec in model_specs
    ]
    is_jury = len(judges) > 1

    def _judge_batch(batch_comps: list[Comparison]) -> list[ComparisonResult]:
        """Run judge(s) on a batch of comparisons and return ComparisonResults."""
        all_judge_outputs: list[list[dict]] = []
        for judge in judges:
            results = judge.judge(batch_comps)
            all_judge_outputs.append(results)
        if is_jury:
            judge_names = [j.name for j in judges]
            aggregated = aggregate_jury_votes(all_judge_outputs, judge_names)
        else:
            aggregated = all_judge_outputs[0]
        return _convert_results(batch_comps, aggregated)

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

        new_results: list[ComparisonResult] = []
        total_comparisons = 0
        for batch_num, batch_start in enumerate(range(0, len(all_indices), batch_samples)):
            batch_indices = all_indices[batch_start : batch_start + batch_samples]
            batch_comps = build_comparisons(
                ds,
                ocr_columns,
                skip_pairs=skip_pairs,
                indices=batch_indices,
                seed=args.seed,
            )
            if not batch_comps:
                continue

            batch_results = _judge_batch(batch_comps)
            new_results.extend(batch_results)
            total_comparisons += len(batch_comps)
            # batch_comps goes out of scope → GC can free images

            total = len(existing_results) + len(new_results)
            console.print(f"  Batch {batch_num + 1}: {len(batch_results)} new, {total} total")

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

        console.print(f"\n{len(new_results)}/{total_comparisons} valid comparisons")
    else:
        # --- Standard single-pass flow ---
        comparisons = build_comparisons(
            ds,
            ocr_columns,
            max_samples=args.max_samples,
            seed=args.seed,
            skip_pairs=skip_pairs,
        )
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
            print_leaderboard(board)
            if results_repo:
                metadata = EvalMetadata(
                    source_dataset=args.dataset,
                    judge_models=[],
                    seed=args.seed,
                    max_samples=args.max_samples or len(ds),
                    total_comparisons=0,
                    valid_comparisons=0,
                    from_prs=from_prs,
                )
                publish_results(
                    results_repo,
                    board,
                    metadata,
                    existing_metadata=existing_meta_rows,
                )
                console.print(f"\nResults published to [bold]{results_repo}[/bold]")
            return

        if is_jury:
            console.print(f"\nJury mode: {len(judges)} judges")

        for judge in judges:
            console.print(f"\nRunning judge: {judge.name}")

        new_results = _judge_batch(comparisons)
        total_comparisons = len(comparisons)
        console.print(f"\n{len(new_results)}/{total_comparisons} valid comparisons")

    # --- Merge existing + new, compute ELO ---
    all_results = existing_results + new_results
    board = compute_elo(all_results, model_names)
    console.print()
    print_leaderboard(board)

    # --- Publish ---
    if results_repo:
        metadata = EvalMetadata(
            source_dataset=args.dataset,
            judge_models=[j.name for j in judges],
            seed=args.seed,
            max_samples=args.max_samples or len(ds),
            total_comparisons=total_comparisons,
            valid_comparisons=len(new_results),
            from_prs=from_prs,
        )
        publish_results(results_repo, board, metadata, existing_metadata=existing_meta_rows)
        console.print(f"\nResults published to [bold]{results_repo}[/bold]")


def cmd_run(args: argparse.Namespace) -> None:
    """Launch OCR models on a dataset via HF Jobs."""
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
            table.add_row(slug + default, cfg.model_id, cfg.size, cfg.default_flavor)

        console.print(table)
        console.print(f"\nDefault set: {', '.join(DEFAULT_MODELS)}")
        return

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
            console.print(f"  Script:  {cfg.script}")
            console.print(f"  Args:    {' '.join(script_args)}")
            console.print()
        console.print("Remove --dry-run to launch these jobs.")
        return

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
        console.print("\n[bold green]All jobs finished![/bold green]")
        console.print("\nEvaluate:")
        console.print(f"  ocr-bench judge {args.output_repo}")
    else:
        console.print("\nJobs running in background.")
        console.print("Check status at: https://huggingface.co/settings/jobs")
        console.print(f"When complete: ocr-bench judge {args.output_repo}")


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
    from huggingface_hub import HfApi

    api = HfApi()
    results = args.results
    space_id = args.space or f"{results}-viewer"

    console.print(f"Deploying viewer for [bold]{results}[/bold] to [bold]{space_id}[/bold]...")

    api.duplicate_space(
        from_id=SPACE_TEMPLATE,
        to_id=space_id,
        private=args.private if args.private else None,
        hardware="cpu-basic",
        exist_ok=True,
        variables=[{"key": "REPOS", "value": results}],
    )

    api.add_space_variable(repo_id=space_id, key="REPOS", value=results)

    # Update Space metadata to link to results dataset
    try:
        from huggingface_hub import metadata_update

        metadata_update(
            space_id,
            {"datasets": [results], "tags": ["ocr-bench"]},
            repo_type="space",
            overwrite=True,
        )
    except Exception as exc:
        logger.warning("space_metadata_update_failed", error=str(exc))

    url = f"https://huggingface.co/spaces/{space_id}"
    console.print(f"[green]Space published![/green] {url}")


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
    except DatasetError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)
