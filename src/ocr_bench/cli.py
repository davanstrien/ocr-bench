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
    discover_pr_configs,
    load_config_dataset,
    load_flat_dataset,
)
from ocr_bench.elo import ComparisonResult, Leaderboard, compute_elo
from ocr_bench.judge import build_comparisons
from ocr_bench.publish import EvalMetadata, publish_results

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
    judge.add_argument("--columns", nargs="+", default=None, help="Explicit OCR column names")
    judge.add_argument(
        "--configs", nargs="+", default=None, help="Config-per-model: list of config names"
    )
    judge.add_argument(
        "--from-prs", action="store_true", help="Auto-discover configs from open PRs"
    )
    judge.add_argument("--merge-prs", action="store_true", help="Merge PRs before loading")

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
    judge.add_argument("--save-results", default=None, help="HF repo id to publish results to")

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

    return parser


def print_leaderboard(board: Leaderboard) -> None:
    """Print leaderboard as a Rich table."""
    table = Table(title="OCR Model Leaderboard")
    table.add_column("Rank", style="bold")
    table.add_column("Model")
    table.add_column("ELO", justify="right")
    table.add_column("Wins", justify="right")
    table.add_column("Losses", justify="right")
    table.add_column("Ties", justify="right")
    table.add_column("Win%", justify="right")

    for rank, (model, elo) in enumerate(board.ranked, 1):
        pct = board.win_pct(model)
        pct_str = f"{pct:.0f}%" if pct is not None else "-"
        table.add_row(
            str(rank),
            model,
            str(round(elo)),
            str(board.wins[model]),
            str(board.losses[model]),
            str(board.ties[model]),
            pct_str,
        )

    console.print(table)


def cmd_judge(args: argparse.Namespace) -> None:
    """Orchestrate: load → compare → judge → elo → print → publish."""
    # --- Load dataset ---
    if args.from_prs or args.configs:
        if args.from_prs:
            config_names, pr_revisions = discover_pr_configs(args.dataset, merge=args.merge_prs)
            if not config_names:
                raise DatasetError("No configs found in open PRs")
            console.print(f"Discovered {len(config_names)} configs from PRs: {config_names}")
        else:
            config_names = args.configs
            pr_revisions = {}

        ds, ocr_columns = load_config_dataset(
            args.dataset,
            config_names,
            split=args.split,
            pr_revisions=pr_revisions if args.from_prs else None,
        )
    else:
        ds, ocr_columns = load_flat_dataset(args.dataset, split=args.split, columns=args.columns)

    console.print(f"Loaded {len(ds)} samples with {len(ocr_columns)} models:")
    for col, model in ocr_columns.items():
        console.print(f"  {col} → {model}")

    # --- Build comparisons ---
    comparisons = build_comparisons(ds, ocr_columns, max_samples=args.max_samples, seed=args.seed)
    console.print(f"\nBuilt {len(comparisons)} pairwise comparisons")

    if not comparisons:
        console.print("[yellow]No valid comparisons — check that OCR columns have text.[/yellow]")
        return

    # --- Run judge(s) ---
    model_specs = args.models or [DEFAULT_JUDGE]
    judges = [parse_judge_spec(spec, max_tokens=args.max_tokens) for spec in model_specs]
    is_jury = len(judges) > 1

    if is_jury:
        console.print(f"\nJury mode: {len(judges)} judges")

    all_results: list[list[dict]] = []
    for judge in judges:
        console.print(f"\nRunning judge: {judge.name}")
        results = judge.judge(comparisons)
        all_results.append(results)

    # --- Aggregate ---
    if is_jury:
        judge_names = [j.name for j in judges]
        aggregated = aggregate_jury_votes(all_results, judge_names)
    else:
        aggregated = all_results[0]

    # --- Convert to ComparisonResult ---
    model_names = list(set(ocr_columns.values()))
    comparison_results: list[ComparisonResult] = []
    for comp, result in zip(comparisons, aggregated):
        if not result:
            continue
        comparison_results.append(
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

    console.print(f"\n{len(comparison_results)}/{len(comparisons)} valid comparisons")

    # --- Compute ELO ---
    board = compute_elo(comparison_results, model_names)
    console.print()
    print_leaderboard(board)

    # --- Publish ---
    if args.save_results:
        metadata = EvalMetadata(
            source_dataset=args.dataset,
            judge_models=[j.name for j in judges],
            seed=args.seed,
            max_samples=args.max_samples or len(ds),
            total_comparisons=len(comparisons),
            valid_comparisons=len(comparison_results),
            from_prs=args.from_prs,
        )
        publish_results(args.save_results, board, metadata)
        console.print(f"\nResults published to [bold]{args.save_results}[/bold]")


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
        console.print(
            f"\nMerge PRs at: https://huggingface.co/datasets/{args.output_repo}/community"
        )
        console.print("\nThen evaluate:")
        console.print(
            f"  ocr-bench judge {args.output_repo} --from-prs "
            f"--save-results {args.output_repo}-results"
        )
    else:
        console.print("\nJobs running in background.")
        console.print("Check status at: https://huggingface.co/settings/jobs")
        console.print(
            f"When complete, merge PRs at: "
            f"https://huggingface.co/datasets/{args.output_repo}/community"
        )


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
    except DatasetError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)
