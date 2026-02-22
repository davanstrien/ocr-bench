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

    # --- browse subcommand ---
    browse = sub.add_parser("browse", help="Browse evaluation results in a web UI")
    browse.add_argument("results", help="HF dataset repo id with published results")
    browse.add_argument("--port", type=int, default=7860, help="Port for Gradio server")
    browse.add_argument("--share", action="store_true", help="Create a public Gradio link")

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


def cmd_browse(args: argparse.Namespace) -> None:
    """Launch the Gradio results viewer."""
    try:
        from ocr_bench.viewer import launch_viewer
    except ImportError:
        console.print(
            "[red]Error:[/red] Gradio is not installed. "
            "Install the viewer extra: [bold]pip install ocr-bench\\[viewer][/bold]"
        )
        sys.exit(1)

    launch_viewer(args.results, server_port=args.port, share=args.share)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    try:
        if args.command == "judge":
            cmd_judge(args)
        elif args.command == "browse":
            cmd_browse(args)
    except DatasetError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)
