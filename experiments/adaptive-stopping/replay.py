#!/usr/bin/env python3
"""Replay adaptive allocation against stored Britannica comparison outcomes.

This experiment is deliberately read-only: it downloads the pinned ``comparisons``
config, selects already-published outcomes in production sample-batch order, and
never constructs a judge backend or writes to the Hugging Face Hub.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

from datasets import load_dataset
from scipy.stats import kendalltau, spearmanr

from ocr_bench.adaptive import (
    AdjacentPairDecision,
    classify_adjacent_pairs,
    comparison_pair_counts,
    model_parameter_counts,
    normalize_model_pair,
    practical_preferences,
    unresolved_pairs,
)
from ocr_bench.elo import ComparisonResult, Leaderboard, Winner, compute_elo
from ocr_bench.judge import is_sentinel

DEFAULT_REPO = "davanstrien/ocr-bench-britannica-results"
# Pin the board used for the report so later Hub updates cannot change the replay.
DEFAULT_REVISION = "48a0f42de26009892d2784a3a97d6d61525f4040"
BATCH_SAMPLES = 5

StrategyKind = Literal["full", "balanced", "targeted"]


@dataclass(frozen=True)
class StrategyConfig:
    """One replay configuration."""

    name: str
    kind: StrategyKind
    size_tie_ratio: float | None = None
    size_tie_min_samples: int = 10
    warmup_min_per_pair: int | None = None
    balanced_every_n_post_warmup_batches: int | None = None
    size_rule_controls_sampling: bool = True


@dataclass
class ReplayResult:
    """Internal replay result, including objects needed for later metrics."""

    config: StrategyConfig
    comparisons: list[ComparisonResult]
    board: Leaderboard
    decisions: list[AdjacentPairDecision]
    stopping_round: int
    stopping_reason: str
    round_history: list[dict[str, Any]]


PRIMARY_CONFIGS = (
    StrategyConfig("full_stored_board", "full"),
    StrategyConfig("balanced_adaptive", "balanced"),
    StrategyConfig("targeted", "targeted"),
    StrategyConfig("targeted_size_3_min_10", "targeted", 3.0, 10),
)

SENSITIVITY_CONFIGS = (
    StrategyConfig("targeted_size_2_min_10", "targeted", 2.0, 10),
    StrategyConfig("targeted_size_3_min_5", "targeted", 3.0, 5),
    StrategyConfig("targeted_size_3_min_15", "targeted", 3.0, 15),
    StrategyConfig("targeted_size_5_min_10", "targeted", 5.0, 10),
)

FOLLOWUP_CONFIGS = (
    StrategyConfig("targeted_v2_warmup_5", "targeted", warmup_min_per_pair=5),
    StrategyConfig("targeted_v2_warmup_10", "targeted", warmup_min_per_pair=10),
    StrategyConfig(
        "targeted_v2_explore_every_3",
        "targeted",
        balanced_every_n_post_warmup_batches=3,
    ),
    StrategyConfig(
        "targeted_v2_warmup_5_explore_3_annotate_3x",
        "targeted",
        size_tie_ratio=3.0,
        size_tie_min_samples=10,
        warmup_min_per_pair=5,
        balanced_every_n_post_warmup_batches=3,
        size_rule_controls_sampling=False,
    ),
    StrategyConfig(
        "targeted_v2_warmup_10_explore_3_annotate_3x",
        "targeted",
        size_tie_ratio=3.0,
        size_tie_min_samples=10,
        warmup_min_per_pair=10,
        balanced_every_n_post_warmup_batches=3,
        size_rule_controls_sampling=False,
    ),
    StrategyConfig(
        "targeted_v2_warmup_5_explore_3_size_stop_3x",
        "targeted",
        size_tie_ratio=3.0,
        size_tie_min_samples=10,
        warmup_min_per_pair=5,
        balanced_every_n_post_warmup_batches=3,
        size_rule_controls_sampling=True,
    ),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=1000,
        help="Bootstrap replicates per interim board (production default: 1000).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=2,
        help="Repeated primary replays used for the determinism check.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    parser.add_argument(
        "--skip-sensitivity",
        action="store_true",
        help="Skip the size ratio/min-evidence sensitivity configurations.",
    )
    parser.add_argument(
        "--skip-followup",
        action="store_true",
        help="Skip targeted-v2 warm-up and balanced-exploration configurations.",
    )
    parser.add_argument(
        "--exclude-sentinel-comparisons",
        action="store_true",
        help=(
            "Robustness mode: remove historical rows containing current error "
            "sentinels. The committed primary report intentionally uses all 4,293 "
            "stored outcomes as requested."
        ),
    )
    return parser.parse_args()


def load_stored_comparisons(
    repo: str,
    revision: str,
    *,
    exclude_sentinels: bool = False,
) -> tuple[list[ComparisonResult], int, int]:
    """Load canonical stored verdicts at a pinned, read-only Hub revision."""
    dataset = load_dataset(repo, "comparisons", split="train", revision=revision)
    source_rows = len(dataset)
    results: list[ComparisonResult] = []
    sentinel_rows = 0
    for row in dataset:
        has_sentinel = is_sentinel(row.get("text_a")) or is_sentinel(row.get("text_b"))
        sentinel_rows += int(has_sentinel)
        if exclude_sentinels and has_sentinel:
            continue
        winner = row["winner"]
        if winner not in {"A", "B", "tie"}:
            raise ValueError(f"Unexpected stored winner: {winner!r}")
        results.append(
            ComparisonResult(
                sample_idx=row["sample_idx"],
                model_a=row["model_a"],
                model_b=row["model_b"],
                winner=cast(Winner, winner),
                reason=row.get("reason", ""),
                agreement=row.get("agreement", "1/1"),
                # Published winners are already canonical/unswapped.
                swapped=False,
                text_a=row.get("text_a", ""),
                text_b=row.get("text_b", ""),
                col_a=row.get("col_a", ""),
                col_b=row.get("col_b", ""),
                truncated_a=row.get("truncated_a", False),
                truncated_b=row.get("truncated_b", False),
            )
        )
    return results, source_rows, sentinel_rows


def _group_by_sample(
    comparisons: Iterable[ComparisonResult],
) -> dict[int, list[ComparisonResult]]:
    grouped: dict[int, list[ComparisonResult]] = defaultdict(list)
    for comparison in comparisons:
        grouped[comparison.sample_idx].append(comparison)
    return dict(grouped)


def _classify(
    board: Leaderboard,
    comparisons: Sequence[ComparisonResult],
    config: StrategyConfig,
    *,
    for_sampling: bool = False,
) -> list[AdjacentPairDecision]:
    """Call the production classifier with the production registry sizes."""
    size_tie_ratio = config.size_tie_ratio
    if for_sampling and not config.size_rule_controls_sampling:
        size_tie_ratio = None
    return classify_adjacent_pairs(
        board,
        comparison_pair_counts(comparisons),
        size_tie_ratio=size_tie_ratio,
        size_tie_min_samples=config.size_tie_min_samples,
        parameter_counts=model_parameter_counts(),
    )


def _status_counts(decisions: Sequence[AdjacentPairDecision]) -> dict[str, int]:
    counts = Counter(decision.status for decision in decisions)
    return {
        "resolved": counts["resolved"],
        "prefer-smaller": counts["prefer-smaller"],
        "unresolved": counts["unresolved"],
    }


def _warmup_ready(
    pair_counts: Counter[tuple[str, str]],
    *,
    n_pairs: int,
    total_comparisons: int,
    min_before_check: int,
    min_per_pair: int | None,
) -> bool:
    """Whether aggregate and optional per-pair balanced warm-up gates are met."""
    if total_comparisons < min_before_check:
        return False
    if min_per_pair is None:
        return True
    return len(pair_counts) == n_pairs and min(pair_counts.values(), default=0) >= min_per_pair


def replay_strategy(
    stored: Sequence[ComparisonResult],
    model_names: Sequence[str],
    config: StrategyConfig,
    *,
    n_bootstrap: int = 1000,
    batch_samples: int = BATCH_SAMPLES,
) -> ReplayResult:
    """Replay one strategy with production decisions and configured allocation.

    The current strategy checks after ``max(3 * n_pairs, 20)`` outcomes. Follow-up
    configurations may additionally require a per-pair warm-up floor, inject a
    balanced batch after every N post-warm-up batches, or keep the size rule as an
    annotation without allowing it to remove pairs from sampling.
    """
    if not stored:
        raise ValueError("Cannot replay an empty comparison set")
    grouped = _group_by_sample(stored)
    sample_order = list(range(min(grouped), max(grouped) + 1))
    n_pairs = len(model_names) * (len(model_names) - 1) // 2
    min_before_check = max(3 * n_pairs, 20)

    if config.kind == "full":
        comparisons = list(stored)
        board = compute_elo(comparisons, list(model_names), n_bootstrap=n_bootstrap)
        decisions = _classify(board, comparisons, config)
        return ReplayResult(
            config=config,
            comparisons=comparisons,
            board=board,
            decisions=decisions,
            stopping_round=math.ceil(len(sample_order) / batch_samples),
            stopping_reason="reference_full_stored_board",
            round_history=[],
        )

    accumulated: list[ComparisonResult] = []
    active_pairs: set[tuple[str, str]] | None = None
    round_history: list[dict[str, Any]] = []
    board: Leaderboard | None = None
    decisions: list[AdjacentPairDecision] = []
    stopping_round = math.ceil(len(sample_order) / batch_samples)
    stopping_reason = "sample_batches_exhausted"
    post_warmup_batches = 0

    for round_number, start in enumerate(range(0, len(sample_order), batch_samples), start=1):
        batch_indices = sample_order[start : start + batch_samples]
        allocation_pairs = active_pairs
        if config.kind == "targeted" and active_pairs is None:
            allocation_mode = "balanced-warmup"
        elif config.kind == "targeted":
            post_warmup_batches += 1
            exploration_every = config.balanced_every_n_post_warmup_batches
            if exploration_every and post_warmup_batches % exploration_every == 0:
                allocation_pairs = None
                allocation_mode = "balanced-exploration"
            else:
                allocation_mode = "targeted"
        else:
            allocation_mode = "balanced"

        batch = [
            comparison
            for sample_idx in batch_indices
            for comparison in grouped.get(sample_idx, [])
            if allocation_pairs is None
            or normalize_model_pair(comparison.model_a, comparison.model_b) in allocation_pairs
        ]
        # Production skips the decision check when build_comparisons returns no rows.
        if not batch:
            continue
        accumulated.extend(batch)
        pair_counts = comparison_pair_counts(accumulated)
        if not _warmup_ready(
            pair_counts,
            n_pairs=n_pairs,
            total_comparisons=len(accumulated),
            min_before_check=min_before_check,
            min_per_pair=config.warmup_min_per_pair,
        ):
            round_history.append(
                {
                    "round": round_number,
                    "sample_indices": batch_indices,
                    "batch_comparisons": len(batch),
                    "cumulative_comparisons": len(accumulated),
                    "allocation_mode": allocation_mode,
                    "active_pairs": None if active_pairs is None else len(active_pairs),
                    "observed_pairs": len(pair_counts),
                    "min_direct_comparisons": min(pair_counts.values(), default=0),
                    "decision": "warm-up",
                }
            )
            continue

        board = compute_elo(accumulated, list(model_names), n_bootstrap=n_bootstrap)
        decisions = _classify(board, accumulated, config, for_sampling=True)
        next_pairs = unresolved_pairs(decisions)
        round_history.append(
            {
                "round": round_number,
                "sample_indices": batch_indices,
                "batch_comparisons": len(batch),
                "cumulative_comparisons": len(accumulated),
                "allocation_mode": allocation_mode,
                "active_pairs": None if active_pairs is None else len(active_pairs),
                "next_active_pairs": len(next_pairs),
                "decision": _status_counts(decisions),
                "rank_order": [model for model, _ in board.ranked],
            }
        )
        if not next_pairs:
            stopping_round = round_number
            stopping_reason = "adaptive_criteria_met"
            break
        if config.kind == "targeted":
            active_pairs = next_pairs

    # cmd_judge refits once more for final publication, even after the last check.
    board = compute_elo(accumulated, list(model_names), n_bootstrap=n_bootstrap)
    decisions = _classify(board, accumulated, config)
    return ReplayResult(
        config=config,
        comparisons=accumulated,
        board=board,
        decisions=decisions,
        stopping_round=stopping_round,
        stopping_reason=stopping_reason,
        round_history=round_history,
    )


def graph_metrics(
    comparisons: Sequence[ComparisonResult], model_names: Sequence[str]
) -> dict[str, Any]:
    """Summarize direct-pair coverage and undirected comparison connectivity."""
    pair_counts = comparison_pair_counts(comparisons)
    possible_pairs = len(model_names) * (len(model_names) - 1) // 2
    adjacency = {model: set() for model in model_names}
    for model_a, model_b in pair_counts:
        adjacency[model_a].add(model_b)
        adjacency[model_b].add(model_a)

    components: list[list[str]] = []
    unseen = set(model_names)
    while unseen:
        root = min(unseen)
        stack = [root]
        component: set[str] = set()
        while stack:
            model = stack.pop()
            if model in component:
                continue
            component.add(model)
            stack.extend(adjacency[model] - component)
        unseen -= component
        components.append(sorted(component))

    direct_counts = list(pair_counts.values())
    return {
        "observed_pairs": len(pair_counts),
        "possible_pairs": possible_pairs,
        "pair_coverage_pct": 100.0 * len(pair_counts) / possible_pairs,
        "min_direct_comparisons": min(direct_counts, default=0),
        "median_direct_comparisons": statistics.median(direct_counts) if direct_counts else 0,
        "max_direct_comparisons": max(direct_counts, default=0),
        "connected": len(components) == 1,
        "connected_components": components,
    }


def _preference_rows(decisions: Sequence[AdjacentPairDecision]) -> list[dict[str, Any]]:
    preferences = practical_preferences(decisions)
    return [
        {
            "smaller_model": smaller,
            "larger_model": decision.larger_model,
            "size_ratio": decision.size_ratio,
            "higher_ranked_model": decision.higher_model,
        }
        for smaller, grouped in sorted(preferences.items())
        for decision in grouped
    ]


def _determinism_signature(result: ReplayResult) -> dict[str, Any]:
    return {
        "comparison_keys": [
            (r.sample_idx, normalize_model_pair(r.model_a, r.model_b), r.winner)
            for r in result.comparisons
        ],
        "elo": result.board.elo,
        "elo_ci": result.board.elo_ci,
        "ranked": result.board.ranked,
        "decisions": [asdict(decision) for decision in result.decisions],
        "stopping_round": result.stopping_round,
        "stopping_reason": result.stopping_reason,
        "round_history": result.round_history,
    }


def strategy_metrics(
    result: ReplayResult,
    reference: ReplayResult,
    model_names: Sequence[str],
    *,
    deterministic: bool | None,
) -> dict[str, Any]:
    """Compute all requested comparison-to-reference metrics."""
    reference_rank = [model for model, _ in reference.board.ranked]
    strategy_rank = [model for model, _ in result.board.ranked]
    ref_position = {model: rank for rank, model in enumerate(reference_rank, start=1)}
    strategy_position = {model: rank for rank, model in enumerate(strategy_rank, start=1)}
    ordered_models = sorted(model_names)
    ref_positions = [ref_position[model] for model in ordered_models]
    strategy_positions = [strategy_position[model] for model in ordered_models]
    kendall = kendalltau(ref_positions, strategy_positions).statistic
    spearman = spearmanr(ref_positions, strategy_positions).statistic

    elo_deltas = {
        model: result.board.elo[model] - reference.board.elo[model] for model in ordered_models
    }
    absolute_deltas = [abs(delta) for delta in elo_deltas.values()]
    statuses = _status_counts(result.decisions)
    allocation_statuses = _status_counts(
        _classify(result.board, result.comparisons, result.config, for_sampling=True)
    )
    if result.stopping_reason == "adaptive_criteria_met":
        resolution = (
            f"allocation criteria met: {allocation_statuses['resolved']} statistically resolved, "
            f"{allocation_statuses['prefer-smaller']} sampling preferences"
        )
    elif result.config.kind == "full":
        resolution = "reference only; no adaptive stop"
    else:
        resolution = (
            "sample batches exhausted: "
            f"{allocation_statuses['unresolved']} allocation pairs unresolved"
        )

    full_count = len(reference.comparisons)
    percentage_saved = 100.0 * (full_count - len(result.comparisons)) / full_count
    top3_reference = reference_rank[:3]
    top3_strategy = strategy_rank[:3]
    top3_order_matches = top3_strategy == top3_reference
    max_absolute_delta = max(absolute_deltas)
    followup_gate = {
        "top3_order_exact": top3_order_matches,
        "kendall_tau_at_least_0_95": float(kendall) >= 0.95,
        "max_absolute_elo_delta_at_most_50": max_absolute_delta <= 50.0,
        "percentage_saved_at_least_60": percentage_saved >= 60.0,
    }
    followup_gate["passed"] = all(followup_gate.values())
    return {
        "name": result.config.name,
        "kind": result.config.kind,
        "size_tie_ratio": result.config.size_tie_ratio,
        "size_tie_min_samples": result.config.size_tie_min_samples,
        "warmup_min_per_pair": result.config.warmup_min_per_pair,
        "balanced_every_n_post_warmup_batches": (
            result.config.balanced_every_n_post_warmup_batches
        ),
        "size_rule_controls_sampling": result.config.size_rule_controls_sampling,
        "comparisons_consumed": len(result.comparisons),
        "comparisons_saved": full_count - len(result.comparisons),
        "percentage_saved": percentage_saved,
        "stopping_round": result.stopping_round,
        "stopping_reason": result.stopping_reason,
        "resolution_summary": resolution,
        "final_rank_order": strategy_rank,
        "kendall_tau": float(kendall),
        "spearman_rho": float(spearman),
        "top3": top3_strategy,
        "top3_membership_matches": set(top3_strategy) == set(top3_reference),
        "top3_membership_overlap": len(set(top3_strategy) & set(top3_reference)),
        "top3_order_matches": top3_order_matches,
        "elo": result.board.elo,
        "elo_ci": result.board.elo_ci,
        "elo_delta_from_full": elo_deltas,
        "median_absolute_elo_delta": statistics.median(absolute_deltas),
        "max_absolute_elo_delta": max_absolute_delta,
        "max_absolute_elo_delta_model": max(elo_deltas, key=lambda model: abs(elo_deltas[model])),
        "decision_statuses": statuses,
        "allocation_decision_statuses": allocation_statuses,
        "practical_preferences": _preference_rows(result.decisions),
        "followup_acceptance_gate": followup_gate,
        "graph": graph_metrics(result.comparisons, model_names),
        "deterministic_across_repeats": deterministic,
        "round_history": result.round_history,
    }


def _write_csvs(output_dir: Path, metrics: Sequence[dict[str, Any]]) -> None:
    summary_fields = [
        "name",
        "comparisons_consumed",
        "comparisons_saved",
        "percentage_saved",
        "stopping_round",
        "stopping_reason",
        "kendall_tau",
        "spearman_rho",
        "top3_membership_matches",
        "top3_order_matches",
        "median_absolute_elo_delta",
        "max_absolute_elo_delta",
        "followup_gate_passed",
        "deterministic_across_repeats",
    ]
    with (output_dir / "strategy-summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields, lineterminator="\n")
        writer.writeheader()
        for metric in metrics:
            row = {field: metric.get(field) for field in summary_fields}
            row["followup_gate_passed"] = metric["followup_acceptance_gate"]["passed"]
            writer.writerow(row)

    with (output_dir / "elo-deltas.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["strategy", "model", "elo", "full_elo", "delta", "absolute_delta"],
            lineterminator="\n",
        )
        writer.writeheader()
        reference = metrics[0]
        for metric in metrics:
            for model, delta in metric["elo_delta_from_full"].items():
                writer.writerow(
                    {
                        "strategy": metric["name"],
                        "model": model,
                        "elo": metric["elo"][model],
                        "full_elo": reference["elo"][model],
                        "delta": delta,
                        "absolute_delta": abs(delta),
                    }
                )

    with (output_dir / "round-history.csv").open("w", newline="") as handle:
        fields = [
            "strategy",
            "round",
            "sample_indices",
            "batch_comparisons",
            "cumulative_comparisons",
            "allocation_mode",
            "active_pairs",
            "next_active_pairs",
            "min_direct_comparisons",
            "resolved",
            "prefer_smaller",
            "unresolved",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for metric in metrics:
            for row in metric["round_history"]:
                decision = row.get("decision", {})
                if not isinstance(decision, dict):
                    decision = {}
                writer.writerow(
                    {
                        "strategy": metric["name"],
                        "round": row["round"],
                        "sample_indices": json.dumps(row["sample_indices"]),
                        "batch_comparisons": row["batch_comparisons"],
                        "cumulative_comparisons": row["cumulative_comparisons"],
                        "allocation_mode": row.get("allocation_mode"),
                        "active_pairs": row.get("active_pairs"),
                        "next_active_pairs": row.get("next_active_pairs"),
                        "min_direct_comparisons": row.get("min_direct_comparisons"),
                        "resolved": decision.get("resolved"),
                        "prefer_smaller": decision.get("prefer-smaller"),
                        "unresolved": decision.get("unresolved"),
                    }
                )


def main() -> None:
    args = _parse_args()
    if args.bootstrap < 1:
        raise ValueError("--bootstrap must be at least 1 because adaptive decisions require CIs")
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1")

    stored, source_rows, sentinel_rows = load_stored_comparisons(
        args.repo,
        args.revision,
        exclude_sentinels=args.exclude_sentinel_comparisons,
    )
    model_names = sorted({model for result in stored for model in (result.model_a, result.model_b)})
    configs = list(PRIMARY_CONFIGS)
    if not args.skip_sensitivity:
        configs.extend(SENSITIVITY_CONFIGS)
    if not args.skip_followup:
        configs.extend(FOLLOWUP_CONFIGS)

    results: dict[str, ReplayResult] = {}
    deterministic: dict[str, bool | None] = {}
    repeat_names = {config.name for config in PRIMARY_CONFIGS} | {
        "targeted_v2_warmup_5_explore_3_annotate_3x"
    }
    for config in configs:
        print(f"Replaying {config.name} ...", flush=True)
        first = replay_strategy(stored, model_names, config, n_bootstrap=args.bootstrap)
        results[config.name] = first
        signatures = [_determinism_signature(first)]
        repeats = args.repeats if config.name in repeat_names else 1
        for _ in range(1, repeats):
            repeated = replay_strategy(
                stored,
                model_names,
                config,
                n_bootstrap=args.bootstrap,
            )
            signatures.append(_determinism_signature(repeated))
        deterministic[config.name] = (
            all(signature == signatures[0] for signature in signatures[1:]) if repeats > 1 else None
        )

    reference = results[PRIMARY_CONFIGS[0].name]
    metrics = [
        strategy_metrics(
            results[config.name],
            reference,
            model_names,
            deterministic=deterministic[config.name],
        )
        for config in configs
    ]
    payload = {
        "experiment": {
            "source_repo": args.repo,
            "source_revision": args.revision,
            "stored_comparisons_at_revision": source_rows,
            "comparisons_replayed": len(stored),
            "historical_sentinel_comparisons": sentinel_rows,
            "sentinel_comparisons_excluded": args.exclude_sentinel_comparisons,
            "models": model_names,
            "model_count": len(model_names),
            "sample_count": len({result.sample_idx for result in stored}),
            "batch_samples": BATCH_SAMPLES,
            "min_before_check": max(3 * (len(model_names) * (len(model_names) - 1) // 2), 20),
            "bootstrap_replicates": args.bootstrap,
            "bootstrap_seed": 42,
            "repeats": args.repeats,
            "repeated_strategy_names": sorted(repeat_names & {config.name for config in configs}),
            "followup_acceptance_gate": {
                "top3_order_exact": True,
                "kendall_tau_minimum": 0.95,
                "max_absolute_elo_delta_maximum": 50.0,
                "percentage_saved_minimum": 60.0,
            },
            "production_helpers": [
                "compute_elo",
                "classify_adjacent_pairs",
                "comparison_pair_counts",
                "model_parameter_counts",
                "normalize_model_pair",
                "practical_preferences",
                "unresolved_pairs",
            ],
        },
        "strategies": metrics,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "-sentinels-excluded" if args.exclude_sentinel_comparisons else ""
    json_path = args.output_dir / f"results{suffix}.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if not args.exclude_sentinel_comparisons:
        _write_csvs(args.output_dir, metrics)
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
