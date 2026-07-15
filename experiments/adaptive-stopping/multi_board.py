#!/usr/bin/env python3
"""Cross-collection replay of fixed OCR comparison allocation designs.

The script is read-only. It downloads pinned published comparison configs, selects
stored outcomes without inspecting winner values, and writes local experiment files.
It makes no judge API calls and performs no Hugging Face Hub writes.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, replace
from itertools import combinations
from pathlib import Path
from typing import Any, Literal

import numpy as np
from replay import _fixed_pair_balanced_sample, graph_metrics, load_stored_comparisons
from scipy.stats import kendalltau, spearmanr

from ocr_bench.elo import ComparisonResult, Leaderboard, compute_elo

DesignKind = Literal["pair-balanced", "mixed-random"]


@dataclass(frozen=True)
class BoardSpec:
    """Pinned published comparison board."""

    name: str
    collection: str
    repo: str
    revision: str
    role: str


BOARDS = (
    BoardSpec(
        "rubenstein-30b",
        "rubenstein",
        "davanstrien/ocr-bench-rubenstein-judge-30b",
        "8559f10e71d77d65d56169d4261b7c9ac25bebb3",
        "independent collection",
    ),
    BoardSpec(
        "rubenstein-jury",
        "rubenstein",
        "davanstrien/ocr-bench-rubenstein-judge",
        "62e579c82f2713a58156b8e4248fb54a070ae110",
        "judge sensitivity",
    ),
    BoardSpec(
        "rubenstein-kimi",
        "rubenstein",
        "davanstrien/ocr-bench-rubenstein-judge-kimi-k25",
        "1e76f23a91480a5ec741064ad9deb19004db64d3",
        "judge sensitivity",
    ),
    BoardSpec(
        "ufo-30b",
        "ufo",
        "davanstrien/ocr-bench-ufo-judge-30b",
        "97623b511a0510e13b2f9e849f58d4002663a51b",
        "independent collection",
    ),
    BoardSpec(
        "bpl",
        "bpl",
        "davanstrien/bpl-ocr-bench-results",
        "fa9e7a032b7b245d5dd69fd40e87646613af6401",
        "independent collection; incomplete board",
    ),
    BoardSpec(
        "britannica-qwen35",
        "britannica",
        "davanstrien/ocr-bench-britannica-results-qwen35",
        "8b6602826e2b71c615ed229d388221a7aa7c69b9",
        "same-corpus judge/model-grid sensitivity",
    ),
)

BUDGET_FRACTIONS = (0.25, 0.40, 0.60)
ALLOCATION_SEEDS = (42, 43, 44, 45, 46)
DESIGNS: tuple[DesignKind, ...] = ("pair-balanced", "mixed-random")
TOP_K = 3


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cluster-replicates",
        type=int,
        default=200,
        help="Page-cluster bootstrap replicates per board/design/seed (default: 200).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    return parser.parse_args()


def _rank_order(board: Leaderboard) -> list[str]:
    return [model for model, _ in board.ranked]


def _rank_metrics(
    board: Leaderboard,
    reference: Leaderboard,
    model_names: Sequence[str],
) -> dict[str, Any]:
    reference_rank = _rank_order(reference)
    rank = _rank_order(board)
    reference_position = {model: i for i, model in enumerate(reference_rank, start=1)}
    position = {model: i for i, model in enumerate(rank, start=1)}
    ordered_models = sorted(model_names)
    ref_positions = [reference_position[model] for model in ordered_models]
    positions = [position[model] for model in ordered_models]
    elo_deltas = [abs(board.elo[model] - reference.elo[model]) for model in ordered_models]
    top_k = min(TOP_K, len(model_names))
    return {
        "kendall_tau": float(kendalltau(ref_positions, positions).statistic),
        "spearman_rho": float(spearmanr(ref_positions, positions).statistic),
        "topk_membership_matches": set(rank[:top_k]) == set(reference_rank[:top_k]),
        "topk_order_matches": rank[:top_k] == reference_rank[:top_k],
        "median_absolute_elo_delta": statistics.median(elo_deltas),
        "max_absolute_elo_delta": max(elo_deltas),
        "rank_order": rank,
    }


def _cluster_resample(
    comparisons: Sequence[ComparisonResult],
    *,
    seed: int,
) -> list[ComparisonResult]:
    """Resample page clusters and assign unique synthetic sample indices."""
    grouped: dict[int, list[ComparisonResult]] = defaultdict(list)
    for comparison in comparisons:
        grouped[comparison.sample_idx].append(comparison)
    sample_ids = sorted(grouped)
    rng = random.Random(seed)
    drawn = rng.choices(sample_ids, k=len(sample_ids))
    return [
        replace(comparison, sample_idx=occurrence)
        for occurrence, source_idx in enumerate(drawn)
        for comparison in grouped[source_idx]
    ]


def _mixed_warmup(comparisons: Sequence[ComparisonResult]) -> list[ComparisonResult]:
    sample_ids = sorted({comparison.sample_idx for comparison in comparisons})
    warmup_ids = set(sample_ids[:5])
    return [comparison for comparison in comparisons if comparison.sample_idx in warmup_ids]


def _select_design(
    comparisons: Sequence[ComparisonResult],
    *,
    design: DesignKind,
    budget_fraction: float,
    allocation_seed: int,
) -> list[ComparisonResult]:
    required = _mixed_warmup(comparisons) if design == "mixed-random" else []
    budget = max(len(required), round(len(comparisons) * budget_fraction))
    return _fixed_pair_balanced_sample(
        comparisons,
        budget=budget,
        seed=allocation_seed,
        required=required,
    )


def _robust_pairs(
    reference: Leaderboard,
    bootstrap_boards: Sequence[Leaderboard],
) -> list[tuple[str, str]]:
    """Pairs whose full-board direction repeats in at least 95% of page bootstraps."""
    reference_position = {model: i for i, (model, _) in enumerate(reference.ranked, start=1)}
    robust: list[tuple[str, str]] = []
    for model_a, model_b in combinations(reference_position, 2):
        higher, lower = (
            (model_a, model_b)
            if reference_position[model_a] < reference_position[model_b]
            else (model_b, model_a)
        )
        agreements = 0
        for board in bootstrap_boards:
            position = {model: i for i, (model, _) in enumerate(board.ranked, start=1)}
            agreements += position[higher] < position[lower]
        if agreements / len(bootstrap_boards) >= 0.95:
            robust.append((higher, lower))
    return robust


def _robust_pair_agreement(
    board: Leaderboard,
    robust_pairs: Sequence[tuple[str, str]],
) -> float | None:
    if not robust_pairs:
        return None
    position = {model: i for i, (model, _) in enumerate(board.ranked, start=1)}
    return statistics.mean(position[higher] < position[lower] for higher, lower in robust_pairs)


def _quantile(values: Sequence[float], q: float) -> float:
    return float(np.quantile(np.asarray(values), q))


def _aggregate_design(
    *,
    spec: BoardSpec,
    comparisons: Sequence[ComparisonResult],
    model_names: Sequence[str],
    reference: Leaderboard,
    bootstrap_rows: Sequence[Sequence[ComparisonResult]],
    bootstrap_boards: Sequence[Leaderboard],
    robust_pairs: Sequence[tuple[str, str]],
    design: DesignKind,
    budget_fraction: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    seed_rows: list[dict[str, Any]] = []
    clustered_metrics: list[dict[str, Any]] = []

    for allocation_seed in ALLOCATION_SEEDS:
        selected = _select_design(
            comparisons,
            design=design,
            budget_fraction=budget_fraction,
            allocation_seed=allocation_seed,
        )
        board = compute_elo(selected, list(model_names), n_bootstrap=0)
        point = _rank_metrics(board, reference, model_names)
        point.update(
            {
                "allocation_seed": allocation_seed,
                "comparisons_consumed": len(selected),
                "percentage_saved": 100.0 * (len(comparisons) - len(selected)) / len(comparisons),
                "robust_pair_agreement": _robust_pair_agreement(board, robust_pairs),
                "graph": graph_metrics(selected, model_names),
            }
        )

        per_seed_clustered: list[dict[str, Any]] = []
        for resampled, bootstrap_reference in zip(bootstrap_rows, bootstrap_boards, strict=True):
            bootstrap_selected = _select_design(
                resampled,
                design=design,
                budget_fraction=budget_fraction,
                allocation_seed=allocation_seed,
            )
            bootstrap_board = compute_elo(
                bootstrap_selected,
                list(model_names),
                n_bootstrap=0,
            )
            metric = _rank_metrics(bootstrap_board, bootstrap_reference, model_names)
            metric["robust_pair_agreement"] = _robust_pair_agreement(
                bootstrap_board,
                robust_pairs,
            )
            per_seed_clustered.append(metric)
            clustered_metrics.append(metric)

        point["clustered"] = {
            "kendall_tau_median": statistics.median(
                metric["kendall_tau"] for metric in per_seed_clustered
            ),
            "topk_membership_rate": statistics.mean(
                metric["topk_membership_matches"] for metric in per_seed_clustered
            ),
            "robust_pair_agreement_rate": statistics.mean(
                metric["robust_pair_agreement"]
                for metric in per_seed_clustered
                if metric["robust_pair_agreement"] is not None
            )
            if robust_pairs
            else None,
        }
        seed_rows.append(point)

    point_taus = [row["kendall_tau"] for row in seed_rows]
    point_rhos = [row["spearman_rho"] for row in seed_rows]
    point_max_deltas = [row["max_absolute_elo_delta"] for row in seed_rows]
    clustered_taus = [metric["kendall_tau"] for metric in clustered_metrics]
    clustered_rhos = [metric["spearman_rho"] for metric in clustered_metrics]
    clustered_max_deltas = [metric["max_absolute_elo_delta"] for metric in clustered_metrics]
    clustered_robust = [
        metric["robust_pair_agreement"]
        for metric in clustered_metrics
        if metric["robust_pair_agreement"] is not None
    ]

    clustered_topk_rate = statistics.mean(
        metric["topk_membership_matches"] for metric in clustered_metrics
    )
    clustered_robust_rate = statistics.mean(clustered_robust) if clustered_robust else None
    clustered_spearman_median = statistics.median(clustered_rhos)
    cross_board_criteria = {
        "topk_membership_rate_at_least_0_90": clustered_topk_rate >= 0.90,
        "robust_pair_agreement_at_least_0_95": (
            clustered_robust_rate >= 0.95 if clustered_robust_rate is not None else False
        ),
        "spearman_rho_median_at_least_0_90": clustered_spearman_median >= 0.90,
    }
    cross_board_criteria["passed"] = all(cross_board_criteria.values())

    aggregate = {
        "board": spec.name,
        "collection": spec.collection,
        "role": spec.role,
        "design": design,
        "budget_fraction": budget_fraction,
        "comparisons_consumed": seed_rows[0]["comparisons_consumed"],
        "percentage_saved": seed_rows[0]["percentage_saved"],
        "allocation_seeds": list(ALLOCATION_SEEDS),
        "robust_full_board_pairs": len(robust_pairs),
        "possible_pairs": len(model_names) * (len(model_names) - 1) // 2,
        "point_kendall_tau_median": statistics.median(point_taus),
        "point_kendall_tau_min": min(point_taus),
        "point_kendall_tau_max": max(point_taus),
        "point_spearman_rho_median": statistics.median(point_rhos),
        "point_topk_membership_matches": sum(row["topk_membership_matches"] for row in seed_rows),
        "point_topk_order_matches": sum(row["topk_order_matches"] for row in seed_rows),
        "point_max_absolute_elo_delta_median": statistics.median(point_max_deltas),
        "point_max_absolute_elo_delta_worst": max(point_max_deltas),
        "clustered_kendall_tau_median": statistics.median(clustered_taus),
        "clustered_kendall_tau_p05": _quantile(clustered_taus, 0.05),
        "clustered_spearman_rho_median": clustered_spearman_median,
        "clustered_topk_membership_rate": clustered_topk_rate,
        "clustered_topk_order_rate": statistics.mean(
            metric["topk_order_matches"] for metric in clustered_metrics
        ),
        "clustered_robust_pair_agreement": clustered_robust_rate,
        "clustered_max_absolute_elo_delta_median": statistics.median(clustered_max_deltas),
        "clustered_max_absolute_elo_delta_p95": _quantile(
            clustered_max_deltas,
            0.95,
        ),
        "minimum_pair_evidence": min(row["graph"]["min_direct_comparisons"] for row in seed_rows),
        "cross_board_criteria": cross_board_criteria,
    }
    return aggregate, seed_rows


def _full_board_stability(
    reference: Leaderboard,
    bootstrap_boards: Sequence[Leaderboard],
    model_names: Sequence[str],
) -> dict[str, Any]:
    metrics = [_rank_metrics(board, reference, model_names) for board in bootstrap_boards]
    return {
        "kendall_tau_median": statistics.median(metric["kendall_tau"] for metric in metrics),
        "kendall_tau_p05": _quantile([metric["kendall_tau"] for metric in metrics], 0.05),
        "topk_membership_rate": statistics.mean(
            metric["topk_membership_matches"] for metric in metrics
        ),
        "topk_order_rate": statistics.mean(metric["topk_order_matches"] for metric in metrics),
    }


def _write_summary(path: Path, aggregates: Sequence[dict[str, Any]]) -> None:
    fields = [
        "board",
        "collection",
        "design",
        "budget_fraction",
        "comparisons_consumed",
        "percentage_saved",
        "robust_full_board_pairs",
        "possible_pairs",
        "point_spearman_rho_median",
        "point_topk_membership_matches",
        "clustered_spearman_rho_median",
        "clustered_topk_membership_rate",
        "clustered_robust_pair_agreement",
        "clustered_max_absolute_elo_delta_median",
        "clustered_max_absolute_elo_delta_p95",
        "minimum_pair_evidence",
        "passed",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for aggregate in aggregates:
            row = {field: aggregate.get(field) for field in fields}
            row["passed"] = aggregate["cross_board_criteria"]["passed"]
            writer.writerow(row)


def main() -> None:
    args = _parse_args()
    if args.cluster_replicates < 1:
        raise ValueError("--cluster-replicates must be at least 1")

    board_results: list[dict[str, Any]] = []
    aggregates: list[dict[str, Any]] = []
    for spec in BOARDS:
        print(f"Loading {spec.name} ...", flush=True)
        comparisons, source_rows, sentinel_rows = load_stored_comparisons(
            spec.repo,
            spec.revision,
        )
        model_names = sorted({model for row in comparisons for model in (row.model_a, row.model_b)})
        reference = compute_elo(comparisons, model_names, n_bootstrap=0)

        bootstrap_rows = [
            _cluster_resample(comparisons, seed=10_000 + replicate)
            for replicate in range(args.cluster_replicates)
        ]
        bootstrap_boards = [
            compute_elo(rows, model_names, n_bootstrap=0) for rows in bootstrap_rows
        ]
        robust_pairs = _robust_pairs(reference, bootstrap_boards)
        board_aggregates: list[dict[str, Any]] = []
        board_seed_rows: list[dict[str, Any]] = []
        for design in DESIGNS:
            for budget_fraction in BUDGET_FRACTIONS:
                print(
                    f"  {design} at {budget_fraction:.0%} budget ...",
                    flush=True,
                )
                aggregate, seed_rows = _aggregate_design(
                    spec=spec,
                    comparisons=comparisons,
                    model_names=model_names,
                    reference=reference,
                    bootstrap_rows=bootstrap_rows,
                    bootstrap_boards=bootstrap_boards,
                    robust_pairs=robust_pairs,
                    design=design,
                    budget_fraction=budget_fraction,
                )
                board_aggregates.append(aggregate)
                aggregates.append(aggregate)
                board_seed_rows.extend(
                    {
                        "design": design,
                        "budget_fraction": budget_fraction,
                        **row,
                    }
                    for row in seed_rows
                )

        board_results.append(
            {
                "name": spec.name,
                "collection": spec.collection,
                "repo": spec.repo,
                "revision": spec.revision,
                "role": spec.role,
                "stored_comparisons": source_rows,
                "sentinel_comparisons_detectable": sentinel_rows,
                "models": model_names,
                "sample_count": len({row.sample_idx for row in comparisons}),
                "full_rank_order": _rank_order(reference),
                "robust_pairs": robust_pairs,
                "full_board_page_bootstrap_stability": _full_board_stability(
                    reference,
                    bootstrap_boards,
                    model_names,
                ),
                "aggregates": board_aggregates,
                "seed_results": board_seed_rows,
            }
        )

    collection_passes: dict[str, dict[str, int]] = defaultdict(lambda: {"passed": 0, "total": 0})
    for aggregate in aggregates:
        key = f"{aggregate['design']}@{aggregate['budget_fraction']:.0%}"
        collection_key = f"{aggregate['collection']}:{key}"
        collection_passes[collection_key]["total"] += 1
        collection_passes[collection_key]["passed"] += int(
            aggregate["cross_board_criteria"]["passed"]
        )

    payload = {
        "experiment": {
            "boards": len(BOARDS),
            "independent_collections": sorted({spec.collection for spec in BOARDS}),
            "budget_fractions": list(BUDGET_FRACTIONS),
            "allocation_seeds": list(ALLOCATION_SEEDS),
            "cluster_replicates": args.cluster_replicates,
            "top_k": TOP_K,
            "robust_pair_definition": (
                "full-board rank direction repeats in at least 95% of page-cluster bootstraps"
            ),
            "cross_board_criteria": {
                "clustered_topk_membership_rate_minimum": 0.90,
                "clustered_robust_pair_agreement_minimum": 0.95,
                "clustered_spearman_rho_median_minimum": 0.90,
            },
            "notes": [
                "Rubenstein judge variants are judge sensitivity, not independent collections.",
                "Allocation uses only published valid rows and never inspects winner values.",
                "Older comparison schemas without OCR text cannot be rechecked for sentinels.",
                "Page bootstrap resamples sample_idx clusters and reruns allocation per replicate.",
            ],
        },
        "boards": board_results,
        "aggregates": aggregates,
        "collection_design_pass_counts": dict(collection_passes),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "multi-board-results.json"
    csv_path = args.output_dir / "multi-board-summary.csv"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _write_summary(csv_path, aggregates)
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
