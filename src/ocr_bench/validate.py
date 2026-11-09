"""Blind human A/B validation for OCR judge quality."""

from __future__ import annotations

import json
import os
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import structlog

logger = structlog.get_logger()

# Confidence thresholds
MIN_ANNOTATIONS_FOR_WARNING = 15
HIGH_AGREEMENT_THRESHOLD = 0.75


@dataclass
class AgreementStats:
    """Tracks agreement between human and VLM judge."""

    agree: int = 0
    soft_disagree: int = 0  # one picks tie, other picks winner
    hard_disagree: int = 0  # both pick winners but opposite
    total: int = 0

    @property
    def agreement_rate(self) -> float:
        """Share of votes where the human and judge gave the same verdict."""
        return self.agree / self.total if self.total else 0.0

    @property
    def soft_disagree_rate(self) -> float:
        """Share of votes where one side picked a winner and the other a tie."""
        return self.soft_disagree / self.total if self.total else 0.0

    @property
    def hard_disagree_rate(self) -> float:
        return self.hard_disagree / self.total if self.total else 0.0


@dataclass
class ValidationComparison:
    """A single comparison for human validation.

    Built from enriched comparison data published by the judge.
    """

    comparison_id: int
    sample_idx: int
    model_a: str
    model_b: str
    winner: str  # judge's verdict (hidden during annotation)
    reason: str
    agreement: str  # jury agreement (e.g. "2/2")
    text_a: str  # OCR text from model A
    text_b: str  # OCR text from model B
    col_a: str
    col_b: str
    swapped: bool  # position-bias randomization for human display
    display_text_a: str = ""  # text shown to human (may be swapped)
    display_text_b: str = ""


@dataclass
class ValidationSession:
    """Holds state for a validation session."""

    comparisons: list[ValidationComparison]
    model_names: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)
    annotations: list[dict[str, Any]] = field(default_factory=list)
    completed_ids: set[int] = field(default_factory=set)


def _is_split_jury(agreement: str) -> bool:
    """Check if a jury vote was split (e.g. '1/2' not '2/2')."""
    parts = agreement.split("/")
    return len(parts) == 2 and parts[0] != parts[1]


def _interleave_by_sample(
    comparisons: list[ValidationComparison],
) -> list[ValidationComparison]:
    """Interleave comparisons so you see different samples before repeating."""
    by_sample: dict[int, list[ValidationComparison]] = defaultdict(list)
    for comp in comparisons:
        by_sample[comp.sample_idx].append(comp)

    result: list[ValidationComparison] = []
    queues = list(by_sample.values())
    while queues:
        next_round = []
        for q in queues:
            result.append(q.pop(0))
            if q:
                next_round.append(q)
        queues = next_round
    return result


def _has_overlapping_cis(
    model_a: str,
    model_b: str,
    ci_map: dict[str, tuple[float, float]],
) -> bool:
    """Check if two models have overlapping confidence intervals."""
    if model_a not in ci_map or model_b not in ci_map:
        return True  # assume overlapping if CI data missing for a model
    a_low, a_high = ci_map[model_a]
    b_low, b_high = ci_map[model_b]
    return max(a_low, b_low) < min(a_high, b_high)


def build_validation_comparisons(
    comparison_rows: list[dict[str, Any]],
    *,
    leaderboard_rows: list[dict[str, Any]] | None = None,
    n: int | None = None,
    prioritize_splits: bool = True,
    seed: int = 42,
) -> list[ValidationComparison]:
    """Build validation comparisons from published judge results.

    Args:
        comparison_rows: Rows from the comparisons config of a results dataset.
        leaderboard_rows: Leaderboard rows with elo_low/elo_high for focus-pairs.
            When provided, comparisons between models with overlapping CIs are
            prioritized (those are where human input can change the ranking).
        n: Max number of comparisons to include (None = all).
        prioritize_splits: Show split-jury cases first (most informative).
        seed: Random seed for position-bias randomization.
    """
    rng = random.Random(seed)

    comps: list[ValidationComparison] = []
    for i, row in enumerate(comparison_rows):
        swapped = rng.random() < 0.5
        text_a = row.get("text_a", "")
        text_b = row.get("text_b", "")

        if swapped:
            display_a, display_b = text_b, text_a
        else:
            display_a, display_b = text_a, text_b

        comps.append(
            ValidationComparison(
                comparison_id=i,
                sample_idx=row.get("sample_idx", i),
                model_a=row.get("model_a", ""),
                model_b=row.get("model_b", ""),
                winner=row.get("winner", "tie"),
                reason=row.get("reason", ""),
                agreement=row.get("agreement", "1/1"),
                text_a=text_a,
                text_b=text_b,
                col_a=row.get("col_a", ""),
                col_b=row.get("col_b", ""),
                swapped=swapped,
                display_text_a=display_a,
                display_text_b=display_b,
            )
        )

    # Build CI map from leaderboard rows (if available and has CI data)
    ci_map: dict[str, tuple[float, float]] = {}
    if leaderboard_rows:
        for row in leaderboard_rows:
            model = row.get("model", "")
            lo = row.get("elo_low")
            hi = row.get("elo_high")
            if model and lo is not None and hi is not None:
                ci_map[model] = (lo, hi)

    if prioritize_splits and ci_map:
        # 4-tier priority: overlapping+split > overlapping+unanimous >
        # resolved+split > resolved+unanimous
        overlap_split: list[ValidationComparison] = []
        overlap_unanimous: list[ValidationComparison] = []
        resolved_split: list[ValidationComparison] = []
        resolved_unanimous: list[ValidationComparison] = []
        for c in comps:
            overlapping = _has_overlapping_cis(c.model_a, c.model_b, ci_map)
            split = _is_split_jury(c.agreement)
            if overlapping and split:
                overlap_split.append(c)
            elif overlapping:
                overlap_unanimous.append(c)
            elif split:
                resolved_split.append(c)
            else:
                resolved_unanimous.append(c)
        ordered = (
            _interleave_by_sample(overlap_split)
            + _interleave_by_sample(overlap_unanimous)
            + _interleave_by_sample(resolved_split)
            + _interleave_by_sample(resolved_unanimous)
        )
    elif prioritize_splits:
        splits = [c for c in comps if _is_split_jury(c.agreement)]
        unanimous = [c for c in comps if not _is_split_jury(c.agreement)]
        ordered = _interleave_by_sample(splits) + _interleave_by_sample(unanimous)
    else:
        ordered = _interleave_by_sample(comps)

    if n is not None and n < len(ordered):
        ordered = ordered[:n]

    # Re-assign comparison IDs after reordering
    return [
        ValidationComparison(
            comparison_id=i,
            sample_idx=c.sample_idx,
            model_a=c.model_a,
            model_b=c.model_b,
            winner=c.winner,
            reason=c.reason,
            agreement=c.agreement,
            text_a=c.text_a,
            text_b=c.text_b,
            col_a=c.col_a,
            col_b=c.col_b,
            swapped=c.swapped,
            display_text_a=c.display_text_a,
            display_text_b=c.display_text_b,
        )
        for i, c in enumerate(ordered)
    ]


def compute_agreement(
    annotations: list[dict[str, Any]],
    comparisons: list[ValidationComparison],
) -> AgreementStats:
    """Compute agreement between human annotations and judge verdicts."""
    comp_by_id = {c.comparison_id: c for c in comparisons}
    stats = AgreementStats()

    for ann in annotations:
        comp = comp_by_id.get(ann.get("comparison_id"))
        if not comp:
            continue

        # Unswap human vote
        human_winner = ann["winner"]
        if comp.swapped:
            if human_winner == "A":
                human_winner = "B"
            elif human_winner == "B":
                human_winner = "A"

        judge_winner = comp.winner
        stats.total += 1

        if human_winner == judge_winner:
            stats.agree += 1
        elif human_winner == "tie" or judge_winner == "tie":
            stats.soft_disagree += 1
        else:
            stats.hard_disagree += 1

    return stats


def compute_human_elo(
    annotations: list[dict[str, Any]],
    comparisons: list[ValidationComparison],
) -> Any:
    """Compute ELO leaderboard from human annotations.

    Returns a ``Leaderboard`` from ``elo.py``, or None if no annotations.
    """
    from ocr_bench.elo import ComparisonResult, compute_elo

    comp_by_id = {c.comparison_id: c for c in comparisons}
    model_set: set[str] = set()
    results: list[ComparisonResult] = []

    for ann in annotations:
        comp = comp_by_id.get(ann.get("comparison_id"))
        if not comp:
            continue

        # Unswap human vote to get canonical winner
        human_winner = ann["winner"]
        if comp.swapped:
            if human_winner == "A":
                human_winner = "B"
            elif human_winner == "B":
                human_winner = "A"

        model_set.add(comp.model_a)
        model_set.add(comp.model_b)
        results.append(
            ComparisonResult(
                sample_idx=comp.sample_idx,
                model_a=comp.model_a,
                model_b=comp.model_b,
                winner=human_winner,
            )
        )

    if not results:
        return None

    return compute_elo(results, sorted(model_set))


def save_annotations(
    path: str,
    metadata: dict[str, Any],
    annotations: list[dict[str, Any]],
) -> None:
    """Atomically save annotations to JSON file."""
    data = {"metadata": metadata, "annotations": annotations}
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def load_annotations(path: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load annotations from JSON file. Returns (metadata, annotations)."""
    if not os.path.exists(path):
        return {}, []
    with open(path) as f:
        data = json.load(f)
    return data.get("metadata", {}), data.get("annotations", [])


def _agreement_banner(stats: AgreementStats) -> str:
    """Format agreement stats for display."""
    if stats.total == 0:
        return ""

    parts = [f"Agree: {stats.agree}"]
    if stats.soft_disagree:
        parts.append(f"Soft: {stats.soft_disagree}")
    if stats.hard_disagree:
        parts.append(f"**Hard: {stats.hard_disagree}**")
    parts.append(f"(of {stats.total})")

    warning = ""
    if stats.total >= MIN_ANNOTATIONS_FOR_WARNING and stats.hard_disagree_rate > 0.25:
        warning = (
            f" -- Many hard disagreements ({stats.hard_disagree}/{stats.total}). "
            "Judge may not be calibrated for this content."
        )

    return f"Judge: {' | '.join(parts)}{warning}"
