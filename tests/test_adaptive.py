"""Tests for targeted and size-aware adaptive stopping decisions."""

from ocr_bench.adaptive import (
    classify_adjacent_pairs,
    comparison_pair_counts,
    model_parameter_counts,
    parse_parameter_count,
    practical_preferences,
    unresolved_pairs,
)
from ocr_bench.elo import ComparisonResult, Leaderboard


def _board(overlap: bool = True) -> Leaderboard:
    return Leaderboard(
        elo={"small": 1510.0, "large": 1500.0, "third": 1400.0},
        elo_ci={
            "small": (1480.0, 1540.0),
            "large": (1490.0, 1530.0) if overlap else (1420.0, 1470.0),
            "third": (1300.0, 1350.0),
        },
    )


class TestParameterCounts:
    def test_parses_billions_and_millions(self):
        assert parse_parameter_count("8.3B") == 8_300_000_000
        assert parse_parameter_count("34.5M") == 34_500_000
        assert parse_parameter_count("300K") == 300_000

    def test_unknown_size_is_none(self):
        assert parse_parameter_count("n/a") is None
        assert parse_parameter_count("") is None
        assert parse_parameter_count("about 1B") is None

    def test_registry_lookup_uses_model_ids(self):
        counts = model_parameter_counts()
        assert counts["PaddlePaddle/PP-OCRv6_medium"] == 34_500_000
        assert counts["deepseek-ai/DeepSeek-OCR"] == 4_000_000_000
        assert "tesseract-5" not in counts


class TestComparisonPairCounts:
    def test_accepts_results_and_stored_rows(self):
        comparisons = [
            ComparisonResult(0, "b", "a", "A"),
            {"model_a": "a", "model_b": "b"},
            {"model_a": "a", "model_b": "c"},
        ]
        counts = comparison_pair_counts(comparisons)
        assert counts[("a", "b")] == 2
        assert counts[("a", "c")] == 1


class TestAdjacentPairDecisions:
    def test_non_overlapping_pair_is_resolved(self):
        decisions = classify_adjacent_pairs(
            _board(overlap=False),
            {("large", "small"): 10, ("large", "third"): 10},
        )
        assert decisions[0].status == "resolved"

    def test_overlap_is_unresolved_without_size_rule(self):
        decisions = classify_adjacent_pairs(
            _board(),
            {("large", "small"): 10, ("large", "third"): 10},
            parameter_counts={"small": 1_000_000_000, "large": 4_000_000_000},
        )
        assert decisions[0].status == "unresolved"
        assert decisions[0].pair in unresolved_pairs(decisions)

    def test_size_rule_prefers_smaller_without_changing_rank(self):
        decisions = classify_adjacent_pairs(
            _board(),
            {("large", "small"): 10, ("large", "third"): 10},
            size_tie_ratio=3,
            size_tie_min_samples=10,
            parameter_counts={
                "small": 1_000_000_000,
                "large": 4_000_000_000,
                "third": 2_000_000_000,
            },
        )
        first = decisions[0]
        assert first.status == "prefer-smaller"
        assert first.smaller_model == "small"
        assert first.larger_model == "large"
        assert first.size_ratio == 4.0
        assert first.pair not in unresolved_pairs(decisions)
        assert practical_preferences(decisions) == {"small": [first]}
        assert _board().ranked[0][0] == "small"  # classification never mutates ELO/rank

    def test_minimum_direct_evidence_is_required(self):
        decisions = classify_adjacent_pairs(
            _board(),
            {("large", "small"): 9, ("large", "third"): 10},
            size_tie_ratio=3,
            size_tie_min_samples=10,
            parameter_counts={"small": 1_000_000_000, "large": 4_000_000_000},
        )
        assert decisions[0].status == "unresolved"

    def test_missing_ci_cannot_become_a_practical_preference(self):
        board = _board()
        del board.elo_ci["large"]
        decisions = classify_adjacent_pairs(
            board,
            {("large", "small"): 10, ("large", "third"): 10},
            size_tie_ratio=3,
            parameter_counts={"small": 1_000_000_000, "large": 4_000_000_000},
        )
        assert decisions[0].status == "unresolved"

    def test_unknown_size_fails_safe_as_unresolved(self):
        decisions = classify_adjacent_pairs(
            _board(),
            {("large", "small"): 10, ("large", "third"): 10},
            size_tie_ratio=3,
            parameter_counts={"small": 1_000_000_000},
        )
        assert decisions[0].status == "unresolved"

    def test_ratio_below_threshold_remains_unresolved(self):
        decisions = classify_adjacent_pairs(
            _board(),
            {("large", "small"): 10, ("large", "third"): 10},
            size_tie_ratio=3,
            parameter_counts={"small": 1_000_000_000, "large": 2_000_000_000},
        )
        assert decisions[0].status == "unresolved"
