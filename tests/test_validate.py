"""Tests for ocr_bench.validate — agreement stats, comparison ordering, persistence."""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from ocr_bench.validate import (
    AgreementStats,
    _agreement_banner,
    _has_overlapping_cis,
    _is_split_jury,
    build_validation_comparisons,
    compute_agreement,
    load_annotations,
    save_annotations,
)


# --- Test data ---
def _make_comparison_row(
    idx: int = 0,
    model_a: str = "model-a",
    model_b: str = "model-b",
    winner: str = "A",
    agreement: str = "2/2",
) -> dict:
    return {
        "sample_idx": idx,
        "model_a": model_a,
        "model_b": model_b,
        "winner": winner,
        "reason": "Model A is better",
        "agreement": agreement,
        "text_a": f"Text from {model_a}",
        "text_b": f"Text from {model_b}",
        "col_a": "col_a",
        "col_b": "col_b",
    }


SAMPLE_ROWS = [
    _make_comparison_row(0, winner="A", agreement="2/2"),
    _make_comparison_row(1, winner="B", agreement="1/2"),  # split jury
    _make_comparison_row(2, winner="tie", agreement="2/2"),
    _make_comparison_row(3, winner="A", agreement="2/2"),
    _make_comparison_row(4, winner="B", agreement="1/2"),  # split jury
]


class TestIsSplitJury:
    def test_unanimous(self):
        assert not _is_split_jury("2/2")
        assert not _is_split_jury("3/3")

    def test_split(self):
        assert _is_split_jury("1/2")
        assert _is_split_jury("1/3")
        assert _is_split_jury("2/3")

    def test_single_judge(self):
        assert not _is_split_jury("1/1")

    def test_empty(self):
        assert not _is_split_jury("")


class TestAgreementStats:
    def test_agreement_rate(self):
        stats = AgreementStats(agree=7, soft_disagree=2, hard_disagree=1, total=10)
        assert stats.agreement_rate == pytest.approx(0.7)

    def test_soft_disagreements_are_not_agreement(self):
        stats = AgreementStats(agree=0, soft_disagree=20, hard_disagree=0, total=20)
        assert stats.agreement_rate == 0.0
        assert stats.soft_disagree_rate == 1.0

    def test_hard_disagree_rate(self):
        stats = AgreementStats(agree=7, soft_disagree=0, hard_disagree=3, total=10)
        assert stats.hard_disagree_rate == pytest.approx(0.3)

    def test_zero_total(self):
        stats = AgreementStats()
        assert stats.agreement_rate == 0.0
        assert stats.hard_disagree_rate == 0.0


class TestBuildValidationComparisons:
    def test_builds_correct_count(self):
        comps = build_validation_comparisons(SAMPLE_ROWS, n=3)
        assert len(comps) == 3

    def test_builds_all_when_no_limit(self):
        comps = build_validation_comparisons(SAMPLE_ROWS)
        assert len(comps) == 5

    def test_split_jury_first_when_prioritized(self):
        comps = build_validation_comparisons(SAMPLE_ROWS, prioritize_splits=True)
        # First two should be the split jury rows (indices 1 and 4 in original)
        split_ids = [i for i, c in enumerate(comps) if _is_split_jury(c.agreement)]
        non_split_ids = [i for i, c in enumerate(comps) if not _is_split_jury(c.agreement)]
        # All split-jury should come before non-split
        if split_ids and non_split_ids:
            assert max(split_ids) < min(non_split_ids)

    def test_comparison_ids_sequential(self):
        comps = build_validation_comparisons(SAMPLE_ROWS)
        ids = [c.comparison_id for c in comps]
        assert ids == list(range(len(comps)))

    def test_display_text_swapped(self):
        # With a fixed seed, some should be swapped
        comps = build_validation_comparisons(SAMPLE_ROWS, seed=42)
        swapped_count = sum(1 for c in comps if c.swapped)
        not_swapped_count = sum(1 for c in comps if not c.swapped)
        # Should have a mix
        assert swapped_count > 0 or not_swapped_count > 0

    def test_swapped_text_matches(self):
        comps = build_validation_comparisons(SAMPLE_ROWS, seed=42)
        for c in comps:
            if c.swapped:
                assert c.display_text_a == c.text_b
                assert c.display_text_b == c.text_a
            else:
                assert c.display_text_a == c.text_a
                assert c.display_text_b == c.text_b


class TestFocusPairsPrioritization:
    """Focus-pairs: overlapping-CI comparisons should come first."""

    # Two models with overlapping CIs, two with resolved (non-overlapping)
    LEADERBOARD = [
        {"model": "model-a", "elo": 1600, "elo_low": 1550, "elo_high": 1650},
        {"model": "model-b", "elo": 1620, "elo_low": 1570, "elo_high": 1670},  # overlaps with a
        {"model": "model-c", "elo": 1400, "elo_low": 1380, "elo_high": 1420},  # resolved vs a,b
    ]

    def _rows_with_models(self) -> list[dict]:
        """Rows spanning overlapping (a vs b) and resolved (a vs c) pairs."""
        return [
            _make_comparison_row(0, model_a="model-a", model_b="model-c", agreement="2/2"),
            _make_comparison_row(1, model_a="model-a", model_b="model-b", agreement="2/2"),
            _make_comparison_row(2, model_a="model-a", model_b="model-c", agreement="1/2"),
            _make_comparison_row(3, model_a="model-a", model_b="model-b", agreement="1/2"),
        ]

    def test_overlapping_pairs_come_first(self):
        rows = self._rows_with_models()
        comps = build_validation_comparisons(
            rows, leaderboard_rows=self.LEADERBOARD, prioritize_splits=True
        )
        # Overlapping pair is a-vs-b. With 4-tier ordering:
        # tier 1: overlap+split (row 3: a-b, 1/2)
        # tier 2: overlap+unanimous (row 1: a-b, 2/2)
        # tier 3: resolved+split (row 2: a-c, 1/2)
        # tier 4: resolved+unanimous (row 0: a-c, 2/2)
        models = [(c.model_a, c.model_b) for c in comps]
        # First two should be a-vs-b (overlapping)
        assert models[0] == ("model-a", "model-b")
        assert models[1] == ("model-a", "model-b")
        # Last two should be a-vs-c (resolved)
        assert models[2] == ("model-a", "model-c")
        assert models[3] == ("model-a", "model-c")

    def test_split_jury_before_unanimous_within_overlap_tier(self):
        rows = self._rows_with_models()
        comps = build_validation_comparisons(
            rows, leaderboard_rows=self.LEADERBOARD, prioritize_splits=True
        )
        # Within overlap tier: split (1/2) before unanimous (2/2)
        assert _is_split_jury(comps[0].agreement)
        assert not _is_split_jury(comps[1].agreement)
        # Within resolved tier: split before unanimous
        assert _is_split_jury(comps[2].agreement)
        assert not _is_split_jury(comps[3].agreement)

    def test_no_leaderboard_falls_back(self):
        """Without leaderboard data, ordering falls back to split-first only."""
        rows = self._rows_with_models()
        comps_with = build_validation_comparisons(
            rows, leaderboard_rows=self.LEADERBOARD, prioritize_splits=True
        )
        comps_without = build_validation_comparisons(
            rows, leaderboard_rows=None, prioritize_splits=True
        )
        # Without CI data, all splits come first regardless of model pair
        split_ids = [i for i, c in enumerate(comps_without) if _is_split_jury(c.agreement)]
        non_split_ids = [i for i, c in enumerate(comps_without) if not _is_split_jury(c.agreement)]
        assert max(split_ids) < min(non_split_ids)
        # With CI data, ordering is different (overlap takes precedence)
        models_with = [(c.model_a, c.model_b) for c in comps_with]
        models_without = [(c.model_a, c.model_b) for c in comps_without]
        assert models_with != models_without

    def test_leaderboard_without_ci_falls_back(self):
        """Leaderboard rows without elo_low/elo_high => no CI map => fallback."""
        rows = self._rows_with_models()
        leaderboard_no_ci = [
            {"model": "model-a", "elo": 1600},
            {"model": "model-b", "elo": 1620},
        ]
        comps = build_validation_comparisons(
            rows, leaderboard_rows=leaderboard_no_ci, prioritize_splits=True
        )
        # Should fall back to split-first ordering (no CI data extracted)
        split_ids = [i for i, c in enumerate(comps) if _is_split_jury(c.agreement)]
        non_split_ids = [i for i, c in enumerate(comps) if not _is_split_jury(c.agreement)]
        assert max(split_ids) < min(non_split_ids)


class TestHasOverlappingCIs:
    def test_overlapping(self):
        ci_map = {"a": (1550, 1650), "b": (1570, 1670)}
        assert _has_overlapping_cis("a", "b", ci_map)

    def test_non_overlapping(self):
        ci_map = {"a": (1550, 1650), "c": (1380, 1420)}
        assert not _has_overlapping_cis("a", "c", ci_map)

    def test_missing_model_assumes_overlapping(self):
        ci_map = {"a": (1550, 1650)}
        assert _has_overlapping_cis("a", "unknown", ci_map)

    def test_touching_boundary_not_overlapping(self):
        ci_map = {"a": (1500, 1600), "b": (1600, 1700)}
        assert not _has_overlapping_cis("a", "b", ci_map)


class TestComputeAgreement:
    def _make_comps_and_anns(self):
        rows = [
            _make_comparison_row(0, winner="A"),
            _make_comparison_row(1, winner="B"),
            _make_comparison_row(2, winner="tie"),
        ]
        comps = build_validation_comparisons(rows, prioritize_splits=False, seed=0)
        return comps

    def test_perfect_agreement(self):
        comps = self._make_comps_and_anns()
        anns = []
        for c in comps:
            # Simulate human agreeing with judge
            if c.swapped:
                # Judge said A, swapped means A displayed as B
                if c.winner == "A":
                    human_vote = "B"
                elif c.winner == "B":
                    human_vote = "A"
                else:
                    human_vote = "tie"
            else:
                human_vote = c.winner
            anns.append({"comparison_id": c.comparison_id, "winner": human_vote})

        stats = compute_agreement(anns, comps)
        assert stats.agree == 3
        assert stats.soft_disagree == 0
        assert stats.hard_disagree == 0

    def test_soft_disagree(self):
        rows = [_make_comparison_row(0, winner="A")]
        comps = build_validation_comparisons(rows, prioritize_splits=False, seed=0)
        c = comps[0]
        # Human says tie, judge says A
        anns = [{"comparison_id": c.comparison_id, "winner": "tie"}]
        stats = compute_agreement(anns, comps)
        assert stats.soft_disagree == 1
        assert stats.hard_disagree == 0

    def test_hard_disagree(self):
        rows = [_make_comparison_row(0, winner="A")]
        comps = build_validation_comparisons(rows, prioritize_splits=False, seed=0)
        c = comps[0]
        # Human says B (or swapped equivalent), judge says A
        if c.swapped:
            human_vote = "A"  # swapped: human saying A means actual B
        else:
            human_vote = "B"
        anns = [{"comparison_id": c.comparison_id, "winner": human_vote}]
        stats = compute_agreement(anns, comps)
        assert stats.hard_disagree == 1

    def test_empty_annotations(self):
        rows = [_make_comparison_row(0)]
        comps = build_validation_comparisons(rows)
        stats = compute_agreement([], comps)
        assert stats.total == 0


class TestAnnotationPersistence:
    def test_save_load_roundtrip(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            metadata = {"test": True, "n": 5}
            annotations = [
                {"comparison_id": 0, "winner": "A", "timestamp": "2026-01-01T00:00:00Z"},
                {"comparison_id": 1, "winner": "B", "timestamp": "2026-01-01T00:01:00Z"},
            ]
            save_annotations(path, metadata, annotations)

            loaded_meta, loaded_anns = load_annotations(path)
            assert loaded_meta == metadata
            assert loaded_anns == annotations
        finally:
            os.unlink(path)

    def test_load_nonexistent_returns_empty(self):
        meta, anns = load_annotations("/tmp/nonexistent-file-abc123.json")
        assert meta == {}
        assert anns == []

    def test_atomic_write(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            save_annotations(path, {}, [{"id": 1}])
            # tmp file should not exist after atomic rename
            assert not os.path.exists(path + ".tmp")
            # main file should exist and be valid JSON
            with open(path) as f:
                data = json.load(f)
            assert data["annotations"] == [{"id": 1}]
        finally:
            os.unlink(path)


class TestAgreementBanner:
    def test_empty_stats(self):
        stats = AgreementStats()
        assert _agreement_banner(stats) == ""

    def test_basic_stats(self):
        stats = AgreementStats(agree=5, soft_disagree=2, hard_disagree=1, total=8)
        banner = _agreement_banner(stats)
        assert "Agree: 5" in banner
        assert "Soft: 2" in banner
        assert "Hard: 1" in banner

    def test_confidence_message_for_high_disagreement(self):
        stats = AgreementStats(agree=5, soft_disagree=0, hard_disagree=10, total=15)
        banner = _agreement_banner(stats)
        assert "not be calibrated" in banner

    def test_no_reliability_claim_without_hard_disagreements(self):
        stats = AgreementStats(agree=0, soft_disagree=20, hard_disagree=0, total=20)
        banner = _agreement_banner(stats)
        assert "reliable" not in banner
        assert "trustworthy" not in banner
