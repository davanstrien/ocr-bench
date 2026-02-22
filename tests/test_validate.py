"""Tests for ocr_bench.validate — agreement stats, comparison ordering, persistence."""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from ocr_bench.validate import (
    AgreementStats,
    _agreement_banner,
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
        assert stats.agreement_rate == pytest.approx(0.9)

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

    def test_confidence_message_for_low_disagreement(self):
        stats = AgreementStats(agree=14, soft_disagree=0, hard_disagree=1, total=15)
        banner = _agreement_banner(stats)
        assert "Very few" in banner


class TestBuildPairSummary:
    """Test the viewer pair summary helper (imported from validate for testing)."""

    def test_basic_summary(self):
        from ocr_bench.viewer import _build_pair_summary

        comparisons = [
            {"model_a": "A", "model_b": "B", "winner": "A"},
            {"model_a": "A", "model_b": "B", "winner": "A"},
            {"model_a": "A", "model_b": "B", "winner": "B"},
            {"model_a": "A", "model_b": "B", "winner": "tie"},
        ]
        summary = _build_pair_summary(comparisons)
        assert "2W" in summary
        assert "1L" in summary
        assert "1T" in summary

    def test_empty_comparisons(self):
        from ocr_bench.viewer import _build_pair_summary

        assert _build_pair_summary([]) == ""


class TestModelLabel:
    def test_with_col(self):
        from ocr_bench.viewer import _model_label

        assert _model_label("model-a", "col_a") == "model-a (col_a)"

    def test_without_col(self):
        from ocr_bench.viewer import _model_label

        assert _model_label("model-a", "") == "model-a"
