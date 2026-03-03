"""Tests for the results viewer data layer."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from ocr_bench.viewer import _filter_comparisons, _winner_badge, load_results

SAMPLE_LEADERBOARD = [
    {"model": "DeepSeek-OCR", "elo": 1539, "wins": 5, "losses": 2, "ties": 1, "win_pct": 63},
    {"model": "LightOnOCR-2", "elo": 1530, "wins": 4, "losses": 3, "ties": 1, "win_pct": 50},
]

SAMPLE_COMPARISONS = [
    {
        "sample_idx": 0,
        "model_a": "DeepSeek-OCR",
        "model_b": "LightOnOCR-2",
        "winner": "A",
        "reason": "more complete",
        "agreement": "2/2",
        "text_a": "OCR text from DeepSeek",
        "text_b": "OCR text from LightOn",
        "col_a": "ocr_deepseek",
        "col_b": "ocr_lighton",
    },
    {
        "sample_idx": 1,
        "model_a": "DeepSeek-OCR",
        "model_b": "LightOnOCR-2",
        "winner": "tie",
        "reason": "similar quality",
        "agreement": "2/2",
        "text_a": "Same text A",
        "text_b": "Same text B",
        "col_a": "ocr_deepseek",
        "col_b": "ocr_lighton",
    },
    {
        "sample_idx": 2,
        "model_a": "DeepSeek-OCR",
        "model_b": "LightOnOCR-2",
        "winner": "B",
        "reason": "better formatting",
        "agreement": "1/1",
        "text_a": "Text A sample 2",
        "text_b": "Text B sample 2",
        "col_a": "ocr_deepseek",
        "col_b": "ocr_lighton",
    },
]


class TestLoadResults:
    @patch("ocr_bench.viewer.load_dataset")
    def test_returns_leaderboard_and_comparisons(self, mock_load):
        mock_lb_ds = MagicMock()
        mock_lb_ds.__iter__ = MagicMock(return_value=iter(SAMPLE_LEADERBOARD))
        mock_comp_ds = MagicMock()
        mock_comp_ds.__iter__ = MagicMock(return_value=iter(SAMPLE_COMPARISONS))

        def side_effect(repo_id, name, split):
            if name == "leaderboard":
                return mock_lb_ds
            return mock_comp_ds

        mock_load.side_effect = side_effect

        lb, comps = load_results("user/results")
        assert len(lb) == 2
        assert lb[0]["model"] == "DeepSeek-OCR"
        assert len(comps) == 3
        assert comps[0]["text_a"] == "OCR text from DeepSeek"


class TestFilterComparisons:
    def test_no_filters(self):
        filtered = _filter_comparisons(SAMPLE_COMPARISONS, "All", "All")
        assert len(filtered) == 3

    def test_filter_by_winner_a(self):
        filtered = _filter_comparisons(SAMPLE_COMPARISONS, "A", "All")
        assert len(filtered) == 1
        assert filtered[0]["winner"] == "A"

    def test_filter_by_winner_tie(self):
        filtered = _filter_comparisons(SAMPLE_COMPARISONS, "tie", "All")
        assert len(filtered) == 1
        assert filtered[0]["winner"] == "tie"

    def test_filter_by_winner_b(self):
        filtered = _filter_comparisons(SAMPLE_COMPARISONS, "B", "All")
        assert len(filtered) == 1

    def test_filter_by_model(self):
        filtered = _filter_comparisons(SAMPLE_COMPARISONS, "All", "DeepSeek-OCR")
        assert len(filtered) == 3  # DeepSeek is model_a in all

    def test_combined_filters(self):
        filtered = _filter_comparisons(SAMPLE_COMPARISONS, "A", "DeepSeek-OCR")
        assert len(filtered) == 1

    def test_no_matches(self):
        filtered = _filter_comparisons(SAMPLE_COMPARISONS, "A", "NonexistentModel")
        assert len(filtered) == 0


class TestWinnerBadge:
    def test_winner_a(self):
        assert _winner_badge("A") == "Winner: A"

    def test_winner_b(self):
        assert _winner_badge("B") == "Winner: B"

    def test_tie(self):
        assert _winner_badge("tie") == "Tie"
