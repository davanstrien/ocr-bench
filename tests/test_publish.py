"""Tests for Hub publishing."""

from __future__ import annotations

from unittest.mock import patch

from ocr_bench.elo import Leaderboard
from ocr_bench.publish import (
    EvalMetadata,
    build_leaderboard_rows,
    build_metadata_row,
    publish_results,
)


def _make_board() -> Leaderboard:
    return Leaderboard(
        elo={"model-a": 1550.0, "model-b": 1450.0, "model-c": 1500.0},
        wins={"model-a": 3, "model-b": 1, "model-c": 2},
        losses={"model-a": 1, "model-b": 3, "model-c": 2},
        ties={"model-a": 1, "model-b": 1, "model-c": 1},
        comparison_log=[
            {"sample_idx": 0, "model_a": "model-a", "model_b": "model-b", "winner": "A"},
        ],
    )


class TestBuildLeaderboardRows:
    def test_rows_ordered_by_elo(self):
        board = _make_board()
        rows = build_leaderboard_rows(board)
        assert rows[0]["model"] == "model-a"
        assert rows[1]["model"] == "model-c"
        assert rows[2]["model"] == "model-b"

    def test_elo_rounded(self):
        board = Leaderboard(
            elo={"m": 1523.7},
            wins={"m": 1},
            losses={"m": 0},
            ties={"m": 0},
        )
        rows = build_leaderboard_rows(board)
        assert rows[0]["elo"] == 1524

    def test_win_pct(self):
        board = _make_board()
        rows = build_leaderboard_rows(board)
        # model-a: 3 wins / 5 total = 60%
        assert rows[0]["win_pct"] == 60

    def test_zero_games_win_pct(self):
        board = Leaderboard(
            elo={"m": 1500.0},
            wins={"m": 0},
            losses={"m": 0},
            ties={"m": 0},
        )
        rows = build_leaderboard_rows(board)
        assert rows[0]["win_pct"] == 0


class TestBuildMetadataRow:
    def test_auto_timestamp(self):
        meta = EvalMetadata(
            source_dataset="repo/data",
            judge_models=["judge-a"],
            seed=42,
            max_samples=10,
            total_comparisons=30,
            valid_comparisons=28,
        )
        row = build_metadata_row(meta)
        assert row["source_dataset"] == "repo/data"
        assert row["timestamp"]  # auto-filled
        assert '"judge-a"' in row["judge_models"]

    def test_preserved_timestamp(self):
        meta = EvalMetadata(
            source_dataset="repo/data",
            judge_models=["judge-a"],
            seed=42,
            max_samples=10,
            total_comparisons=30,
            valid_comparisons=28,
            timestamp="2026-02-20T12:00:00+00:00",
        )
        row = build_metadata_row(meta)
        assert row["timestamp"] == "2026-02-20T12:00:00+00:00"

    def test_from_prs_default(self):
        meta = EvalMetadata(
            source_dataset="repo/data",
            judge_models=[],
            seed=42,
            max_samples=0,
            total_comparisons=0,
            valid_comparisons=0,
        )
        row = build_metadata_row(meta)
        assert row["from_prs"] is False


class TestPublishResults:
    @patch("ocr_bench.publish.Dataset")
    def test_publishes_three_configs(self, mock_ds_cls):
        mock_ds = mock_ds_cls.from_list.return_value
        board = _make_board()
        meta = EvalMetadata(
            source_dataset="repo/data",
            judge_models=["j1"],
            seed=42,
            max_samples=10,
            total_comparisons=10,
            valid_comparisons=8,
        )
        publish_results("user/results", board, meta)

        # Should call push_to_hub 3 times
        assert mock_ds.push_to_hub.call_count == 3
        config_names = [
            call.kwargs["config_name"]
            for call in mock_ds.push_to_hub.call_args_list
        ]
        assert config_names == ["comparisons", "leaderboard", "metadata"]

    @patch("ocr_bench.publish.Dataset")
    def test_skips_comparisons_if_empty(self, mock_ds_cls):
        mock_ds = mock_ds_cls.from_list.return_value
        board = Leaderboard(
            elo={"m": 1500.0},
            wins={"m": 0},
            losses={"m": 0},
            ties={"m": 0},
            comparison_log=[],
        )
        meta = EvalMetadata(
            source_dataset="repo/data",
            judge_models=["j1"],
            seed=42,
            max_samples=0,
            total_comparisons=0,
            valid_comparisons=0,
        )
        publish_results("user/results", board, meta)

        # Only leaderboard + metadata
        assert mock_ds.push_to_hub.call_count == 2
