"""Tests for Hub publishing."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from ocr_bench.elo import ComparisonResult, Leaderboard
from ocr_bench.publish import (
    EvalMetadata,
    build_leaderboard_rows,
    build_metadata_row,
    load_existing_comparisons,
    load_existing_metadata,
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
    @patch("ocr_bench.publish.HfApi")
    @patch("ocr_bench.publish.Dataset")
    def test_publishes_four_configs(self, mock_ds_cls, mock_api_cls):
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

        # 4 pushes: comparisons, default leaderboard, named leaderboard, metadata
        assert mock_ds.push_to_hub.call_count == 4
        calls = mock_ds.push_to_hub.call_args_list
        # comparisons
        assert calls[0].kwargs["config_name"] == "comparisons"
        # default leaderboard (no config_name kwarg)
        assert calls[1] == (("user/results",),)
        # named leaderboard
        assert calls[2].kwargs["config_name"] == "leaderboard"
        # metadata
        assert calls[3].kwargs["config_name"] == "metadata"
        # README uploaded
        mock_api_cls.return_value.upload_file.assert_called_once()

    @patch("ocr_bench.publish.HfApi")
    @patch("ocr_bench.publish.Dataset")
    def test_skips_comparisons_if_empty(self, mock_ds_cls, mock_api_cls):
        mock_ds_cls.from_list.return_value
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

        # default leaderboard + named leaderboard + metadata = 3
        assert mock_ds_cls.from_list.return_value.push_to_hub.call_count == 3

    @patch("ocr_bench.publish.HfApi")
    @patch("ocr_bench.publish.Dataset")
    def test_appends_existing_metadata(self, mock_ds_cls, mock_api_cls):
        mock_ds_cls.from_list.return_value  # noqa: F841
        board = _make_board()
        meta = EvalMetadata(
            source_dataset="repo/data",
            judge_models=["j2"],
            seed=42,
            max_samples=10,
            total_comparisons=5,
            valid_comparisons=5,
        )
        existing_meta = [{"source_dataset": "repo/data", "judge_models": '["j1"]'}]
        publish_results("user/results", board, meta, existing_metadata=existing_meta)

        # The metadata Dataset.from_list call should have 2 rows
        from_list_calls = mock_ds_cls.from_list.call_args_list
        # Last from_list call is metadata
        meta_rows = from_list_calls[-1].args[0]
        assert len(meta_rows) == 2
        assert meta_rows[0]["source_dataset"] == "repo/data"


class TestLoadExistingComparisons:
    @patch("ocr_bench.publish.load_dataset")
    def test_returns_comparison_results(self, mock_load):
        mock_ds = MagicMock()
        mock_ds.__iter__ = lambda self: iter(
            [
                {
                    "sample_idx": 0,
                    "model_a": "ModelA",
                    "model_b": "ModelB",
                    "winner": "A",
                    "reason": "better",
                    "agreement": "1/1",
                    "text_a": "hello",
                    "text_b": "world",
                    "col_a": "col_a",
                    "col_b": "col_b",
                },
            ]
        )
        mock_load.return_value = mock_ds

        results = load_existing_comparisons("user/results")
        assert len(results) == 1
        assert isinstance(results[0], ComparisonResult)
        assert results[0].model_a == "ModelA"
        assert results[0].winner == "A"
        assert results[0].swapped is False

    @patch("ocr_bench.publish.load_dataset")
    def test_returns_empty_on_missing_repo(self, mock_load):
        mock_load.side_effect = Exception("repo not found")
        results = load_existing_comparisons("nonexistent/repo")
        assert results == []


class TestLoadExistingMetadata:
    @patch("ocr_bench.publish.load_dataset")
    def test_returns_rows(self, mock_load):
        mock_ds = MagicMock()
        mock_ds.__iter__ = lambda self: iter(
            [{"source_dataset": "repo/data", "timestamp": "2026-02-20"}]
        )
        mock_load.return_value = mock_ds

        rows = load_existing_metadata("user/results")
        assert len(rows) == 1
        assert rows[0]["source_dataset"] == "repo/data"

    @patch("ocr_bench.publish.load_dataset")
    def test_returns_empty_on_missing(self, mock_load):
        mock_load.side_effect = Exception("not found")
        rows = load_existing_metadata("nonexistent/repo")
        assert rows == []
