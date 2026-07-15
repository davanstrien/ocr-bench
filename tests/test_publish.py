"""Tests for Hub publishing."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from ocr_bench.elo import ComparisonResult, Leaderboard
from ocr_bench.publish import (
    EvalMetadata,
    _align_metadata_rows,
    build_leaderboard_rows,
    build_metadata_row,
    load_existing_comparisons,
    load_existing_metadata,
    publish_checkpoint,
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

    def test_failed_model_is_unranked_row_without_elo(self):
        board = _make_board()
        rows = build_leaderboard_rows(
            board,
            failed_models=["model-b"],
            failed_outputs={"model-b": 50},
        )
        assert [row["model"] for row in rows] == ["model-a", "model-c", "model-b"]
        failed = rows[-1]
        assert failed["status"] == "failed"
        assert failed["elo"] is None
        assert failed["win_pct"] is None
        assert failed["failed_outputs"] == 50

    def test_partial_failure_remains_ranked_but_degraded(self):
        rows = build_leaderboard_rows(_make_board(), failed_outputs={"model-a": 1})
        model_a = next(row for row in rows if row["model"] == "model-a")
        assert model_a["status"] == "degraded"
        assert model_a["elo"] == 1550
        assert model_a["failed_outputs"] == 1


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

    def test_budget_fields_default(self):
        meta = EvalMetadata(
            source_dataset="repo/data",
            judge_models=[],
            seed=42,
            max_samples=0,
            total_comparisons=0,
            valid_comparisons=0,
        )
        row = build_metadata_row(meta)
        assert row["max_comparisons"] is None
        assert row["budget_exhausted"] is False

    def test_budget_fields_recorded(self):
        meta = EvalMetadata(
            source_dataset="repo/data",
            judge_models=["j"],
            seed=42,
            max_samples=10,
            total_comparisons=12,
            valid_comparisons=12,
            max_comparisons=12,
            budget_exhausted=True,
        )
        row = build_metadata_row(meta)
        assert row["max_comparisons"] == 12
        assert row["budget_exhausted"] is True

    def test_auto_tied_defaults_zero(self):
        meta = EvalMetadata(
            source_dataset="repo/data",
            judge_models=[],
            seed=42,
            max_samples=0,
            total_comparisons=0,
            valid_comparisons=0,
        )
        assert build_metadata_row(meta)["auto_tied"] == 0

    def test_auto_tied_recorded(self):
        meta = EvalMetadata(
            source_dataset="repo/data",
            judge_models=["j"],
            seed=42,
            max_samples=10,
            total_comparisons=8,
            valid_comparisons=8,
            auto_tied=2,
        )
        assert build_metadata_row(meta)["auto_tied"] == 2

    def test_failed_outputs_serialized(self):
        import json

        meta = EvalMetadata(
            source_dataset="repo/data",
            judge_models=[],
            seed=42,
            max_samples=10,
            total_comparisons=1,
            valid_comparisons=1,
            failed_outputs={"model-x": 5},
        )
        row = build_metadata_row(meta)
        assert json.loads(row["failed_outputs"]) == {"model-x": 5}

    def test_failed_outputs_default_empty(self):
        meta = EvalMetadata(
            source_dataset="repo/data",
            judge_models=[],
            seed=42,
            max_samples=0,
            total_comparisons=0,
            valid_comparisons=0,
        )
        row = build_metadata_row(meta)
        assert row["failed_outputs"] == "{}"
        assert row["failed_models"] == "[]"

    def test_failed_models_serialized(self):
        import json

        meta = EvalMetadata(
            source_dataset="repo/data",
            judge_models=[],
            seed=42,
            max_samples=10,
            total_comparisons=1,
            valid_comparisons=1,
            failed_models=["model-x"],
        )
        assert json.loads(build_metadata_row(meta)["failed_models"]) == ["model-x"]


class TestAlignMetadataRows:
    def test_union_of_keys_filled_with_none(self):
        # Old row lacks the budget columns a newer row carries.
        rows = [
            {"source_dataset": "d", "total_comparisons": 5},
            {"source_dataset": "d", "total_comparisons": 6, "budget_exhausted": True},
        ]
        aligned = _align_metadata_rows(rows)
        expected_keys = {"source_dataset", "total_comparisons", "budget_exhausted"}
        assert all(set(r) == expected_keys for r in aligned)
        # The older row gets a null for the new column rather than dropping it.
        assert aligned[0]["budget_exhausted"] is None
        assert aligned[1]["budget_exhausted"] is True

    def test_empty(self):
        assert _align_metadata_rows([]) == []


class TestPublishCheckpoint:
    @patch("ocr_bench.publish.Dataset")
    def test_pushes_only_comparisons_config(self, mock_ds_cls):
        results = [
            ComparisonResult(sample_idx=0, model_a="a", model_b="b", winner="A"),
            ComparisonResult(sample_idx=1, model_a="a", model_b="b", winner="B"),
        ]
        publish_checkpoint("user/results", results, ["a", "b"])

        mock_ds = mock_ds_cls.from_list.return_value
        # Exactly one push — no leaderboard/metadata/README churn.
        mock_ds.push_to_hub.assert_called_once()
        assert mock_ds.push_to_hub.call_args.kwargs["config_name"] == "comparisons"

    @patch("ocr_bench.publish.Dataset")
    def test_no_push_when_no_results(self, mock_ds_cls):
        publish_checkpoint("user/results", [], ["a", "b"])
        mock_ds_cls.from_list.assert_not_called()


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
    def test_publishes_failed_model_as_status_row(self, mock_ds_cls, mock_api_cls):
        board = _make_board()
        meta = EvalMetadata(
            source_dataset="repo/data",
            judge_models=["j1"],
            seed=42,
            max_samples=10,
            total_comparisons=1,
            valid_comparisons=1,
            failed_outputs={"model-b": 5},
            failed_models=["model-b"],
        )

        publish_results("user/results", board, meta)

        leaderboard_rows = mock_ds_cls.from_list.call_args_list[1].args[0]
        failed = next(row for row in leaderboard_rows if row["model"] == "model-b")
        assert failed["status"] == "failed"
        assert failed["elo"] is None

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

    @patch("ocr_bench.publish.HfApi")
    @patch("ocr_bench.publish.Dataset")
    def test_aligns_auto_tied_across_metadata_rows(self, mock_ds_cls, mock_api_cls):
        board = _make_board()
        meta = EvalMetadata(
            source_dataset="repo/data",
            judge_models=["j2"],
            seed=42,
            max_samples=10,
            total_comparisons=5,
            valid_comparisons=4,
            auto_tied=1,
        )
        # Older row written before auto_tied existed — _align_metadata_rows must
        # give it the key (filled None) so the appended schema stays consistent.
        existing_meta = [{"source_dataset": "repo/data", "judge_models": '["j1"]'}]
        publish_results("user/results", board, meta, existing_metadata=existing_meta)

        meta_rows = mock_ds_cls.from_list.call_args_list[-1].args[0]
        assert meta_rows[0]["auto_tied"] is None  # aligned onto the old row
        assert meta_rows[1]["auto_tied"] == 1  # this run's count


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


class TestBuildReadme:
    def _make_metadata(self) -> EvalMetadata:
        return EvalMetadata(
            source_dataset="user/data",
            judge_models=["org/judge"],
            seed=42,
            max_samples=10,
            total_comparisons=3,
            valid_comparisons=3,
        )

    def test_no_license_by_default(self):
        """The results data embeds source-derived text — the tool must not
        claim a license on the publisher's behalf."""
        from ocr_bench.publish import _build_readme

        board = _make_board()
        rows = build_leaderboard_rows(board)
        readme = _build_readme("user/results", rows, board, self._make_metadata())
        assert "license:" not in readme
        # Frontmatter must still open cleanly with the tags block
        assert readme.startswith("---\ntags:")

    def test_explicit_license_included(self):
        from ocr_bench.publish import _build_readme

        board = _make_board()
        rows = build_leaderboard_rows(board)
        readme = _build_readme(
            "user/results", rows, board, self._make_metadata(), license_id="cc0-1.0"
        )
        assert "license: cc0-1.0" in readme

    def test_pipes_in_model_names_escaped(self):
        from ocr_bench.publish import _build_readme

        board = Leaderboard(
            elo={"weird|name": 1500.0},
            wins={"weird|name": 1},
            losses={"weird|name": 0},
            ties={"weird|name": 0},
        )
        rows = build_leaderboard_rows(board)
        readme = _build_readme("user/results", rows, board, self._make_metadata())
        assert "weird\\|name" in readme
        assert "| weird|name |" not in readme

    def test_comparisons_plain_count_when_no_auto_ties(self):
        """No auto-ties → a single total, no breakdown (avoids "N + 0")."""
        from ocr_bench.publish import _build_readme

        board = _make_board()  # 1 comparison, no "auto" agreement
        rows = build_leaderboard_rows(board)
        readme = _build_readme("user/results", rows, board, self._make_metadata())
        assert "- **Comparisons**: 1" in readme
        assert "auto-tied" not in readme

    def test_comparisons_breakdown_when_auto_ties_present(self):
        """Auto-ties derived from the log (agreement == "auto") are reported
        separately from judged comparisons, resolving the metadata mismatch."""
        from ocr_bench.publish import _build_readme

        board = Leaderboard(
            elo={"model-a": 1500.0, "model-b": 1500.0},
            wins={"model-a": 1, "model-b": 0},
            losses={"model-a": 0, "model-b": 1},
            ties={"model-a": 1, "model-b": 1},
            comparison_log=[
                {"sample_idx": 0, "model_a": "model-a", "model_b": "model-b",
                 "winner": "A", "agreement": "1/1"},
                {"sample_idx": 1, "model_a": "model-a", "model_b": "model-b",
                 "winner": "tie", "agreement": "auto"},
            ],
        )
        rows = build_leaderboard_rows(board)
        readme = _build_readme("user/results", rows, board, self._make_metadata())
        assert "- **Comparisons**: 1 judged + 1 auto-tied (2 total)" in readme

    def _board_two_models(self) -> Leaderboard:
        return Leaderboard(
            elo={"model-a": 1500.0, "model-b": 1400.0},
            wins={"model-a": 1, "model-b": 0},
            losses={"model-a": 0, "model-b": 1},
            ties={"model-a": 0, "model-b": 0},
        )

    def _metadata_with_failed(self, failed, failed_models=None) -> EvalMetadata:
        return EvalMetadata(
            source_dataset="user/data",
            judge_models=["org/judge"],
            seed=42,
            max_samples=10,
            total_comparisons=1,
            valid_comparisons=1,
            failed_outputs=failed,
            failed_models=failed_models or [],
        )

    def test_failed_outputs_section_and_row_marker(self):
        from ocr_bench.publish import _build_readme

        board = self._board_two_models()
        rows = build_leaderboard_rows(board)
        readme = _build_readme(
            "user/results",
            rows,
            board,
            self._metadata_with_failed({"model-b": 50}, failed_models=["model-b"]),
        )
        assert "## ⚠ Failed outputs" in readme
        assert "| — | model-b ⚠" in readme
        assert "**FAILED**" in readme
        assert "| model-b | 50 |" in readme  # listed in the failures table
        # A flagged run must not be mistaken for a low-quality model.
        assert "excluded from judging" in readme.lower()

    def test_no_failed_section_when_clean(self):
        from ocr_bench.publish import _build_readme

        board = self._board_two_models()
        rows = build_leaderboard_rows(board)
        readme = _build_readme("user/results", rows, board, self._metadata_with_failed({}))
        assert "## ⚠ Failed outputs" not in readme
        assert "⚠" not in readme

    def test_failed_outputs_accepts_json_string(self):
        """A metadata object rebuilt from a stored row carries failed_outputs
        as a JSON string — the card must still render it."""
        import json

        from ocr_bench.publish import _build_readme

        board = self._board_two_models()
        rows = build_leaderboard_rows(board)
        readme = _build_readme(
            "user/results", rows, board, self._metadata_with_failed(json.dumps({"model-b": 7}))
        )
        assert "| model-b | 7 |" in readme
