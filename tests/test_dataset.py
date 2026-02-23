"""Tests for dataset loading and OCR column discovery."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from datasets import Dataset

from ocr_bench.dataset import (
    DatasetError,
    _find_text_column,
    discover_ocr_columns,
    discover_pr_configs,
    load_config_dataset,
    load_flat_dataset,
)

# ---------------------------------------------------------------------------
# discover_ocr_columns
# ---------------------------------------------------------------------------


class TestDiscoverOcrColumns:
    def test_inference_info_single_entry(self):
        info = json.dumps({"column_name": "markdown_col", "model_id": "org/model-a"})
        ds = Dataset.from_dict({
            "image": [None], "markdown_col": ["text"], "inference_info": [info],
        })
        result = discover_ocr_columns(ds)
        assert result == {"markdown_col": "org/model-a"}

    def test_inference_info_list(self):
        info = json.dumps([
            {"column_name": "col_a", "model_id": "model-a"},
            {"column_name": "col_b", "model_id": "model-b"},
        ])
        ds = Dataset.from_dict({
            "image": [None],
            "col_a": ["text a"],
            "col_b": ["text b"],
            "inference_info": [info],
        })
        result = discover_ocr_columns(ds)
        assert result == {"col_a": "model-a", "col_b": "model-b"}

    def test_inference_info_fallback_to_model_name(self):
        info = json.dumps({"column_name": "ocr_out", "model_name": "fallback-model"})
        ds = Dataset.from_dict({"image": [None], "ocr_out": ["text"], "inference_info": [info]})
        result = discover_ocr_columns(ds)
        assert result == {"ocr_out": "fallback-model"}

    def test_heuristic_fallback(self):
        ds = Dataset.from_dict({
            "image": [None],
            "markdown_output": ["text"],
            "other": ["data"],
        })
        result = discover_ocr_columns(ds)
        assert "markdown_output" in result
        assert "other" not in result

    def test_heuristic_text_column(self):
        ds = Dataset.from_dict({"image": [None], "text": ["hello"]})
        result = discover_ocr_columns(ds)
        assert result == {"text": "text"}

    def test_no_columns_raises(self):
        ds = Dataset.from_dict({"image": [None], "something_else": ["data"]})
        with pytest.raises(DatasetError, match="No OCR columns"):
            discover_ocr_columns(ds)

    def test_malformed_json_falls_back(self):
        ds = Dataset.from_dict({
            "image": [None],
            "ocr_col": ["text"],
            "inference_info": ["not-json"],
        })
        result = discover_ocr_columns(ds)
        assert "ocr_col" in result

    def test_disambiguates_duplicate_models(self):
        info = json.dumps([
            {"column_name": "col_a", "model_id": "org/same-model"},
            {"column_name": "col_b", "model_id": "org/same-model"},
        ])
        ds = Dataset.from_dict({
            "image": [None],
            "col_a": ["a"],
            "col_b": ["b"],
            "inference_info": [info],
        })
        result = discover_ocr_columns(ds)
        assert result["col_a"] == "same-model (col_a)"
        assert result["col_b"] == "same-model (col_b)"

    def test_column_not_in_dataset_skipped(self):
        """inference_info references a column that doesn't exist — skip it."""
        info = json.dumps({"column_name": "missing_col", "model_id": "model-a"})
        ds = Dataset.from_dict({"image": [None], "ocr_output": ["text"], "inference_info": [info]})
        result = discover_ocr_columns(ds)
        # Falls back to heuristic since inference_info column didn't match
        assert "ocr_output" in result


# ---------------------------------------------------------------------------
# _find_text_column
# ---------------------------------------------------------------------------


class TestFindTextColumn:
    def test_prefers_markdown_over_text(self):
        """BPL bug: text appears before markdown in column order, should pick markdown."""
        ds = Dataset.from_dict({
            "image": [None],
            "text": ["tesseract baseline"],
            "markdown": ["model output"],
        })
        assert _find_text_column(ds) == "markdown"

    def test_prefers_inference_info_column_name(self):
        """inference_info column_name should take highest priority."""
        info = json.dumps({"column_name": "markdown", "model_id": "model-a"})
        ds = Dataset.from_dict({
            "image": [None],
            "text": ["baseline"],
            "markdown": ["model output"],
            "inference_info": [info],
        })
        assert _find_text_column(ds) == "markdown"

    def test_inference_info_missing_column_falls_back(self):
        """If inference_info references a missing column, fall back to heuristic."""
        info = json.dumps({"column_name": "nonexistent", "model_id": "model-a"})
        ds = Dataset.from_dict({
            "image": [None],
            "ocr_output": ["text"],
            "inference_info": [info],
        })
        assert _find_text_column(ds) == "ocr_output"

    def test_prefers_ocr_over_text(self):
        ds = Dataset.from_dict({
            "image": [None],
            "text": ["baseline"],
            "ocr_output": ["model output"],
        })
        assert _find_text_column(ds) == "ocr_output"

    def test_returns_text_as_last_resort(self):
        ds = Dataset.from_dict({"image": [None], "text": ["hello"]})
        assert _find_text_column(ds) == "text"

    def test_returns_none_when_no_match(self):
        ds = Dataset.from_dict({"image": [None], "something_else": ["data"]})
        assert _find_text_column(ds) is None

    def test_inference_info_list_format(self):
        info = json.dumps([{"column_name": "markdown", "model_id": "model-a"}])
        ds = Dataset.from_dict({
            "image": [None],
            "text": ["baseline"],
            "markdown": ["output"],
            "inference_info": [info],
        })
        assert _find_text_column(ds) == "markdown"


# ---------------------------------------------------------------------------
# discover_pr_configs
# ---------------------------------------------------------------------------


class TestDiscoverPrConfigs:
    def _make_disc(self, num, title, is_pr=True, status="open"):
        d = MagicMock()
        d.num = num
        d.title = title
        d.is_pull_request = is_pr
        d.status = status
        return d

    def test_extracts_config_from_bracket_title(self):
        api = MagicMock()
        api.get_repo_discussions.return_value = [
            self._make_disc(1, "Add model output [config_a]"),
        ]
        names, revisions = discover_pr_configs("repo/id", api=api)
        assert names == ["config_a"]
        assert revisions == {"config_a": "refs/pr/1"}

    def test_skips_pr_without_brackets(self):
        api = MagicMock()
        api.get_repo_discussions.return_value = [
            self._make_disc(1, "Just a normal PR"),
        ]
        names, _ = discover_pr_configs("repo/id", api=api)
        assert names == []

    def test_skips_non_pr_discussions(self):
        api = MagicMock()
        api.get_repo_discussions.return_value = [
            self._make_disc(1, "Question [config_a]", is_pr=False),
        ]
        names, _ = discover_pr_configs("repo/id", api=api)
        assert names == []

    def test_skips_closed_prs(self):
        api = MagicMock()
        api.get_repo_discussions.return_value = [
            self._make_disc(1, "Old PR [config_a]", status="closed"),
        ]
        names, _ = discover_pr_configs("repo/id", api=api)
        assert names == []

    def test_merge_mode(self):
        api = MagicMock()
        api.get_repo_discussions.return_value = [
            self._make_disc(1, "Add [config_a]"),
        ]
        names, revisions = discover_pr_configs("repo/id", merge=True, api=api)
        assert names == ["config_a"]
        # When merging, no revisions needed (loaded from main)
        assert revisions == {}
        api.merge_pull_request.assert_called_once_with("repo/id", 1, repo_type="dataset")

    def test_multiple_prs(self):
        api = MagicMock()
        api.get_repo_discussions.return_value = [
            self._make_disc(1, "Add [config_a]"),
            self._make_disc(2, "Add [config_b]"),
            self._make_disc(3, "Not a config PR"),
        ]
        names, revisions = discover_pr_configs("repo/id", api=api)
        assert names == ["config_a", "config_b"]
        assert len(revisions) == 2


# ---------------------------------------------------------------------------
# load_config_dataset
# ---------------------------------------------------------------------------


class TestLoadConfigDataset:
    @patch("ocr_bench.dataset.load_dataset")
    def test_merges_two_configs(self, mock_load):
        ds_a = Dataset.from_dict({
            "image": [None, None],
            "markdown": ["text_a1", "text_a2"],
            "inference_info": [
                json.dumps({"model_id": "model-a"}),
                json.dumps({"model_id": "model-a"}),
            ],
        })
        ds_b = Dataset.from_dict({
            "image": [None, None],
            "markdown": ["text_b1", "text_b2"],
            "inference_info": [
                json.dumps({"model_id": "model-b"}),
                json.dumps({"model_id": "model-b"}),
            ],
        })
        mock_load.side_effect = [ds_a, ds_b]

        ds, ocr_cols = load_config_dataset("repo/id", ["cfg_a", "cfg_b"])
        assert set(ocr_cols.keys()) == {"cfg_a", "cfg_b"}
        assert ocr_cols["cfg_a"] == "model-a"
        assert ocr_cols["cfg_b"] == "model-b"
        assert len(ds) == 2
        assert ds[0]["cfg_a"] == "text_a1"
        assert ds[0]["cfg_b"] == "text_b1"

    @patch("ocr_bench.dataset.load_dataset")
    def test_uses_pr_revision(self, mock_load):
        ds = Dataset.from_dict({"image": [None], "markdown": ["text"]})
        mock_load.return_value = ds

        load_config_dataset(
            "repo/id", ["cfg"], pr_revisions={"cfg": "refs/pr/1"}
        )
        mock_load.assert_called_once_with(
            path="repo/id", name="cfg", split="train", revision="refs/pr/1"
        )

    def test_empty_configs_raises(self):
        with pytest.raises(DatasetError, match="No config names"):
            load_config_dataset("repo/id", [])


# ---------------------------------------------------------------------------
# load_flat_dataset
# ---------------------------------------------------------------------------


class TestLoadFlatDataset:
    @patch("ocr_bench.dataset.load_dataset")
    def test_auto_discover(self, mock_load):
        info = json.dumps({"column_name": "ocr_out", "model_id": "model-x"})
        ds = Dataset.from_dict({
            "image": [None],
            "ocr_out": ["text"],
            "inference_info": [info],
        })
        mock_load.return_value = ds

        result_ds, ocr_cols = load_flat_dataset("repo/id")
        assert ocr_cols == {"ocr_out": "model-x"}
        assert len(result_ds) == 1

    @patch("ocr_bench.dataset.load_dataset")
    def test_explicit_columns(self, mock_load):
        ds = Dataset.from_dict({"image": [None], "my_col": ["text"]})
        mock_load.return_value = ds

        _, ocr_cols = load_flat_dataset("repo/id", columns=["my_col"])
        assert ocr_cols == {"my_col": "my_col"}

    @patch("ocr_bench.dataset.load_dataset")
    def test_invalid_column_raises(self, mock_load):
        ds = Dataset.from_dict({"image": [None], "real_col": ["text"]})
        mock_load.return_value = ds

        with pytest.raises(DatasetError, match="Column 'missing' not found"):
            load_flat_dataset("repo/id", columns=["missing"])
