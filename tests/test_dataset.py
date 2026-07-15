"""Tests for dataset loading and OCR column discovery."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from datasets import Dataset

from ocr_bench.dataset import (
    DatasetError,
    LoadedConfig,
    _find_text_column,
    check_config_alignment,
    discover_configs,
    discover_ocr_columns,
    discover_pr_configs,
    find_alignment_mismatch,
    load_config_dataset,
    load_flat_dataset,
    shared_alignment_keys,
)

# ---------------------------------------------------------------------------
# discover_ocr_columns
# ---------------------------------------------------------------------------


class TestDiscoverOcrColumns:
    def test_inference_info_single_entry(self):
        info = json.dumps({"column_name": "markdown_col", "model_id": "org/model-a"})
        ds = Dataset.from_dict(
            {
                "image": [None],
                "markdown_col": ["text"],
                "inference_info": [info],
            }
        )
        result = discover_ocr_columns(ds)
        assert result == {"markdown_col": "org/model-a"}

    def test_inference_info_list(self):
        info = json.dumps(
            [
                {"column_name": "col_a", "model_id": "model-a"},
                {"column_name": "col_b", "model_id": "model-b"},
            ]
        )
        ds = Dataset.from_dict(
            {
                "image": [None],
                "col_a": ["text a"],
                "col_b": ["text b"],
                "inference_info": [info],
            }
        )
        result = discover_ocr_columns(ds)
        assert result == {"col_a": "model-a", "col_b": "model-b"}

    def test_inference_info_fallback_to_model_name(self):
        info = json.dumps({"column_name": "ocr_out", "model_name": "fallback-model"})
        ds = Dataset.from_dict({"image": [None], "ocr_out": ["text"], "inference_info": [info]})
        result = discover_ocr_columns(ds)
        assert result == {"ocr_out": "fallback-model"}

    def test_heuristic_fallback(self):
        ds = Dataset.from_dict(
            {
                "image": [None],
                "markdown_output": ["text"],
                "other": ["data"],
            }
        )
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
        ds = Dataset.from_dict(
            {
                "image": [None],
                "ocr_col": ["text"],
                "inference_info": ["not-json"],
            }
        )
        result = discover_ocr_columns(ds)
        assert "ocr_col" in result

    def test_disambiguates_duplicate_models(self):
        info = json.dumps(
            [
                {"column_name": "col_a", "model_id": "org/same-model"},
                {"column_name": "col_b", "model_id": "org/same-model"},
            ]
        )
        ds = Dataset.from_dict(
            {
                "image": [None],
                "col_a": ["a"],
                "col_b": ["b"],
                "inference_info": [info],
            }
        )
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
        ds = Dataset.from_dict(
            {
                "image": [None],
                "text": ["tesseract baseline"],
                "markdown": ["model output"],
            }
        )
        assert _find_text_column(ds) == "markdown"

    def test_prefers_inference_info_column_name(self):
        """inference_info column_name should take highest priority."""
        info = json.dumps({"column_name": "markdown", "model_id": "model-a"})
        ds = Dataset.from_dict(
            {
                "image": [None],
                "text": ["baseline"],
                "markdown": ["model output"],
                "inference_info": [info],
            }
        )
        assert _find_text_column(ds) == "markdown"

    def test_inference_info_missing_column_falls_back(self):
        """If inference_info references a missing column, fall back to heuristic."""
        info = json.dumps({"column_name": "nonexistent", "model_id": "model-a"})
        ds = Dataset.from_dict(
            {
                "image": [None],
                "ocr_output": ["text"],
                "inference_info": [info],
            }
        )
        assert _find_text_column(ds) == "ocr_output"

    def test_prefers_ocr_over_text(self):
        ds = Dataset.from_dict(
            {
                "image": [None],
                "text": ["baseline"],
                "ocr_output": ["model output"],
            }
        )
        assert _find_text_column(ds) == "ocr_output"

    def test_returns_text_as_last_resort(self):
        ds = Dataset.from_dict({"image": [None], "text": ["hello"]})
        assert _find_text_column(ds) == "text"

    def test_returns_none_when_no_match(self):
        ds = Dataset.from_dict({"image": [None], "something_else": ["data"]})
        assert _find_text_column(ds) is None

    def test_inference_info_list_format(self):
        info = json.dumps([{"column_name": "markdown", "model_id": "model-a"}])
        ds = Dataset.from_dict(
            {
                "image": [None],
                "text": ["baseline"],
                "markdown": ["output"],
                "inference_info": [info],
            }
        )
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
# discover_configs (main branch)
# ---------------------------------------------------------------------------


class TestDiscoverConfigs:
    @patch("ocr_bench.dataset.get_dataset_config_names")
    def test_returns_non_default_configs(self, mock_get):
        mock_get.return_value = ["default", "model_a", "model_b"]
        result = discover_configs("repo/id")
        assert result == ["model_a", "model_b"]
        mock_get.assert_called_once_with("repo/id")

    @patch("ocr_bench.dataset.get_dataset_config_names")
    def test_returns_empty_when_only_default(self, mock_get):
        mock_get.return_value = ["default"]
        result = discover_configs("repo/id")
        assert result == []

    @patch("ocr_bench.dataset.get_dataset_config_names")
    def test_returns_empty_on_error(self, mock_get):
        mock_get.side_effect = Exception("repo not found")
        result = discover_configs("repo/id")
        assert result == []

    @patch("ocr_bench.dataset.get_dataset_config_names")
    def test_returns_all_when_no_default(self, mock_get):
        mock_get.return_value = ["config_a", "config_b"]
        result = discover_configs("repo/id")
        assert result == ["config_a", "config_b"]


# ---------------------------------------------------------------------------
# load_config_dataset
# ---------------------------------------------------------------------------


class TestLoadConfigDataset:
    @patch("ocr_bench.dataset.load_dataset")
    def test_merges_two_configs(self, mock_load):
        ds_a = Dataset.from_dict(
            {
                "image": [None, None],
                "markdown": ["text_a1", "text_a2"],
                "inference_info": [
                    json.dumps({"model_id": "model-a"}),
                    json.dumps({"model_id": "model-a"}),
                ],
            }
        )
        ds_b = Dataset.from_dict(
            {
                "image": [None, None],
                "markdown": ["text_b1", "text_b2"],
                "inference_info": [
                    json.dumps({"model_id": "model-b"}),
                    json.dumps({"model_id": "model-b"}),
                ],
            }
        )
        mock_load.side_effect = [ds_a, ds_b]

        ds, ocr_cols = load_config_dataset("repo/id", ["cfg_a", "cfg_b"])
        assert set(ocr_cols.keys()) == {"cfg_a", "cfg_b"}
        assert ocr_cols["cfg_a"] == "model-a"
        assert ocr_cols["cfg_b"] == "model-b"
        assert len(ds) == 2
        assert ds[0]["cfg_a"] == "text_a1"
        assert ds[0]["cfg_b"] == "text_b1"

    @patch("ocr_bench.dataset.load_dataset")
    def test_disambiguates_same_model_id_configs(self, mock_load):
        # Two configs = same model, different run settings -> must stay distinct,
        # not collapse to one leaderboard row.
        ds_a = Dataset.from_dict(
            {
                "image": [None],
                "markdown": ["a"],
                "inference_info": [json.dumps({"model_id": "numind/NuExtract3"})],
            }
        )
        ds_b = Dataset.from_dict(
            {
                "image": [None],
                "markdown": ["b"],
                "inference_info": [json.dumps({"model_id": "numind/NuExtract3"})],
            }
        )
        mock_load.side_effect = [ds_a, ds_b]

        _, ocr_cols = load_config_dataset("repo/id", ["nuextract3", "nuextract3-rep"])
        assert ocr_cols["nuextract3"] == "NuExtract3 (nuextract3)"
        assert ocr_cols["nuextract3-rep"] == "NuExtract3 (nuextract3-rep)"
        # distinct values -> downstream model set won't collapse them
        assert len(set(ocr_cols.values())) == 2

    @patch("ocr_bench.dataset.load_dataset")
    def test_duplicate_warning_is_actionable(self, mock_load):
        # The collision warning must name the repo, the duplicated model_id, and
        # the colliding config -> model_id mapping so it's diagnosable.
        import structlog

        ds_a = Dataset.from_dict(
            {
                "image": [None],
                "markdown": ["a"],
                "inference_info": [json.dumps({"model_id": "numind/NuExtract3"})],
            }
        )
        ds_b = Dataset.from_dict(
            {
                "image": [None],
                "markdown": ["b"],
                "inference_info": [json.dumps({"model_id": "numind/NuExtract3"})],
            }
        )
        mock_load.side_effect = [ds_a, ds_b]

        with structlog.testing.capture_logs() as logs:
            load_config_dataset("org/repo", ["nuextract3", "nuextract3-rep"])

        warns = [e for e in logs if e.get("event") == "duplicate_model_ids"]
        assert warns, "expected a duplicate_model_ids warning"
        w = warns[0]
        assert w["repo_id"] == "org/repo"
        assert w["model_ids"] == ["numind/NuExtract3"]
        assert w["collided_configs"] == {
            "nuextract3": "numind/NuExtract3",
            "nuextract3-rep": "numind/NuExtract3",
        }

    @patch("ocr_bench.dataset.load_dataset")
    def test_unique_model_ids_keep_bare_id(self, mock_load):
        ds_a = Dataset.from_dict(
            {
                "image": [None],
                "markdown": ["a"],
                "inference_info": [json.dumps({"model_id": "org/a"})],
            }
        )
        ds_b = Dataset.from_dict(
            {
                "image": [None],
                "markdown": ["b"],
                "inference_info": [json.dumps({"model_id": "org/b"})],
            }
        )
        mock_load.side_effect = [ds_a, ds_b]

        _, ocr_cols = load_config_dataset("repo/id", ["cfg_a", "cfg_b"])
        assert ocr_cols == {"cfg_a": "org/a", "cfg_b": "org/b"}

    @patch("ocr_bench.dataset.load_dataset")
    def test_uses_pr_revision(self, mock_load):
        ds = Dataset.from_dict({"image": [None], "markdown": ["text"]})
        mock_load.return_value = ds

        load_config_dataset("repo/id", ["cfg"], pr_revisions={"cfg": "refs/pr/1"})
        mock_load.assert_called_once_with(
            path="repo/id", name="cfg", split="train", revision="refs/pr/1"
        )

    def test_empty_configs_raises(self):
        with pytest.raises(DatasetError, match="No config names"):
            load_config_dataset("repo/id", [])


# ---------------------------------------------------------------------------
# Cross-config alignment (issue #5)
# ---------------------------------------------------------------------------


def _config_ds(model_id: str, texts: list[str], **passthrough) -> Dataset:
    data = {
        "image": [None] * len(texts),
        "markdown": texts,
        "inference_info": [json.dumps({"model_id": model_id})] * len(texts),
    }
    data.update(passthrough)
    return Dataset.from_dict(data)


class TestFindAlignmentMismatch:
    def test_no_mismatch(self):
        a = Dataset.from_dict({"b_number": [1, 2, 3]})
        b = Dataset.from_dict({"b_number": [1, 2, 3]})
        assert find_alignment_mismatch(a, b, ["b_number"]) is None

    def test_mid_sequence_swap(self):
        a = Dataset.from_dict({"b_number": [1, 2, 3]})
        b = Dataset.from_dict({"b_number": [1, 3, 2]})
        assert find_alignment_mismatch(a, b, ["b_number"]) == ("b_number", 1)

    def test_length_shortfall_is_mismatch(self):
        a = Dataset.from_dict({"b_number": [1, 2, 3]})
        b = Dataset.from_dict({"b_number": [1, 2]})
        assert find_alignment_mismatch(a, b, ["b_number"]) == ("b_number", 2)

    def test_shared_keys_intersection(self):
        a = Dataset.from_dict({"b_number": [1], "id": ["x"]})
        b = Dataset.from_dict({"b_number": [1], "page_index": [0]})
        assert shared_alignment_keys(a, b) == ["b_number"]


class TestCheckConfigAlignment:
    def _lc(self, name: str, ds: Dataset) -> LoadedConfig:
        return LoadedConfig(config=name, model_id="m", ds=ds, text_col="markdown")

    def test_single_config_is_na(self):
        result = check_config_alignment([self._lc("a", _config_ds("m", ["t"], b_number=[1]))])
        assert result.status == "n/a"

    def test_aligned_is_ok(self):
        loaded = [
            self._lc("a", _config_ds("m-a", ["a1", "a2"], b_number=[1, 2])),
            self._lc("b", _config_ds("m-b", ["b1", "b2"], b_number=[1, 2])),
        ]
        result = check_config_alignment(loaded)
        assert result.status == "ok"
        assert result.shared_keys == ["b_number"]

    def test_misaligned(self):
        loaded = [
            self._lc("a", _config_ds("m-a", ["a1", "a2", "a3"], b_number=[1, 2, 3])),
            self._lc("b", _config_ds("m-b", ["b1", "b2", "b3"], b_number=[1, 3, 2])),
        ]
        result = check_config_alignment(loaded)
        assert result.status == "misaligned"
        assert result.config == "b"
        assert result.reference_config == "a"
        assert result.index == 1
        assert result.column == "b_number"

    def test_no_shared_columns_is_unverified(self):
        loaded = [
            self._lc("a", _config_ds("m-a", ["a1", "a2"])),
            self._lc("b", _config_ds("m-b", ["b1", "b2"])),
        ]
        assert check_config_alignment(loaded).status == "unverified"

    def test_constant_shared_key_is_unverified(self):
        # A document-level identifier can match after rows are permuted, so it
        # cannot prove row-for-row alignment by itself.
        loaded = [
            self._lc("a", _config_ds("m-a", ["a1", "a2"], b_number=[7, 7])),
            self._lc("b", _config_ds("m-b", ["b2", "b1"], b_number=[7, 7])),
        ]
        assert check_config_alignment(loaded).status == "unverified"

    def test_null_shared_key_is_unverified(self):
        loaded = [
            self._lc("a", _config_ds("m-a", ["a1", "a2"], id=[None, None])),
            self._lc("b", _config_ds("m-b", ["b2", "b1"], id=[None, None])),
        ]
        assert check_config_alignment(loaded).status == "unverified"

    def test_combined_keys_can_identify_rows(self):
        loaded = [
            self._lc(
                "a",
                _config_ds("m-a", ["a1", "a2"], b_number=[7, 7], page_index=[0, 1]),
            ),
            self._lc(
                "b",
                _config_ds("m-b", ["b1", "b2"], b_number=[7, 7], page_index=[0, 1]),
            ),
        ]
        assert check_config_alignment(loaded).status == "ok"

    def test_partial_when_some_configs_share_no_keys(self):
        # One sibling verifies, another shares nothing — the whole set must NOT
        # read as "ok" off the single passing config (review finding #1).
        loaded = [
            self._lc("a", _config_ds("m-a", ["a1", "a2"], b_number=[1, 2])),
            self._lc("b", _config_ds("m-b", ["b1", "b2"], b_number=[1, 2])),
            self._lc("c", _config_ds("m-c", ["c1", "c2"])),
        ]
        result = check_config_alignment(loaded)
        assert result.status == "partial"
        assert result.verified_configs == ["b"]
        assert result.unverified_configs == ["c"]

    def test_config_status_reports_per_config(self):
        loaded = [
            self._lc("a", _config_ds("m-a", ["a1", "a2"], b_number=[1, 2])),
            self._lc("b", _config_ds("m-b", ["b1", "b2"], b_number=[1, 2])),
            self._lc("c", _config_ds("m-c", ["c1", "c2"])),
        ]
        result = check_config_alignment(loaded)
        assert result.config_status("a") == "reference"
        assert result.config_status("b") == "ok"
        assert result.config_status("c") == "unverified"


class TestLoadConfigDatasetAlignment:
    @patch("ocr_bench.dataset.load_dataset")
    def test_aligned_configs_merge(self, mock_load):
        mock_load.side_effect = [
            _config_ds("m-a", ["a1", "a2"], b_number=[10, 20]),
            _config_ds("m-b", ["b1", "b2"], b_number=[10, 20]),
        ]
        ds, cols = load_config_dataset("repo/id", ["cfg_a", "cfg_b"])
        assert len(ds) == 2
        assert cols == {"cfg_a": "m-a", "cfg_b": "m-b"}
        assert ds[1]["cfg_a"] == "a2"
        assert ds[1]["cfg_b"] == "b2"

    @patch("ocr_bench.dataset.load_dataset")
    def test_misaligned_configs_raise(self, mock_load):
        # Rows 1 and 2 swapped in the second config → row-1 mismatch.
        mock_load.side_effect = [
            _config_ds("m-a", ["a1", "a2", "a3"], b_number=[10, 20, 30]),
            _config_ds("m-b", ["b1", "b2", "b3"], b_number=[10, 30, 20]),
        ]
        with pytest.raises(DatasetError, match=r"row 1"):
            load_config_dataset("repo/id", ["cfg_a", "cfg_b"])

    @patch("ocr_bench.dataset.load_dataset")
    def test_misaligned_error_names_config(self, mock_load):
        mock_load.side_effect = [
            _config_ds("m-a", ["a1", "a2"], b_number=[10, 20]),
            _config_ds("m-b", ["b1", "b2"], b_number=[10, 99]),
        ]
        with pytest.raises(DatasetError, match=r"cfg_b"):
            load_config_dataset("repo/id", ["cfg_a", "cfg_b"])

    @patch("ocr_bench.dataset.load_dataset")
    def test_no_shared_columns_warns(self, mock_load):
        import structlog

        mock_load.side_effect = [
            _config_ds("m-a", ["a1", "a2"]),
            _config_ds("m-b", ["b1", "b2"]),
        ]
        with structlog.testing.capture_logs() as logs:
            load_config_dataset("repo/id", ["cfg_a", "cfg_b"])
        assert any(e.get("event") == "alignment_unverified" for e in logs)

    @patch("ocr_bench.dataset.load_dataset")
    def test_row_count_mismatch_raises(self, mock_load):
        # Differing lengths must fail loudly, not silently truncate (review #2).
        mock_load.side_effect = [
            _config_ds("m-a", ["a1", "a2", "a3"], b_number=[10, 20, 30]),
            _config_ds("m-b", ["b1", "b2"], b_number=[10, 20]),
        ]
        with pytest.raises(DatasetError, match=r"[Rr]ow-count mismatch"):
            load_config_dataset("repo/id", ["cfg_a", "cfg_b"])

    @patch("ocr_bench.dataset.load_dataset")
    def test_partial_alignment_warns_not_raises(self, mock_load):
        import structlog

        # cfg_b shares b_number (verified); cfg_c shares nothing (unverified).
        mock_load.side_effect = [
            _config_ds("m-a", ["a1", "a2"], b_number=[10, 20]),
            _config_ds("m-b", ["b1", "b2"], b_number=[10, 20]),
            _config_ds("m-c", ["c1", "c2"]),
        ]
        with structlog.testing.capture_logs() as logs:
            _, cols = load_config_dataset("repo/id", ["cfg_a", "cfg_b", "cfg_c"])
        assert len(cols) == 3  # merge still proceeds
        warns = [e for e in logs if e.get("event") == "alignment_unverified"]
        assert warns and warns[0]["status"] == "partial"
        assert warns[0]["unverified_configs"] == ["cfg_c"]


# ---------------------------------------------------------------------------
# load_flat_dataset
# ---------------------------------------------------------------------------


class TestLoadFlatDataset:
    @patch("ocr_bench.dataset.load_dataset")
    def test_auto_discover(self, mock_load):
        info = json.dumps({"column_name": "ocr_out", "model_id": "model-x"})
        ds = Dataset.from_dict(
            {
                "image": [None],
                "ocr_out": ["text"],
                "inference_info": [info],
            }
        )
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
