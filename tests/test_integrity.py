"""Tests for the shared input-integrity checks (stats + audit)."""

from __future__ import annotations

import json
from unittest.mock import patch

from datasets import Dataset

from ocr_bench.dataset import LoadedConfig
from ocr_bench.integrity import (
    audit_repo,
    compute_column_stats,
    compute_model_stats,
    failed_output_counts,
)


class TestComputeColumnStats:
    def test_partitions_rows(self):
        texts = [
            "",  # empty
            "   ",  # whitespace-only → empty
            "[OCR ERROR]",  # sentinel
            "short",  # < 20 chars, non-empty, non-sentinel
            "a normal, sufficiently long transcription of the page",  # normal + over-max
        ]
        s = compute_column_stats("cfg", "model-x", texts, max_ocr_text_len=20)
        assert s.n_rows == 5
        assert s.n_empty == 2
        assert s.n_sentinel == 1
        assert s.n_short == 1
        assert s.n_over_max == 1

    def test_rates(self):
        texts = ["", "[OCR ERROR]", "short", "x" * 40]
        s = compute_column_stats("cfg", "m", texts, max_ocr_text_len=20)
        assert s.empty_rate == 0.25
        assert s.sentinel_rate == 0.25
        assert s.short_rate == 0.25
        assert s.over_max_rate == 0.25

    def test_length_stats_over_all_rows(self):
        texts = ["", "ab", "abcd"]  # lengths 0, 2, 4
        s = compute_column_stats("cfg", "m", texts)
        assert s.median_len == 2
        assert s.max_len == 4

    def test_normalized_mode_measures_content_not_html_markup(self):
        html = "<table><tr><td>alpha</td><td>beta</td></tr></table>"
        normalized = compute_column_stats("cfg", "m", [html], max_ocr_text_len=20)
        raw = compute_column_stats(
            "cfg", "m", [html], max_ocr_text_len=20, normalize=False
        )
        assert normalized.max_len == len("alpha | beta")
        assert normalized.n_over_max == 0
        assert raw.max_len == len(html)
        assert raw.n_over_max == 1

    def test_empty_input(self):
        s = compute_column_stats("cfg", "m", [])
        assert s.n_rows == 0
        assert s.empty_rate == 0.0
        assert s.median_len == 0.0
        assert s.max_len == 0
        assert not s.failed

    def test_failed_flag_strictly_above_threshold(self):
        # Exactly 10% is not flagged; above 10% is.
        ten_pct = ["[OCR ERROR]"] + ["good text here"] * 9
        assert compute_column_stats("c", "m", ten_pct).sentinel_rate == 0.1
        assert not compute_column_stats("c", "m", ten_pct).failed

        over = ["[OCR ERROR]", "[OCR ERROR]"] + ["good text here"] * 8
        assert compute_column_stats("c", "m", over).failed

    def test_none_values_counted_as_empty(self):
        s = compute_column_stats("c", "m", [None, None, "text here longer"])
        assert s.n_empty == 2


class TestComputeModelStats:
    def test_over_dataset(self):
        ds = Dataset.from_dict(
            {
                "image": [None, None],
                "cfg_a": ["[OCR ERROR]", "[OCR ERROR]"],
                "cfg_b": ["good text one", "good text two"],
            }
        )
        stats = compute_model_stats(ds, {"cfg_a": "model-a", "cfg_b": "model-b"})
        by_model = {s.model: s for s in stats}
        assert by_model["model-a"].sentinel_rate == 1.0
        assert by_model["model-a"].failed
        assert by_model["model-b"].sentinel_rate == 0.0

    def test_over_list_of_dicts(self):
        ds = [{"col": "[OCR ERROR]"}, {"col": "real text output"}]
        stats = compute_model_stats(ds, {"col": "model-x"})
        assert stats[0].n_sentinel == 1


class TestFailedOutputCounts:
    def test_only_nonzero(self):
        ds = Dataset.from_dict(
            {
                "image": [None, None],
                "cfg_a": ["[OCR ERROR]", "text is fine here"],
                "cfg_b": ["all good one", "all good two"],
            }
        )
        stats = compute_model_stats(ds, {"cfg_a": "model-a", "cfg_b": "model-b"})
        assert failed_output_counts(stats) == {"model-a": 1}


def _loaded_config(name, model, texts, extra=None):
    data = {
        "image": [None] * len(texts),
        "markdown": texts,
        "inference_info": [json.dumps({"model_id": model})] * len(texts),
    }
    if extra:
        data.update(extra)
    return LoadedConfig(
        config=name, model_id=model, ds=Dataset.from_dict(data), text_col="markdown"
    )


class TestAuditRepo:
    @patch("ocr_bench.integrity._load_configs")
    @patch("ocr_bench.integrity.discover_configs")
    @patch("ocr_bench.integrity.discover_pr_configs")
    def test_flags_sentinel_config(self, mock_pr, mock_main, mock_load):
        mock_pr.return_value = ([], {})
        mock_main.return_value = ["cfg_a", "cfg_b"]
        mock_load.return_value = [
            _loaded_config(
                "cfg_a", "model-a", ["[OCR ERROR]", "[OCR ERROR]"], {"b_number": [1, 2]}
            ),
            _loaded_config(
                "cfg_b", "model-b", ["good one here", "good two here"], {"b_number": [1, 2]}
            ),
        ]
        report = audit_repo("user/repo")
        assert report.alignment.status == "ok"
        assert report.flagged_models == ["model-a"]
        assert report.has_problems

    @patch("ocr_bench.integrity._load_configs")
    @patch("ocr_bench.integrity.discover_configs")
    @patch("ocr_bench.integrity.discover_pr_configs")
    def test_detects_misalignment(self, mock_pr, mock_main, mock_load):
        mock_pr.return_value = ([], {})
        mock_main.return_value = ["cfg_a", "cfg_b"]
        mock_load.return_value = [
            _loaded_config("cfg_a", "model-a", ["one", "two"], {"b_number": [1, 2]}),
            _loaded_config("cfg_b", "model-b", ["three", "four"], {"b_number": [1, 3]}),
        ]
        report = audit_repo("user/repo")
        assert report.alignment.status == "misaligned"
        assert report.alignment.index == 1
        assert report.alignment.column == "b_number"
        assert report.has_problems

    @patch("ocr_bench.integrity._load_configs")
    @patch("ocr_bench.integrity.discover_configs")
    @patch("ocr_bench.integrity.discover_pr_configs")
    def test_clean_repo_has_no_problems(self, mock_pr, mock_main, mock_load):
        mock_pr.return_value = ([], {})
        mock_main.return_value = ["cfg_a", "cfg_b"]
        mock_load.return_value = [
            _loaded_config("cfg_a", "model-a", ["good a one", "good a two"], {"b_number": [1, 2]}),
            _loaded_config("cfg_b", "model-b", ["good b one", "good b two"], {"b_number": [1, 2]}),
        ]
        report = audit_repo("user/repo")
        assert not report.has_problems
        assert report.flagged_models == []

    @patch("ocr_bench.integrity._load_configs")
    @patch("ocr_bench.integrity.discover_configs")
    @patch("ocr_bench.integrity.discover_pr_configs")
    def test_row_count_mismatch_is_a_problem(self, mock_pr, mock_main, mock_load):
        mock_pr.return_value = ([], {})
        mock_main.return_value = ["cfg_a", "cfg_b"]
        mock_load.return_value = [
            _loaded_config("cfg_a", "model-a", ["good one here", "good two here", "good three"]),
            _loaded_config("cfg_b", "model-b", ["good one here", "good two here"]),
        ]
        report = audit_repo("user/repo")
        assert report.row_count_mismatch
        assert report.has_problems

    @patch("ocr_bench.integrity._load_configs")
    @patch("ocr_bench.integrity.discover_configs")
    @patch("ocr_bench.integrity.discover_pr_configs")
    def test_partial_alignment_is_not_a_problem(self, mock_pr, mock_main, mock_load):
        mock_pr.return_value = ([], {})
        mock_main.return_value = ["cfg_a", "cfg_b", "cfg_c"]
        mock_load.return_value = [
            _loaded_config(
                "cfg_a", "model-a", ["good a1 here", "good a2 here"], {"b_number": [1, 2]}
            ),
            _loaded_config(
                "cfg_b", "model-b", ["good b1 here", "good b2 here"], {"b_number": [1, 2]}
            ),
            _loaded_config("cfg_c", "model-c", ["good c1 here", "good c2 here"]),
        ]
        report = audit_repo("user/repo")
        assert report.alignment.status == "partial"
        assert report.alignment.config_status("cfg_c") == "unverified"
        assert not report.has_problems  # partial is a caveat, not a failure

    @patch("ocr_bench.integrity.load_flat_dataset")
    @patch("ocr_bench.integrity.discover_configs")
    @patch("ocr_bench.integrity.discover_pr_configs")
    def test_flat_fallback(self, mock_pr, mock_main, mock_flat):
        mock_pr.return_value = ([], {})
        mock_main.return_value = []
        ds = Dataset.from_dict({"image": [None, None], "ocr": ["good text here", "[OCR ERROR]"]})
        mock_flat.return_value = (ds, {"ocr": "model-x"})
        report = audit_repo("user/repo")
        assert report.alignment.status == "n/a"
        assert report.judge_text_mode == "normalized"
        assert len(report.configs) == 1
        assert report.has_problems  # 50% sentinels
