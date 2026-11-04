"""Tests for the shared input-integrity checks (stats + audit)."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from datasets import Dataset

from ocr_bench.dataset import DatasetError, LoadedConfig
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
    def test_reports_passed_over_prs_and_loads_main(self, mock_pr, mock_main, mock_load):
        def discover(repo_id, api=None, duplicates=None, main_configs=(), superseded=None):
            assert list(main_configs) == ["cfg_a", "cfg_b"]
            superseded["cfg_a"] = [2]
            duplicates["cfg_c"] = [3]
            return ["cfg_c"], {"cfg_c": "refs/pr/7"}

        mock_pr.side_effect = discover
        mock_main.return_value = ["cfg_a", "cfg_b"]
        mock_load.return_value = [
            _loaded_config(name, f"model-{name}", ["good text here"], {"b_number": [1]})
            for name in ("cfg_c", "cfg_a", "cfg_b")
        ]
        report = audit_repo("user/repo")
        names, _, revisions = mock_load.call_args.args[1:]
        assert names == ["cfg_c", "cfg_a", "cfg_b"]
        assert revisions == {"cfg_c": "refs/pr/7"}  # cfg_a loads from main
        assert report.superseded_prs == {"cfg_a": [2]}
        assert report.duplicate_prs == {"cfg_c": [3]}
        assert report.pr_revisions == {"cfg_c": "refs/pr/7"}

    @patch("ocr_bench.integrity._load_configs")
    @patch("ocr_bench.integrity.discover_configs")
    @patch("ocr_bench.integrity.discover_pr_configs")
    def test_mixed_usable_and_missing_text_is_a_problem(self, mock_pr, mock_main, mock_load):
        mock_pr.return_value = ([], {})
        mock_main.return_value = ["cfg_a", "cfg_b", "cfg_c"]
        no_text = LoadedConfig(
            config="cfg_c", model_id="model-c", ds=Dataset.from_dict({"inference_info": ["{}"]}),
            text_col=None,
        )
        mock_load.return_value = [
            _loaded_config("cfg_a", "model-a", ["good one here"], {"b_number": [1]}),
            _loaded_config("cfg_b", "model-b", ["good two here"], {"b_number": [1]}),
            no_text,
        ]
        report = audit_repo("user/repo")
        assert report.configs_without_text == ["cfg_c"]
        assert report.unjudgeable_configs == ["cfg_c"]
        assert not report.too_few_models
        assert report.has_problems

    @patch("ocr_bench.integrity._load_configs")
    @patch("ocr_bench.integrity.discover_configs")
    @patch("ocr_bench.integrity.discover_pr_configs")
    def test_auxiliary_config_is_reported_but_not_a_problem(self, mock_pr, mock_main, mock_load):
        mock_pr.return_value = ([], {})
        mock_main.return_value = ["cfg_a", "cfg_b", "metadata"]
        metadata = LoadedConfig(
            config="metadata", model_id="", ds=Dataset.from_dict({"run": ["x"]}), text_col=None
        )
        mock_load.return_value = [
            _loaded_config("cfg_a", "model-a", ["good one here"], {"b_number": [1]}),
            _loaded_config("cfg_b", "model-b", ["good two here"], {"b_number": [1]}),
            metadata,
        ]
        report = audit_repo("user/repo")
        assert report.auxiliary_configs == ["metadata"]
        assert report.configs_without_text == []
        assert not report.has_problems

    @patch("ocr_bench.integrity._load_configs")
    @patch("ocr_bench.integrity.discover_configs")
    @patch("ocr_bench.integrity.discover_pr_configs")
    def test_all_empty_configs_are_a_problem(self, mock_pr, mock_main, mock_load):
        mock_pr.return_value = ([], {})
        mock_main.return_value = ["cfg_a", "cfg_b"]
        mock_load.return_value = [
            _loaded_config("cfg_a", "model-a", ["", ""], {"b_number": [1, 2]}),
            _loaded_config("cfg_b", "model-b", ["", ""], {"b_number": [1, 2]}),
        ]
        report = audit_repo("user/repo")
        assert report.unjudgeable_configs == ["cfg_a", "cfg_b"]
        assert report.too_few_models
        assert report.has_problems

    @patch("ocr_bench.integrity._load_configs")
    @patch("ocr_bench.integrity.discover_configs")
    @patch("ocr_bench.integrity.discover_pr_configs")
    def test_zero_row_config_is_a_problem(self, mock_pr, mock_main, mock_load):
        mock_pr.return_value = ([], {})
        mock_main.return_value = ["cfg_a", "cfg_b"]
        mock_load.return_value = [
            _loaded_config("cfg_a", "model-a", [], {}),
            _loaded_config("cfg_b", "model-b", [], {}),
        ]
        report = audit_repo("user/repo")
        assert report.unjudgeable_configs == ["cfg_a", "cfg_b"]
        assert report.has_problems

    @patch("ocr_bench.integrity._load_configs")
    @patch("ocr_bench.integrity.discover_configs")
    @patch("ocr_bench.integrity.discover_pr_configs")
    def test_single_usable_model_is_a_problem(self, mock_pr, mock_main, mock_load):
        mock_pr.return_value = ([], {})
        mock_main.return_value = ["cfg_a"]
        mock_load.return_value = [
            _loaded_config("cfg_a", "model-a", ["good one here"], {"b_number": [1]}),
        ]
        report = audit_repo("user/repo")
        assert report.too_few_models
        assert report.has_problems

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


_PAGES = [
    "first page of a longer transcription",
    "second page of a longer transcription",
]


def _three_configs(names=("cfg_a", "cfg_b", "cfg_c")):
    return [
        _loaded_config(name, f"model-{name}", list(_PAGES), {"b_number": [1, 2]})
        for name in names
    ]


class TestAuditDiscovery:
    """audit selects configs the way judge does with the same flags."""

    @patch("ocr_bench.integrity._load_configs")
    @patch("ocr_bench.integrity.discover_configs")
    @patch("ocr_bench.integrity.discover_pr_configs")
    def test_configs_alone_load_from_main(self, mock_pr, mock_main, mock_load):
        mock_load.return_value = _three_configs(("cfg_b", "cfg_a"))
        report = audit_repo("user/repo", configs=["cfg_b", "cfg_a"])
        mock_pr.assert_not_called()
        names, _, revisions = mock_load.call_args.args[1:]
        assert names == ["cfg_b", "cfg_a"]
        assert revisions == {}
        assert report.pr_revisions == {}

    @patch("ocr_bench.integrity._load_configs")
    @patch("ocr_bench.integrity.discover_configs")
    @patch("ocr_bench.integrity.discover_pr_configs")
    def test_configs_with_from_prs_use_the_newest_pr(self, mock_pr, mock_main, mock_load):
        def discover(
            repo_id, api=None, only=None, duplicates=None, main_configs=(), superseded=None
        ):
            assert only == ["cfg_a", "cfg_b"]
            assert list(main_configs) == ["cfg_b"]
            duplicates["cfg_a"] = [2]
            return ["cfg_a"], {"cfg_a": "refs/pr/5"}

        mock_pr.side_effect = discover
        mock_main.return_value = ["cfg_b"]
        mock_load.return_value = _three_configs(("cfg_a", "cfg_b"))
        report = audit_repo("user/repo", configs=["cfg_a", "cfg_b"], from_prs=True)
        names, _, revisions = mock_load.call_args.args[1:]
        assert names == ["cfg_a", "cfg_b"]
        assert revisions == {"cfg_a": "refs/pr/5"}
        assert report.duplicate_prs == {"cfg_a": [2]}

    @patch("ocr_bench.integrity._load_configs")
    @patch("ocr_bench.integrity.discover_configs")
    @patch("ocr_bench.integrity.discover_pr_configs")
    def test_from_prs_adds_superseded_configs_from_main(self, mock_pr, mock_main, mock_load):
        def discover(repo_id, api=None, duplicates=None, main_configs=(), superseded=None):
            superseded["cfg_b"] = [3]
            return ["cfg_a"], {"cfg_a": "refs/pr/4"}

        mock_pr.side_effect = discover
        mock_main.return_value = ["cfg_b", "cfg_c"]
        mock_load.return_value = _three_configs(("cfg_a", "cfg_b"))
        report = audit_repo("user/repo", from_prs=True)
        names, _, revisions = mock_load.call_args.args[1:]
        assert names == ["cfg_a", "cfg_b"]  # cfg_c is on main only: not audited
        assert revisions == {"cfg_a": "refs/pr/4"}
        assert report.superseded_prs == {"cfg_b": [3]}

    @patch("ocr_bench.integrity.discover_configs")
    @patch("ocr_bench.integrity.discover_pr_configs")
    def test_from_prs_without_prs_is_an_error(self, mock_pr, mock_main):
        mock_pr.return_value = ([], {})
        mock_main.return_value = ["cfg_a"]
        with pytest.raises(DatasetError, match="No configs found in open PRs"):
            audit_repo("user/repo", from_prs=True)


class TestAuditScreen:
    @patch("ocr_bench.integrity._load_configs")
    @patch("ocr_bench.integrity.discover_configs")
    @patch("ocr_bench.integrity.discover_pr_configs")
    def test_screen_runs_on_three_configs(self, mock_pr, mock_main, mock_load):
        mock_pr.return_value = ([], {})
        mock_main.return_value = ["cfg_a", "cfg_b", "cfg_c"]
        mock_load.return_value = _three_configs()
        report = audit_repo("user/repo", screen=True)
        assert report.screen is not None
        assert report.screen_skipped is None
        assert [m.name for m in report.screen.models] == ["cfg_a", "cfg_b", "cfg_c"]
        assert report.screen.models[0].model == "model-cfg_a"
        assert report.screen.rows == 2

    @patch("ocr_bench.integrity._load_configs")
    @patch("ocr_bench.integrity.discover_configs")
    @patch("ocr_bench.integrity.discover_pr_configs")
    def test_no_screen_unless_asked(self, mock_pr, mock_main, mock_load):
        mock_pr.return_value = ([], {})
        mock_main.return_value = ["cfg_a", "cfg_b", "cfg_c"]
        mock_load.return_value = _three_configs()
        report = audit_repo("user/repo")
        assert report.screen is None
        assert report.screen_skipped is None

    @patch("ocr_bench.integrity._load_configs")
    @patch("ocr_bench.integrity.discover_configs")
    @patch("ocr_bench.integrity.discover_pr_configs")
    def test_screen_skipped_with_two_models(self, mock_pr, mock_main, mock_load):
        mock_pr.return_value = ([], {})
        mock_main.return_value = ["cfg_a", "cfg_b"]
        mock_load.return_value = _three_configs(("cfg_a", "cfg_b"))
        report = audit_repo("user/repo", screen=True)
        assert report.screen is None
        assert "at least 3 models" in report.screen_skipped

    @patch("ocr_bench.integrity._load_configs")
    @patch("ocr_bench.integrity.discover_configs")
    @patch("ocr_bench.integrity.discover_pr_configs")
    def test_screen_leaves_out_unjudgeable_configs(self, mock_pr, mock_main, mock_load):
        mock_pr.return_value = ([], {})
        mock_main.return_value = ["cfg_a", "cfg_b", "cfg_c", "cfg_d"]
        mock_load.return_value = _three_configs() + [
            _loaded_config("cfg_d", "model-d", ["", ""], {"b_number": [1, 2]})
        ]
        report = audit_repo("user/repo", screen=True)
        assert report.unjudgeable_configs == ["cfg_d"]
        assert [m.name for m in report.screen.models] == ["cfg_a", "cfg_b", "cfg_c"]

    @patch("ocr_bench.integrity._load_configs")
    @patch("ocr_bench.integrity.discover_configs")
    @patch("ocr_bench.integrity.discover_pr_configs")
    def test_all_sentinel_config_does_not_count_as_a_model(self, mock_pr, mock_main, mock_load):
        mock_pr.return_value = ([], {})
        mock_main.return_value = ["cfg_a", "cfg_b", "cfg_c"]
        loaded = _three_configs(("cfg_a", "cfg_b"))
        loaded.append(
            _loaded_config(
                "cfg_c", "model-c", ["[OCR ERROR]", "[OCR ERROR]"], {"b_number": [1, 2]}
            )
        )
        mock_load.return_value = loaded
        report = audit_repo("user/repo", screen=True)
        assert report.screen is None
        assert "at least 3 models" in report.screen_skipped

    @patch("ocr_bench.integrity._load_configs")
    @patch("ocr_bench.integrity.discover_configs")
    @patch("ocr_bench.integrity.discover_pr_configs")
    def test_screen_skipped_on_misalignment(self, mock_pr, mock_main, mock_load):
        mock_pr.return_value = ([], {})
        mock_main.return_value = ["cfg_a", "cfg_b", "cfg_c"]
        loaded = _three_configs()
        loaded[2] = _loaded_config("cfg_c", "model-c", list(_PAGES), {"b_number": [2, 1]})
        mock_load.return_value = loaded
        report = audit_repo("user/repo", screen=True)
        assert report.alignment.status == "misaligned"
        assert report.screen is None
        assert "misaligned" in report.screen_skipped

    @patch("ocr_bench.integrity._load_configs")
    @patch("ocr_bench.integrity.discover_configs")
    @patch("ocr_bench.integrity.discover_pr_configs")
    def test_screen_skipped_on_row_count_mismatch(self, mock_pr, mock_main, mock_load):
        mock_pr.return_value = ([], {})
        mock_main.return_value = ["cfg_a", "cfg_b", "cfg_c"]
        loaded = _three_configs()
        loaded[2] = _loaded_config("cfg_c", "model-c", ["only one page here"])
        mock_load.return_value = loaded
        report = audit_repo("user/repo", screen=True)
        assert report.screen is None
        assert "row counts" in report.screen_skipped

    @patch("ocr_bench.integrity.load_flat_dataset")
    @patch("ocr_bench.integrity.discover_configs")
    @patch("ocr_bench.integrity.discover_pr_configs")
    def test_screen_on_flat_columns(self, mock_pr, mock_main, mock_flat):
        mock_pr.return_value = ([], {})
        mock_main.return_value = []
        ds = Dataset.from_dict({"image": [None, None], "x": _PAGES, "y": _PAGES, "z": _PAGES})
        mock_flat.return_value = (ds, {"x": "model-x", "y": "model-y", "z": "model-z"})
        report = audit_repo("user/repo", screen=True)
        assert [m.model for m in report.screen.models] == ["model-x", "model-y", "model-z"]


class TestAuditReportToDict:
    @patch("ocr_bench.integrity._load_configs")
    @patch("ocr_bench.integrity.discover_configs")
    @patch("ocr_bench.integrity.discover_pr_configs")
    def test_round_trips_through_json(self, mock_pr, mock_main, mock_load):
        mock_pr.return_value = (["cfg_a"], {"cfg_a": "refs/pr/9"})
        mock_main.return_value = ["cfg_b", "cfg_c"]
        mock_load.return_value = _three_configs()
        data = json.loads(json.dumps(audit_repo("user/repo", screen=True).to_dict()))
        assert data["ok"] is True
        assert data["problems"] == []
        assert [c["source"] for c in data["configs"]] == ["refs/pr/9", "main", "main"]
        assert data["screen"]["rows"] == 2
        assert data["screen_skipped"] is None

    @patch("ocr_bench.integrity._load_configs")
    @patch("ocr_bench.integrity.discover_configs")
    @patch("ocr_bench.integrity.discover_pr_configs")
    def test_lists_problems(self, mock_pr, mock_main, mock_load):
        mock_pr.return_value = ([], {})
        mock_main.return_value = ["cfg_a"]
        mock_load.return_value = _three_configs(("cfg_a",))
        data = audit_repo("user/repo").to_dict()
        assert data["ok"] is False
        assert data["problems"] == ["fewer than two judgeable configs"]
        assert data["screen"] is None
