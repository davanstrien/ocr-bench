"""Tests for the CLI."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest
from openai import OpenAIError

from ocr_bench import cli
from ocr_bench.cli import (
    _convert_results,
    _existing_criteria_provenance,
    _existing_preprocessing_provenance,
    _merge_auto_ties,
    _refresh_viewer_space,
    _resolve_criteria,
    _resolve_results_repo,
    _trim_to_budget,
    build_parser,
)
from ocr_bench.dataset import AlignmentResult, DatasetError
from ocr_bench.elo import compute_elo
from ocr_bench.integrity import AuditReport, ColumnStats, ConfigAudit
from ocr_bench.judge import CRITERIA_PROFILES, DEFAULT_CRITERIA, Comparison, prompt_hash


class TestBuildParser:
    def test_judge_subcommand_defaults(self):
        parser = build_parser()
        args = parser.parse_args(["judge", "user/dataset"])
        assert args.command == "judge"
        assert args.dataset == "user/dataset"
        assert args.split == "train"
        assert args.columns is None
        assert args.configs is None
        assert args.from_prs is False
        assert args.merge is False
        assert args.no_adaptive is False
        assert args.no_publish is False
        assert args.models is None
        assert args.max_samples is None
        assert args.seed == 42
        assert args.save_results is None
        assert args.min_chars == 20
        assert args.max_ocr_text_len == 2500
        assert args.judge_image_dim == 1024
        assert args.judge_text_mode == "normalized"
        assert args.adaptive_strategy == "balanced"
        assert args.size_tie_ratio is None
        assert args.size_tie_min_samples == 10

    @pytest.mark.parametrize("flag", ["--max-ocr-text-len", "--judge-image-dim"])
    @pytest.mark.parametrize("value", ["0", "-1"])
    def test_judge_caps_require_positive_values(self, flag, value):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["judge", "user/dataset", flag, value])

    def test_judge_text_mode_raw(self):
        parser = build_parser()
        args = parser.parse_args(["judge", "user/dataset", "--judge-text-mode", "raw"])
        assert args.judge_text_mode == "raw"

    def test_judge_text_mode_normalized(self):
        parser = build_parser()
        args = parser.parse_args(["judge", "user/dataset", "--judge-text-mode", "normalized"])
        assert args.judge_text_mode == "normalized"

    def test_judge_text_mode_rejects_unknown(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["judge", "user/dataset", "--judge-text-mode", "cooked"])

    def test_bench_judge_text_mode(self):
        parser = build_parser()
        args = parser.parse_args(["bench", "user/imgs", "user/out"])
        assert args.judge_text_mode == "normalized"
        args = parser.parse_args(
            ["bench", "user/imgs", "user/out", "--judge-text-mode", "raw"]
        )
        assert args.judge_text_mode == "raw"

    def test_multiple_models(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "judge",
                "user/dataset",
                "--model",
                "novita:org/model-a",
                "--model",
                "together:org/model-b",
            ]
        )
        assert args.models == ["novita:org/model-a", "together:org/model-b"]

    def test_explicit_columns(self):
        parser = build_parser()
        args = parser.parse_args(["judge", "user/dataset", "--columns", "col_a", "col_b"])
        assert args.columns == ["col_a", "col_b"]

    def test_config_mode(self):
        parser = build_parser()
        args = parser.parse_args(["judge", "user/dataset", "--configs", "cfg_a", "cfg_b"])
        assert args.configs == ["cfg_a", "cfg_b"]

    def test_from_prs_flag(self):
        parser = build_parser()
        args = parser.parse_args(["judge", "user/dataset", "--from-prs"])
        assert args.from_prs is True

    def test_merge_flag(self):
        parser = build_parser()
        args = parser.parse_args(["judge", "user/dataset", "--merge"])
        assert args.merge is True

    def test_no_adaptive_flag(self):
        parser = build_parser()
        args = parser.parse_args(["judge", "user/dataset", "--no-adaptive"])
        assert args.no_adaptive is True

    def test_targeted_adaptive_and_size_rule(self):
        args = build_parser().parse_args(
            [
                "judge",
                "user/dataset",
                "--adaptive-strategy",
                "targeted",
                "--size-tie-ratio",
                "3",
                "--size-tie-min-samples",
                "5",
            ]
        )
        assert args.adaptive_strategy == "targeted"
        assert args.size_tie_ratio == 3.0
        assert args.size_tie_min_samples == 5

    @pytest.mark.parametrize("value", ["1", "0", "-2", "nan", "inf"])
    def test_size_tie_ratio_must_exceed_one(self, value):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["judge", "user/dataset", "--size-tie-ratio", value])

    def test_no_publish_flag(self):
        parser = build_parser()
        args = parser.parse_args(["judge", "user/dataset", "--no-publish"])
        assert args.no_publish is True

    def test_max_samples_and_seed(self):
        parser = build_parser()
        args = parser.parse_args(["judge", "user/dataset", "--max-samples", "10", "--seed", "123"])
        assert args.max_samples == 10
        assert args.seed == 123

    def test_save_results(self):
        parser = build_parser()
        args = parser.parse_args(["judge", "user/dataset", "--save-results", "user/results"])
        assert args.save_results == "user/results"

    def test_no_command_exits_zero(self):
        """No subcommand should print help and exit 0."""
        parser = build_parser()
        args = parser.parse_args([])
        assert args.command is None

    def test_help_exits(self):
        parser = build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--help"])
        assert exc_info.value.code == 0

    def test_judge_help_exits(self):
        parser = build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["judge", "--help"])
        assert exc_info.value.code == 0

    def test_full_rejudge_flag(self):
        parser = build_parser()
        args = parser.parse_args(["judge", "user/dataset", "--full-rejudge"])
        assert args.full_rejudge is True

    def test_full_rejudge_defaults_false(self):
        parser = build_parser()
        args = parser.parse_args(["judge", "user/dataset"])
        assert args.full_rejudge is False

    def test_criteria_default_is_none_sentinel(self):
        # Parser default is the None sentinel; _resolve_criteria maps it to
        # DEFAULT_CRITERIA (so explicit-vs-unset is distinguishable).
        parser = build_parser()
        args = parser.parse_args(["judge", "user/dataset"])
        assert args.criteria is None
        assert _resolve_criteria(args)[0] == "default"

    def test_criteria_accepts_table_fidelity(self):
        parser = build_parser()
        args = parser.parse_args(["judge", "user/dataset", "--criteria", "table-fidelity"])
        assert args.criteria == "table-fidelity"

    def test_criteria_rejects_unknown(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["judge", "user/dataset", "--criteria", "nonsense"])

    def test_criteria_file_default_none(self):
        parser = build_parser()
        args = parser.parse_args(["judge", "user/dataset"])
        assert args.criteria_file is None

    def test_criteria_file_parses(self):
        parser = build_parser()
        args = parser.parse_args(["judge", "user/dataset", "--criteria-file", "my/rubric.txt"])
        assert args.criteria_file == "my/rubric.txt"
        assert args.criteria is None  # unset sentinel when only the file is given

    def test_criteria_and_criteria_file_conflict_errors(self):
        # Exclusivity is enforced in _resolve_criteria (not the argparse layer,
        # which is unreliable on Python 3.13.0), so both parse cleanly and the
        # conflict surfaces as a DatasetError.
        args = build_parser().parse_args(
            ["judge", "user/dataset", "--criteria", "default", "--criteria-file", "r.txt"]
        )
        with pytest.raises(DatasetError, match="mutually exclusive"):
            _resolve_criteria(args)


class TestResolveCriteria:
    def test_builtin_default(self):
        args = build_parser().parse_args(["judge", "user/ds"])
        name, tmpl, phash = _resolve_criteria(args)
        assert name == "default"
        assert tmpl == CRITERIA_PROFILES["default"]
        assert phash == prompt_hash(CRITERIA_PROFILES["default"])

    def test_builtin_table_fidelity(self):
        args = build_parser().parse_args(["judge", "user/ds", "--criteria", "table-fidelity"])
        name, tmpl, phash = _resolve_criteria(args)
        assert name == "table-fidelity"
        assert phash == prompt_hash(CRITERIA_PROFILES["table-fidelity"])

    def test_custom_file(self, tmp_path):
        f = tmp_path / "table-rubric.txt"
        f.write_text("Judge these. A: {ocr_text_a} B: {ocr_text_b}")
        args = build_parser().parse_args(["judge", "user/ds", "--criteria-file", str(f)])
        name, tmpl, phash = _resolve_criteria(args)
        assert name == "custom:table-rubric.txt"
        assert tmpl == f.read_text()
        assert phash == prompt_hash(tmpl)

    def test_custom_file_invalid_template_raises(self, tmp_path):
        f = tmp_path / "bad.txt"
        f.write_text("missing the b placeholder: {ocr_text_a}")
        args = build_parser().parse_args(["judge", "user/ds", "--criteria-file", str(f)])
        with pytest.raises(DatasetError, match="ocr_text_b"):
            _resolve_criteria(args)

    def test_custom_file_missing_raises(self, tmp_path):
        args = build_parser().parse_args(
            ["judge", "user/ds", "--criteria-file", str(tmp_path / "nope.txt")]
        )
        with pytest.raises(DatasetError, match="could not read"):
            _resolve_criteria(args)


class TestAuditParser:
    def test_audit_defaults(self):
        args = build_parser().parse_args(["audit", "user/data"])
        assert args.command == "audit"
        assert args.dataset == "user/data"
        assert args.split == "train"
        assert args.judge_text_mode == "normalized"
        assert args.max_ocr_text_len == 2500

    def test_audit_overrides(self):
        args = build_parser().parse_args(
            [
                "audit",
                "user/data",
                "--split",
                "test",
                "--judge-text-mode",
                "raw",
                "--max-ocr-text-len",
                "500",
            ]
        )
        assert args.split == "test"
        assert args.judge_text_mode == "raw"
        assert args.max_ocr_text_len == 500


def _audit_report(sentinel_count: int = 0, n_rows: int = 10, status: str = "ok") -> AuditReport:
    stats = ColumnStats(
        name="cfg",
        model="m",
        n_rows=n_rows,
        n_empty=0,
        n_sentinel=sentinel_count,
        n_short=0,
        n_over_max=0,
        median_len=100.0,
        max_len=200,
        max_ocr_text_len=2500,
    )
    return AuditReport(
        repo_id="user/data",
        configs=[ConfigAudit(stats)],
        alignment=AlignmentResult(status=status),
        max_ocr_text_len=2500,
    )


class TestCmdAudit:
    def test_clean_repo_does_not_exit(self, monkeypatch):
        monkeypatch.setattr(cli, "audit_repo", lambda *a, **k: _audit_report())
        args = build_parser().parse_args(["audit", "user/data"])
        cli.cmd_audit(args)  # no SystemExit

    def test_high_sentinels_exit_one(self, monkeypatch):
        monkeypatch.setattr(cli, "audit_repo", lambda *a, **k: _audit_report(sentinel_count=5))
        args = build_parser().parse_args(["audit", "user/data"])
        with pytest.raises(SystemExit) as exc_info:
            cli.cmd_audit(args)
        assert exc_info.value.code == 1

    def test_misalignment_exit_one(self, monkeypatch):
        monkeypatch.setattr(
            cli, "audit_repo", lambda *a, **k: _audit_report(status="misaligned")
        )
        args = build_parser().parse_args(["audit", "user/data"])
        with pytest.raises(SystemExit) as exc_info:
            cli.cmd_audit(args)
        assert exc_info.value.code == 1

    def _stats(self, name: str, n_rows: int) -> ColumnStats:
        return ColumnStats(
            name=name,
            model=name,
            n_rows=n_rows,
            n_empty=0,
            n_sentinel=0,
            n_short=0,
            n_over_max=0,
            median_len=100.0,
            max_len=200,
            max_ocr_text_len=2500,
        )

    def test_row_count_mismatch_exit_one(self, monkeypatch):
        report = AuditReport(
            repo_id="user/data",
            configs=[ConfigAudit(self._stats("cfg_a", 3)), ConfigAudit(self._stats("cfg_b", 2))],
            alignment=AlignmentResult(status="unverified"),
            max_ocr_text_len=2500,
        )
        monkeypatch.setattr(cli, "audit_repo", lambda *a, **k: report)
        args = build_parser().parse_args(["audit", "user/data"])
        with pytest.raises(SystemExit) as exc_info:
            cli.cmd_audit(args)
        assert exc_info.value.code == 1

    def test_no_configs_exit_two(self, monkeypatch):
        # Couldn't audit anything → operational (2), not an integrity failure (1).
        empty = AuditReport(
            repo_id="user/data",
            configs=[],
            alignment=AlignmentResult(status="n/a"),
            max_ocr_text_len=2500,
        )
        monkeypatch.setattr(cli, "audit_repo", lambda *a, **k: empty)
        args = build_parser().parse_args(["audit", "user/data"])
        with pytest.raises(SystemExit) as exc_info:
            cli.cmd_audit(args)
        assert exc_info.value.code == 2

    def test_operational_error_exit_two(self, monkeypatch):
        # A Hub/network/load failure during the audit is exit 2, so CI can tell
        # a broken run from a bad repo.
        def boom(*a, **k):
            raise OSError("hub unreachable")

        monkeypatch.setattr(cli, "audit_repo", boom)
        args = build_parser().parse_args(["audit", "user/data"])
        with pytest.raises(SystemExit) as exc_info:
            cli.cmd_audit(args)
        assert exc_info.value.code == 2


class TestMainErrorHandling:
    """A judge/Hub failure should exit cleanly, not dump a traceback."""

    def _run_main_with(self, monkeypatch, exc):
        def boom(_args):
            raise exc

        monkeypatch.setattr(cli, "cmd_judge", boom)
        monkeypatch.setattr(
            sys, "argv", ["ocr-bench", "judge", "user/dataset", "--no-publish"]
        )
        with pytest.raises(SystemExit) as exc_info:
            cli.main()
        return exc_info.value.code

    def test_openai_error_exits_one(self, monkeypatch, capsys):
        code = self._run_main_with(monkeypatch, OpenAIError("bad token"))
        assert code == 1
        assert "Error" in capsys.readouterr().out

    def test_hub_connection_error_exits_one(self, monkeypatch, capsys):
        # requests/HfHubHTTPError all subclass OSError; builtin ConnectionError
        # exercises the same except branch without importing requests.
        code = self._run_main_with(monkeypatch, ConnectionError("no network"))
        assert code == 1
        assert "Error" in capsys.readouterr().out


class TestResolveResultsRepo:
    def test_auto_derives_from_dataset(self):
        result = _resolve_results_repo("user/my-dataset", None, False)
        assert result == "user/my-dataset-results"

    def test_explicit_save_results_overrides(self):
        result = _resolve_results_repo("user/my-dataset", "user/custom-results", False)
        assert result == "user/custom-results"

    def test_no_publish_returns_none(self):
        result = _resolve_results_repo("user/my-dataset", None, True)
        assert result is None

    def test_no_publish_overrides_explicit(self):
        result = _resolve_results_repo("user/my-dataset", "user/custom", True)
        assert result is None


class TestExistingPreprocessingProvenance:
    def test_missing_metadata_is_legacy_raw(self):
        assert _existing_preprocessing_provenance([]) == ("raw", 2500, 1024)
        assert _existing_preprocessing_provenance([{"source_dataset": "d"}]) == (
            "raw",
            2500,
            1024,
        )

    def test_reads_latest_recorded_settings(self):
        rows = [
            {"judge_text_mode": "raw", "max_ocr_text_len": 2500, "judge_image_dim": 1024},
            {
                "judge_text_mode": "normalized",
                "max_ocr_text_len": 5000,
                "judge_image_dim": 2048,
            },
        ]
        assert _existing_preprocessing_provenance(rows) == ("normalized", 5000, 2048)

    def test_aligned_none_values_use_legacy_defaults(self):
        rows = [
            {"judge_text_mode": None, "max_ocr_text_len": None, "judge_image_dim": None}
        ]
        assert _existing_preprocessing_provenance(rows) == ("raw", 2500, 1024)


class TestExistingCriteriaProvenance:
    """Reads the criteria/prompt_hash the existing comparisons were judged under."""

    _DEFAULT_HASH = prompt_hash(CRITERIA_PROFILES[DEFAULT_CRITERIA])

    def test_no_rows_is_default(self):
        assert _existing_criteria_provenance([]) == (DEFAULT_CRITERIA, self._DEFAULT_HASH)

    def test_reads_last_row(self):
        rows = [
            {"criteria": "default", "prompt_hash": self._DEFAULT_HASH},
            {"criteria": "table-fidelity", "prompt_hash": "fe138e71ecc3"},
        ]
        assert _existing_criteria_provenance(rows) == ("table-fidelity", "fe138e71ecc3")

    def test_none_values_treated_as_default(self):
        # Post-alignment old rows carry explicit None for the new columns.
        rows = [{"criteria": None, "prompt_hash": None}]
        assert _existing_criteria_provenance(rows) == (DEFAULT_CRITERIA, self._DEFAULT_HASH)

    def test_missing_keys_treated_as_default(self):
        # Genuinely pre-#44 rows lack the columns entirely.
        rows = [{"source_dataset": "d", "judge_models": "[]"}]
        assert _existing_criteria_provenance(rows) == (DEFAULT_CRITERIA, self._DEFAULT_HASH)


_AUTO_TIE = {"winner": "tie", "reason": "identical outputs — auto-tie", "agreement": "auto"}


def _make_comparison(idx: int = 0) -> Comparison:
    return Comparison(
        sample_idx=idx,
        model_a="model-a",
        model_b="model-b",
        col_a="col_a",
        col_b="col_b",
        swapped=False,
        messages=[{"role": "user", "content": "test"}],
    )


def _auto_comparison(idx: int = 0) -> Comparison:
    return Comparison(
        sample_idx=idx,
        model_a="model-a",
        model_b="model-b",
        col_a="col_a",
        col_b="col_b",
        swapped=False,
        messages=[],
        auto_result=dict(_AUTO_TIE),
    )


class TestConvertResults:
    def test_valid_result_kept(self):
        results = _convert_results(
            [_make_comparison()],
            [{"winner": "A", "reason": "better", "agreement": "3/3"}],
        )
        assert len(results) == 1
        assert results[0].winner == "A"
        assert results[0].agreement == "3/3"

    def test_empty_result_skipped(self):
        results = _convert_results([_make_comparison()], [{}])
        assert results == []

    def test_all_judges_failed_tie_skipped(self):
        """A 0/0 'tie' means every judge in the jury failed — it must not
        enter the ELO computation as a real verdict."""
        failed = {"winner": "tie", "reason": "no valid votes", "agreement": "0/0"}
        results = _convert_results([_make_comparison()], [failed])
        assert results == []

    def test_mixed_results_keep_only_valid(self):
        comps = [_make_comparison(i) for i in range(3)]
        aggregated = [
            {"winner": "A", "reason": "ok", "agreement": "2/3"},
            {},
            {"winner": "tie", "reason": "no valid votes", "agreement": "0/0"},
        ]
        results = _convert_results(comps, aggregated)
        assert len(results) == 1
        assert results[0].sample_idx == 0

    def test_truncation_flags_propagated(self):
        comp = Comparison(
            sample_idx=0,
            model_a="model-a",
            model_b="model-b",
            col_a="col_a",
            col_b="col_b",
            swapped=False,
            messages=[{"role": "user", "content": "test"}],
            truncated_a=True,
            truncated_b=False,
        )
        results = _convert_results([comp], [{"winner": "A", "reason": "ok"}])
        assert results[0].truncated_a is True
        assert results[0].truncated_b is False


def _space_var(value: str) -> MagicMock:
    """A stand-in for huggingface_hub's SpaceVariable (has a .value)."""
    var = MagicMock()
    var.value = value
    return var


class TestRefreshViewerSpace:
    """Layers 0+2 of issue #37: judge→Space chaining with a wiring-drift guard."""

    @patch("huggingface_hub.HfApi")
    def test_restarts_when_repos_matches(self, mock_api_cls):
        api = mock_api_cls.return_value
        api.repo_exists.return_value = True
        api.get_space_variables.return_value = {"REPOS": _space_var("user/x-results")}

        _refresh_viewer_space("user/x-results")

        api.restart_space.assert_called_once_with(
            "user/x-results-viewer", factory_reboot=True
        )

    @patch("huggingface_hub.HfApi")
    def test_restarts_when_repos_unset(self, mock_api_cls):
        """No REPOS variable is not drift — the Space defaults to its own repo."""
        api = mock_api_cls.return_value
        api.repo_exists.return_value = True
        api.get_space_variables.return_value = {}

        _refresh_viewer_space("user/x-results")

        api.restart_space.assert_called_once_with(
            "user/x-results-viewer", factory_reboot=True
        )

    @patch("huggingface_hub.HfApi")
    def test_warns_and_skips_on_repos_mismatch(self, mock_api_cls):
        api = mock_api_cls.return_value
        api.repo_exists.return_value = True
        api.get_space_variables.return_value = {"REPOS": _space_var("user/OTHER-results")}

        _refresh_viewer_space("user/x-results")

        api.restart_space.assert_not_called()

    @patch("huggingface_hub.HfApi")
    def test_no_space_does_nothing(self, mock_api_cls):
        api = mock_api_cls.return_value
        api.repo_exists.return_value = False

        _refresh_viewer_space("user/x-results")

        api.get_space_variables.assert_not_called()
        api.restart_space.assert_not_called()

    @patch("huggingface_hub.HfApi")
    def test_hub_error_never_raises(self, mock_api_cls):
        """A Space hiccup must never fail an otherwise-successful judge run."""
        api = mock_api_cls.return_value
        api.repo_exists.side_effect = RuntimeError("hub down")

        _refresh_viewer_space("user/x-results")  # must not raise

        api.restart_space.assert_not_called()
    def test_auto_tie_kept_as_tie(self):
        """An auto-tie verdict is a real tie, not a failure — it must survive."""
        results = _convert_results([_auto_comparison()], [dict(_AUTO_TIE)])
        assert len(results) == 1
        assert results[0].winner == "tie"
        assert results[0].agreement == "auto"


class TestMergeAutoTies:
    def test_interleaves_in_order(self):
        comps = [_make_comparison(0), _auto_comparison(1), _make_comparison(2)]
        judged = [
            {"winner": "A", "reason": "x", "agreement": "1/1"},
            {"winner": "B", "reason": "y", "agreement": "1/1"},
        ]
        merged = _merge_auto_ties(comps, judged)
        assert merged[0]["winner"] == "A"
        assert merged[1] == _AUTO_TIE  # auto-tie slotted in place, not judged
        assert merged[2]["winner"] == "B"

    def test_all_judged_passthrough(self):
        comps = [_make_comparison(i) for i in range(2)]
        judged = [{"winner": "A"}, {"winner": "tie"}]
        assert _merge_auto_ties(comps, judged) == judged

    def test_all_auto_no_judge_calls(self):
        comps = [_auto_comparison(0), _auto_comparison(1)]
        assert _merge_auto_ties(comps, []) == [_AUTO_TIE, _AUTO_TIE]


class TestTrimToBudget:
    """The budget caps judge calls; auto-ties are free and always survive."""

    def test_under_budget_unchanged(self):
        comps = [_make_comparison(i) for i in range(3)]  # 3 judgeable
        out, hit = _trim_to_budget(comps, 5)
        assert out is comps
        assert hit is False

    def test_caps_judged_but_keeps_every_auto_tie(self):
        # 3 judgeable + 2 auto-ties, budget of 1 judge call.
        comps = [
            _make_comparison(0),  # judged
            _auto_comparison(1),  # auto-tie
            _make_comparison(2),  # judged
            _auto_comparison(3),  # auto-tie
            _make_comparison(4),  # judged
        ]
        out, hit = _trim_to_budget(comps, 1)
        assert hit is True
        judged = [c for c in out if c.auto_result is None]
        autos = [c for c in out if c.auto_result is not None]
        assert len(judged) == 1  # capped at the budget
        assert len(autos) == 2  # every auto-tie kept — they cost no judge call
        assert judged[0].sample_idx == 0  # order preserved

    def test_all_auto_ties_never_trimmed(self):
        # Zero judge budget, but auto-ties don't count against it.
        comps = [_auto_comparison(i) for i in range(4)]
        out, hit = _trim_to_budget(comps, 0)
        assert out is comps
        assert hit is False


class TestAutoTieElo:
    def test_auto_tie_flows_into_elo_as_tie(self):
        """End to end: an auto-tie comparison scores as an ordinary tie."""
        comp = _auto_comparison()
        merged = _merge_auto_ties([comp], [])  # no judge calls made
        results = _convert_results([comp], merged)
        board = compute_elo(results, ["model-a", "model-b"])
        assert board.ties["model-a"] == 1
        assert board.ties["model-b"] == 1
        assert board.wins["model-a"] == 0
        assert board.wins["model-b"] == 0
        assert board.comparison_log[0]["agreement"] == "auto"


class TestBenchParser:
    def test_defaults(self):
        parser = build_parser()
        args = parser.parse_args(["bench", "user/imgs", "user/out"])
        assert args.command == "bench"
        assert args.input_dataset == "user/imgs"
        assert args.output_repo == "user/out"
        assert args.models is None
        assert args.judge_models is None
        assert args.criteria is None  # None sentinel; resolves to default later
        assert args.criteria_file is None
        assert args.max_samples is None
        assert args.seed == 42
        assert args.adaptive_strategy == "balanced"
        assert args.size_tie_ratio is None
        assert args.size_tie_min_samples == 10
        assert args.no_publish is False
        assert args.port == 7860
        assert args.host == "127.0.0.1"

    def test_flags(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "bench",
                "user/imgs",
                "user/out",
                "--models",
                "glm-ocr",
                "dots-ocr",
                "--judge-model",
                "novita:org/m",
                "--max-samples",
                "25",
                "--no-publish",
                "--port",
                "9000",
            ]
        )
        assert args.models == ["glm-ocr", "dots-ocr"]
        assert args.judge_models == ["novita:org/m"]
        assert args.max_samples == 25
        assert args.no_publish is True
        assert args.port == 9000


class TestCmdBench:
    """cmd_bench chains run → judge → view, threading shared flags."""

    def _patch(self, monkeypatch, run_jobs=None):
        from ocr_bench.run import JobRun

        calls: list[tuple[str, object]] = []
        jobs = run_jobs if run_jobs is not None else [JobRun("glm-ocr", "j1", "u", "completed")]

        def rec_run(a):
            calls.append(("run", a))
            return jobs

        monkeypatch.setattr(cli, "cmd_run", rec_run)
        monkeypatch.setattr(cli, "cmd_judge", lambda a: calls.append(("judge", a)))
        monkeypatch.setattr(cli, "cmd_view", lambda a: calls.append(("view", a)))
        return calls

    def test_phase_ordering(self, monkeypatch):
        calls = self._patch(monkeypatch)
        args = build_parser().parse_args(["bench", "user/imgs", "user/out"])
        cli.cmd_bench(args)
        assert [c[0] for c in calls] == ["run", "judge", "view"]
        run_a, judge_a, view_a = (c[1] for c in calls)
        assert run_a.command == "run"
        assert run_a.input_dataset == "user/imgs"
        assert run_a.output_repo == "user/out"
        assert run_a.no_wait is False  # bench waits for jobs before judging
        assert judge_a.command == "judge"
        assert judge_a.dataset == "user/out"
        assert judge_a.from_prs is True
        assert view_a.command == "view"
        assert view_a.results == "user/out-results"

    def test_no_publish_skips_view(self, monkeypatch):
        calls = self._patch(monkeypatch)
        args = build_parser().parse_args(["bench", "user/imgs", "user/out", "--no-publish"])
        cli.cmd_bench(args)
        assert [c[0] for c in calls] == ["run", "judge"]
        assert calls[1][1].no_publish is True

    def test_threads_shared_flags(self, monkeypatch):
        calls = self._patch(monkeypatch)
        args = build_parser().parse_args(
            [
                "bench",
                "user/imgs",
                "user/out",
                "--models",
                "glm-ocr",
                "dots-ocr",
                "--judge-model",
                "novita:org/m",
                "--max-samples",
                "25",
                "--seed",
                "7",
            ]
        )
        cli.cmd_bench(args)
        run_a, judge_a, _ = (c[1] for c in calls)
        assert run_a.models == ["glm-ocr", "dots-ocr"]
        assert run_a.max_samples == 25
        assert run_a.seed == 7
        assert judge_a.models == ["novita:org/m"]  # judge --model, dest=models
        assert judge_a.max_samples == 25
        assert judge_a.seed == 7

    def test_threads_adaptive_options_to_judge(self, monkeypatch):
        calls = self._patch(monkeypatch)
        args = build_parser().parse_args(
            [
                "bench",
                "user/imgs",
                "user/out",
                "--adaptive-strategy",
                "targeted",
                "--size-tie-ratio",
                "3",
                "--size-tie-min-samples",
                "5",
            ]
        )
        cli.cmd_bench(args)
        judge_a = calls[1][1]
        assert judge_a.adaptive_strategy == "targeted"
        assert judge_a.size_tie_ratio == 3.0
        assert judge_a.size_tie_min_samples == 5

    def test_threads_criteria_to_judge(self, monkeypatch):
        """bench --criteria reaches the judge phase's namespace."""
        calls = self._patch(monkeypatch)
        args = build_parser().parse_args(
            ["bench", "user/imgs", "user/out", "--criteria", "table-fidelity"]
        )
        cli.cmd_bench(args)
        judge_a = calls[1][1]
        assert judge_a.criteria == "table-fidelity"

    def test_criteria_defaults_to_default_in_judge(self, monkeypatch):
        # Neither flag given → bench forwards nothing → the judge phase applies
        # its own default (criteria=None sentinel, resolved to 'default').
        calls = self._patch(monkeypatch)
        args = build_parser().parse_args(["bench", "user/imgs", "user/out"])
        cli.cmd_bench(args)
        judge_a = calls[1][1]
        assert judge_a.criteria is None
        assert judge_a.criteria_file is None

    def test_threads_criteria_file_to_judge(self, monkeypatch, tmp_path):
        """bench --criteria-file is forwarded (not --criteria) to the judge phase.

        Uses a real, valid file since cmd_bench validates it before launching.
        """
        calls = self._patch(monkeypatch)
        f = tmp_path / "rubric.txt"
        f.write_text("Judge. A={ocr_text_a} B={ocr_text_b}")
        args = build_parser().parse_args(
            ["bench", "user/imgs", "user/out", "--criteria-file", str(f)]
        )
        cli.cmd_bench(args)
        judge_a = calls[1][1]
        assert judge_a.criteria_file == str(f)

    def test_invalid_criteria_file_errors_before_running(self, monkeypatch, tmp_path):
        """A bad custom prompt file fails fast — before any OCR job is launched."""
        calls = self._patch(monkeypatch)
        f = tmp_path / "bad.txt"
        f.write_text("missing the b placeholder: {ocr_text_a}")
        args = build_parser().parse_args(
            ["bench", "user/imgs", "user/out", "--criteria-file", str(f)]
        )
        with pytest.raises(DatasetError, match="ocr_text_b"):
            cli.cmd_bench(args)
        assert calls == []  # never reached the run phase

    def test_conflicting_criteria_flags_error_before_running(self, monkeypatch):
        """bench --criteria + --criteria-file fails fast (before launching jobs)."""
        calls = self._patch(monkeypatch)
        args = build_parser().parse_args(
            ["bench", "user/imgs", "user/out", "--criteria", "default",
             "--criteria-file", "r.txt"]
        )
        with pytest.raises(DatasetError, match="mutually exclusive"):
            cli.cmd_bench(args)
        assert calls == []  # nothing ran — not even the OCR phase

    def test_aborts_when_a_job_fails(self, monkeypatch, capsys):
        """A failed OCR job must stop bench before judging — a partial
        leaderboard is the worst outcome for the 'just try it' path."""
        from ocr_bench.run import JobRun

        run_jobs = [
            JobRun("glm-ocr", "j1", "u", "completed"),
            JobRun("dots-ocr", "j2", "u", "error"),
        ]
        calls = self._patch(monkeypatch, run_jobs=run_jobs)
        args = build_parser().parse_args(["bench", "user/imgs", "user/out"])
        cli.cmd_bench(args)
        # Only the run phase ran — no judge, no view.
        assert [c[0] for c in calls] == ["run"]
        out = capsys.readouterr().out
        assert "Aborting" in out
        assert "dots-ocr" in out  # names the failed model


class TestCmdRun:
    """cmd_run returns the launched jobs and prints a per-job status summary."""

    def test_list_models_needs_no_repo_arguments(self, capsys):
        args = build_parser().parse_args(["run", "--list-models"])
        assert cli.cmd_run(args) == []
        out = capsys.readouterr().out
        assert "falcon-ocr" in out
        assert "ovis-ocr2" in out

    def test_missing_repo_arguments_fails_cleanly(self, capsys):
        args = build_parser().parse_args(["run"])
        with pytest.raises(SystemExit) as exc:
            cli.cmd_run(args)
        assert exc.value.code == 2
        assert "requires INPUT_DATASET and OUTPUT_REPO" in capsys.readouterr().out

    def _run(self, monkeypatch, statuses):
        from ocr_bench.run import JobRun

        launched = [
            JobRun(slug, f"j{i}", f"u{i}")
            for i, slug in enumerate(["glm-ocr", "dots-ocr"])
        ]

        def fake_launch(*a, **k):
            return launched

        def fake_poll(jobs, **k):
            for job, status in zip(jobs, statuses):
                job.status = status
            return jobs

        monkeypatch.setattr("ocr_bench.run.launch_ocr_jobs", fake_launch)
        monkeypatch.setattr("ocr_bench.run.poll_jobs", fake_poll)
        args = build_parser().parse_args(
            ["run", "in/ds", "out/repo", "--models", "glm-ocr", "dots-ocr"]
        )
        return cli.cmd_run(args), launched

    def test_all_completed_returns_jobs_and_eval_hint(self, monkeypatch, capsys):
        result, launched = self._run(monkeypatch, ["completed", "completed"])
        assert result is launched
        out = capsys.readouterr().out
        assert "All 2 job(s) completed." in out
        assert "Evaluate:" in out

    def test_failed_job_flagged_and_no_eval_hint(self, monkeypatch, capsys):
        result, launched = self._run(monkeypatch, ["completed", "error"])
        assert result is launched
        out = capsys.readouterr().out
        assert "did not complete" in out
        assert "Evaluate:" not in out  # don't suggest judging a partial run


class TestCmdJudgeEmptyGuard:
    def test_adaptive_all_filtered_does_not_publish_empty_board(self, monkeypatch, capsys):
        """Aggressive filtering can leave the adaptive loop with zero results;
        it must print the no-comparisons message, not an all-1500 board."""
        from PIL import Image

        ds = [
            {"image": Image.new("RGB", (20, 20)), "a": "x", "b": "y"}
            for _ in range(4)
        ]

        def fake_flat(dataset, split="train", columns=None):
            return ds, {"a": "ModelA", "b": "ModelB"}

        monkeypatch.setattr(cli, "load_flat_dataset", fake_flat)
        # --no-publish keeps it offline; adaptive is on by default. Every text is
        # 1 char (< default min_chars) so all batches filter to nothing.
        args = build_parser().parse_args(
            ["judge", "user/ds", "--columns", "a", "b", "--no-publish"]
        )
        cli.cmd_judge(args)
        out = capsys.readouterr().out
        assert "No valid comparisons" in out
        assert "OCR Model Leaderboard" not in out
