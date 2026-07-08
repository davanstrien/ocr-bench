"""Tests for the CLI."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest
from openai import OpenAIError

from ocr_bench import cli
from ocr_bench.cli import (
    _convert_results,
    _refresh_viewer_space,
    _resolve_results_repo,
    build_parser,
)
from ocr_bench.judge import Comparison


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
