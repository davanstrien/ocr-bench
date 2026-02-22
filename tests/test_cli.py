"""Tests for the CLI."""

from __future__ import annotations

import pytest

from ocr_bench.cli import build_parser


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
        assert args.merge_prs is False
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

    def test_browse_subcommand_defaults(self):
        parser = build_parser()
        args = parser.parse_args(["browse", "user/results"])
        assert args.command == "browse"
        assert args.results == "user/results"
        assert args.port == 7860
        assert args.share is False

    def test_browse_with_options(self):
        parser = build_parser()
        args = parser.parse_args(["browse", "user/results", "--port", "8080", "--share"])
        assert args.port == 8080
        assert args.share is True

    def test_browse_help_exits(self):
        parser = build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["browse", "--help"])
        assert exc_info.value.code == 0
