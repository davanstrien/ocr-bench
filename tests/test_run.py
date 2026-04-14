"""Tests for ocr_bench.run — model registry, job launching, polling."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from ocr_bench.run import (
    DEFAULT_TASK_PROMPT,
    DEFAULT_MODELS,
    MODEL_REGISTRY,
    JobRun,
    ModelConfig,
    build_script_args,
    launch_ocr_jobs,
    list_models,
    poll_jobs,
)


class TestModelConfig:
    def test_defaults(self):
        cfg = ModelConfig(
            script="https://example.com/script.py",
            model_id="org/model",
            size="1B",
        )
        assert cfg.default_flavor == "l4x1"
        assert cfg.default_args == []

    def test_custom_args(self):
        cfg = ModelConfig(
            script="https://example.com/script.py",
            model_id="org/model",
            size="4B",
            default_args=["--prompt-mode", "free"],
        )
        assert cfg.default_args == ["--prompt-mode", "free"]


class TestModelRegistry:
    def test_has_core_models(self):
        assert len(MODEL_REGISTRY) == 3

    def test_default_models_exist_in_registry(self):
        for slug in DEFAULT_MODELS:
            assert slug in MODEL_REGISTRY

    def test_all_configs_have_required_fields(self):
        for slug, cfg in MODEL_REGISTRY.items():
            assert cfg.script.startswith("https://"), f"{slug} script not HTTPS"
            assert cfg.model_id, f"{slug} missing model_id"
            assert cfg.size, f"{slug} missing size"
            assert cfg.default_flavor, f"{slug} missing default_flavor"

    def test_qwen_model_exists(self):
        cfg = MODEL_REGISTRY["qwen3-vl-4b-instruct"]
        assert cfg.model_id == "Qwen/Qwen3-VL-4B-Instruct"


class TestListModels:
    def test_returns_sorted_slugs(self):
        slugs = list_models()
        assert slugs == sorted(slugs)
        assert len(slugs) == len(MODEL_REGISTRY)


class TestBuildScriptArgs:
    def test_basic_args(self):
        args = build_script_args("input/ds", "output/repo", "glm-ocr")
        assert args[:5] == ["input/ds", "output/repo", "--config", "glm-ocr", "--create-pr"]
        assert "--image-column" in args
        assert "image_uri" in args
        assert "--prompt" in args
        assert "Extract the document title from this cover page." in args[args.index("--prompt") + 1]

    def test_max_samples(self):
        args = build_script_args("in", "out", "x", max_samples=50)
        assert "--max-samples" in args
        idx = args.index("--max-samples")
        assert args[idx + 1] == "50"

    def test_shuffle(self):
        args = build_script_args("in", "out", "x", shuffle=True)
        assert "--shuffle" in args

    def test_non_default_seed(self):
        args = build_script_args("in", "out", "x", seed=123)
        assert "--seed" in args
        idx = args.index("--seed")
        assert args[idx + 1] == "123"

    def test_default_seed_omitted(self):
        args = build_script_args("in", "out", "x", seed=42)
        assert "--seed" not in args

    def test_extra_args(self):
        args = build_script_args("in", "out", "x", extra_args=["--prompt-mode", "free"])
        assert "--prompt-mode" in args
        assert "free" in args


class TestLaunchOcrJobs:
    @patch("ocr_bench.run.get_token", return_value="fake-token")
    def test_launches_all_default_models(self, mock_token):
        mock_api = MagicMock()
        mock_job = MagicMock()
        mock_job.id = "job-123"
        mock_job.url = "https://huggingface.co/jobs/job-123"
        mock_api.run_uv_job.return_value = mock_job

        jobs = launch_ocr_jobs("input/ds", "output/repo", api=mock_api)

        assert len(jobs) == 3
        assert mock_api.run_uv_job.call_count == 3
        for job in jobs:
            assert isinstance(job, JobRun)
            assert job.status == "running"

    def test_prompt_contains_degree_and_discipline_constraints(self):
        assert "Thèse d'État" in DEFAULT_TASK_PROMPT
        assert "Informatique, information, généralités" in DEFAULT_TASK_PROMPT

    @patch("ocr_bench.run.get_token", return_value="fake-token")
    def test_launches_subset(self, mock_token):
        mock_api = MagicMock()
        mock_job = MagicMock()
        mock_job.id = "job-1"
        mock_job.url = "https://example.com"
        mock_api.run_uv_job.return_value = mock_job

        jobs = launch_ocr_jobs(
            "input/ds",
            "output/repo",
            models=["qwen3-vl-4b-instruct", "nanonets-ocr2-3b"],
            api=mock_api,
        )

        assert len(jobs) == 2
        assert jobs[0].model_slug == "qwen3-vl-4b-instruct"
        assert jobs[1].model_slug == "nanonets-ocr2-3b"

    @patch("ocr_bench.run.get_token", return_value="fake-token")
    def test_unknown_model_raises(self, mock_token):
        mock_api = MagicMock()
        try:
            launch_ocr_jobs("in", "out", models=["nonexistent"], api=mock_api)
            assert False, "Should have raised"
        except ValueError as e:
            assert "nonexistent" in str(e)

    @patch("ocr_bench.run.get_token", return_value=None)
    def test_no_token_raises(self, mock_token):
        mock_api = MagicMock()
        try:
            launch_ocr_jobs("in", "out", api=mock_api)
            assert False, "Should have raised"
        except RuntimeError as e:
            assert "token" in str(e).lower()

    @patch("ocr_bench.run.get_token", return_value="fake-token")
    def test_flavor_override(self, mock_token):
        mock_api = MagicMock()
        mock_job = MagicMock()
        mock_job.id = "j1"
        mock_job.url = "https://example.com"
        mock_api.run_uv_job.return_value = mock_job

        launch_ocr_jobs(
            "in",
            "out",
            models=["qwen3-vl-4b-instruct"],
            flavor_override="a100-large",
            api=mock_api,
        )

        call_kwargs = mock_api.run_uv_job.call_args
        assert call_kwargs.kwargs["flavor"] == "a100-large"


class TestPollJobs:
    @patch("ocr_bench.run.time.sleep")
    def test_polls_until_complete(self, mock_sleep):
        mock_api = MagicMock()
        mock_info = MagicMock()
        mock_info.status.stage = "COMPLETED"
        mock_api.inspect_job.return_value = mock_info

        jobs = [
            JobRun(model_slug="qwen3-vl-4b-instruct", job_id="j1", job_url="url1"),
            JobRun(model_slug="nanonets-ocr2-3b", job_id="j2", job_url="url2"),
        ]

        result = poll_jobs(jobs, interval=1, api=mock_api)
        assert all(j.status == "completed" for j in result)

    @patch("ocr_bench.run.time.sleep")
    def test_handles_error_status(self, mock_sleep):
        mock_api = MagicMock()
        mock_info = MagicMock()
        mock_info.status.stage = "ERROR"
        mock_api.inspect_job.return_value = mock_info

        jobs = [JobRun(model_slug="qwen3-vl-4b-instruct", job_id="j1", job_url="url1")]
        result = poll_jobs(jobs, interval=1, api=mock_api)
        assert result[0].status == "error"

    @patch("ocr_bench.run.time.sleep")
    def test_multiple_poll_rounds(self, mock_sleep):
        mock_api = MagicMock()

        running = MagicMock()
        running.status.stage = "RUNNING"
        done = MagicMock()
        done.status.stage = "COMPLETED"

        # First call returns running, second returns completed
        mock_api.inspect_job.side_effect = [running, done]

        jobs = [JobRun(model_slug="qwen3-vl-4b-instruct", job_id="j1", job_url="url1")]
        result = poll_jobs(jobs, interval=1, api=mock_api)
        assert result[0].status == "completed"
        assert mock_sleep.call_count == 2


class TestCLIParser:
    def test_run_subcommand_parses(self):
        from ocr_bench.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["run", "input/ds", "output/repo", "--max-samples", "50"])
        assert args.command == "run"
        assert args.input_dataset == "input/ds"
        assert args.output_repo == "output/repo"
        assert args.max_samples == 50

    def test_run_list_models(self):
        from ocr_bench.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["run", "in", "out", "--list-models"])
        assert args.list_models is True

    def test_run_dry_run(self):
        from ocr_bench.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["run", "in", "out", "--dry-run"])
        assert args.dry_run is True

    def test_run_models_flag(self):
        from ocr_bench.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(
            ["run", "in", "out", "--models", "qwen3-vl-4b-instruct", "gemma-4-e4b-it"]
        )
        assert args.models == ["qwen3-vl-4b-instruct", "gemma-4-e4b-it"]
