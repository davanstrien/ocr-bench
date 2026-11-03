"""Tests for ocr_bench.run — model registry, job launching, polling."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ocr_bench.run import (
    DEFAULT_MODELS,
    MODEL_REGISTRY,
    JobRun,
    ModelConfig,
    build_script_args,
    failed_jobs,
    launch_ocr_jobs,
    launch_settings,
    list_models,
    parse_hf_jobs_header,
    poll_jobs,
)

# Shape of the uv-scripts/ocr recipe headers (dots-ocr.py, 2026-11).
VLLM_HEADER_SCRIPT = '''# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "datasets>=4.0.0",
#     "huggingface-hub",
# ]
#
# [tool.hf-jobs]
# image = "vllm/vllm-openai:v0.29.0"
# python = "/usr/bin/python3"
# env = { PYTHONPATH = "/usr/local/lib/python3.12/dist-packages" }
# flavor = "a10g-small"
# secrets = ["HF_TOKEN"]
# ///

"""Recipe docstring."""
import vllm
'''

VLLM_HEADER = {
    "image": "vllm/vllm-openai:v0.29.0",
    "python": "/usr/bin/python3",
    "env": {"PYTHONPATH": "/usr/local/lib/python3.12/dist-packages"},
    "flavor": "a10g-small",
    "secrets": ["HF_TOKEN"],
}


@pytest.fixture(autouse=True)
def _no_header_fetch():
    """Keep launch tests offline; tests that need a header patch it themselves."""
    with patch("ocr_bench.run.fetch_hf_jobs_header", return_value={}) as fetch:
        yield fetch


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

    def test_image_mode_fields_default_none(self):
        cfg = ModelConfig(
            script="https://example.com/script.py",
            model_id="org/model",
            size="1B",
        )
        assert cfg.image is None
        assert cfg.python is None
        assert cfg.env is None


class TestModelRegistry:
    def test_has_core_models(self):
        assert len(MODEL_REGISTRY) == 16

    def test_default_models_exist_in_registry(self):
        for slug in DEFAULT_MODELS:
            assert slug in MODEL_REGISTRY

    def test_all_configs_have_required_fields(self):
        for slug, cfg in MODEL_REGISTRY.items():
            assert cfg.script.startswith("https://"), f"{slug} script not HTTPS"
            assert cfg.model_id, f"{slug} missing model_id"
            assert cfg.size, f"{slug} missing size"
            assert cfg.default_flavor, f"{slug} missing default_flavor"

    def test_deepseek_has_prompt_mode_free(self):
        cfg = MODEL_REGISTRY["deepseek-ocr"]
        assert "--prompt-mode" in cfg.default_args
        assert "free" in cfg.default_args

    def test_compact_models_use_standard_l4_launch(self):
        expected = {
            "falcon-ocr": ("tiiuae/Falcon-OCR", "0.3B", "falcon-ocr.py"),
            "ovis-ocr2": ("ATH-MaaS/OvisOCR2", "0.9B", "ovis-ocr2.py"),
        }
        for slug, (model_id, size, script_name) in expected.items():
            cfg = MODEL_REGISTRY[slug]
            assert cfg.model_id == model_id
            assert cfg.size == size
            assert cfg.script.endswith(f"/{script_name}")
            assert cfg.default_flavor == "l4x1"
            assert cfg.default_args == []
            assert cfg.image is None
            assert cfg.python is None
            assert cfg.env is None

    def test_compact_models_are_opt_in(self):
        assert "falcon-ocr" not in DEFAULT_MODELS
        assert "ovis-ocr2" not in DEFAULT_MODELS

    def test_image_mode_models_configured(self):
        # NuExtract3 and PaddleOCR-VL-1.6 need the prebuilt-kernel image on a100.
        for slug in ("nuextract3", "paddleocr-vl-1.6"):
            cfg = MODEL_REGISTRY[slug]
            assert cfg.image == "vllm/vllm-openai:latest", slug
            assert cfg.python == "/usr/bin/python3", slug
            assert cfg.env == {"PYTHONPATH": "/usr/local/lib/python3.12/dist-packages"}, slug
            assert cfg.default_flavor == "a100-large", slug

    def test_image_mode_models_not_in_defaults(self):
        # Opt-in only — they need a100-large and are slower than the default set.
        assert "nuextract3" not in DEFAULT_MODELS
        assert "paddleocr-vl-1.6" not in DEFAULT_MODELS


class TestListModels:
    def test_returns_sorted_slugs(self):
        slugs = list_models()
        assert slugs == sorted(slugs)
        assert len(slugs) == len(MODEL_REGISTRY)


class TestBuildScriptArgs:
    def test_basic_args(self):
        args = build_script_args("input/ds", "output/repo", "glm-ocr")
        assert args == [
            "input/ds",
            "output/repo",
            "--config",
            "glm-ocr",
            "--create-pr",
            "--split",
            "train",
        ]

    def test_non_default_split(self):
        args = build_script_args("in", "out", "x", split="validation")
        assert args[args.index("--split") + 1] == "validation"

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

        assert len(jobs) == 5
        assert mock_api.run_uv_job.call_count == 5
        for job in jobs:
            assert isinstance(job, JobRun)
            assert job.status == "running"

    @patch("ocr_bench.run.get_token", return_value="fake-token")
    def test_launches_subset(self, mock_token):
        mock_api = MagicMock()
        mock_job = MagicMock()
        mock_job.id = "job-1"
        mock_job.url = "https://example.com"
        mock_api.run_uv_job.return_value = mock_job

        jobs = launch_ocr_jobs(
            "input/ds", "output/repo", models=["glm-ocr", "dots-ocr"], api=mock_api
        )

        assert len(jobs) == 2
        assert jobs[0].model_slug == "glm-ocr"
        assert jobs[1].model_slug == "dots-ocr"

    @patch("ocr_bench.run.get_token", return_value="fake-token")
    def test_passes_split_to_every_job(self, mock_token):
        mock_api = MagicMock()
        mock_job = MagicMock(id="job-1", url="https://example.com")
        mock_api.run_uv_job.return_value = mock_job

        launch_ocr_jobs(
            "input/ds",
            "output/repo",
            models=["glm-ocr", "dots-ocr"],
            split="validation",
            api=mock_api,
        )

        assert mock_api.run_uv_job.call_count == 2
        for call in mock_api.run_uv_job.call_args_list:
            script_args = call.kwargs["script_args"]
            assert script_args[script_args.index("--split") + 1] == "validation"

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

        launch_ocr_jobs("in", "out", models=["glm-ocr"], flavor_override="a100-large", api=mock_api)

        call_kwargs = mock_api.run_uv_job.call_args
        assert call_kwargs.kwargs["flavor"] == "a100-large"

    @patch("ocr_bench.run.get_token", return_value="fake-token")
    def test_image_mode_passes_image_python_env(self, mock_token):
        mock_api = MagicMock()
        mock_job = MagicMock()
        mock_job.id = "j1"
        mock_job.url = "https://example.com"
        mock_api.run_uv_job.return_value = mock_job

        launch_ocr_jobs("in", "out", models=["nuextract3"], api=mock_api)

        kwargs = mock_api.run_uv_job.call_args.kwargs
        assert kwargs["image"] == "vllm/vllm-openai:latest"
        assert kwargs["python"] == "/usr/bin/python3"
        assert kwargs["env"] == {"PYTHONPATH": "/usr/local/lib/python3.12/dist-packages"}

    @patch("ocr_bench.run.get_token", return_value="fake-token")
    def test_script_without_header_image_omits_image_kwargs(self, mock_token):
        # No registry image and no header image: the plain uv image, as before.
        mock_api = MagicMock()
        mock_api.run_uv_job.return_value = MagicMock(id="j1", url="https://example.com")

        launch_ocr_jobs("in", "out", models=["glm-ocr"], api=mock_api)

        kwargs = mock_api.run_uv_job.call_args.kwargs
        assert "image" not in kwargs
        assert "python" not in kwargs
        assert "env" not in kwargs

    @patch("ocr_bench.run.get_token", return_value="fake-token")
    def test_header_image_is_passed_to_run_uv_job(self, mock_token, _no_header_fetch):
        # HfApi.run_uv_job ignores [tool.hf-jobs]; ocr-bench must apply it, or
        # recipes that moved vLLM into their header image start without vLLM.
        _no_header_fetch.return_value = VLLM_HEADER
        mock_api = MagicMock()
        mock_api.run_uv_job.return_value = MagicMock(id="j1", url="https://example.com")

        launch_ocr_jobs("in", "out", models=["dots-ocr"], api=mock_api)

        _no_header_fetch.assert_called_once_with(MODEL_REGISTRY["dots-ocr"].script)
        kwargs = mock_api.run_uv_job.call_args.kwargs
        assert kwargs["image"] == "vllm/vllm-openai:v0.29.0"
        assert kwargs["python"] == "/usr/bin/python3"
        assert kwargs["env"] == {"PYTHONPATH": "/usr/local/lib/python3.12/dist-packages"}
        # Hardware stays ocr-bench's choice, not the header's.
        assert kwargs["flavor"] == MODEL_REGISTRY["dots-ocr"].default_flavor

    @patch("ocr_bench.run.get_token", return_value="fake-token")
    def test_registry_image_skips_header_fetch(self, mock_token, _no_header_fetch):
        mock_api = MagicMock()
        mock_api.run_uv_job.return_value = MagicMock(id="j1", url="https://example.com")

        launch_ocr_jobs("in", "out", models=["nuextract3"], api=mock_api)

        _no_header_fetch.assert_not_called()

    @patch("ocr_bench.run.get_token", return_value="fake-token")
    def test_unreadable_header_launches_nothing(self, mock_token, _no_header_fetch):
        def fetch(url):
            if "dots-ocr" in url:
                raise RuntimeError("Could not read the [tool.hf-jobs] header")
            return {}

        _no_header_fetch.side_effect = fetch
        mock_api = MagicMock()

        with pytest.raises(RuntimeError, match="hf-jobs"):
            launch_ocr_jobs("in", "out", models=["glm-ocr", "dots-ocr"], api=mock_api)

        mock_api.run_uv_job.assert_not_called()


class TestParseHfJobsHeader:
    def test_reads_recipe_header(self):
        assert parse_hf_jobs_header(VLLM_HEADER_SCRIPT) == VLLM_HEADER

    def test_script_without_inline_metadata(self):
        assert parse_hf_jobs_header("import sys\nprint('hi')\n") == {}

    def test_script_block_without_hf_jobs_table(self):
        script = '# /// script\n# dependencies = ["pillow"]\n# ///\nimport PIL\n'
        assert parse_hf_jobs_header(script) == {}

    def test_ignores_other_block_types(self):
        script = (
            '# /// other\n# [tool.hf-jobs]\n# image = "wrong"\n# ///\n'
            '# /// script\n# [tool.hf-jobs]\n# image = "right"\n# ///\n'
        )
        assert parse_hf_jobs_header(script) == {"image": "right"}


class TestLaunchSettings:
    def test_registry_image_wins_over_header(self):
        config = MODEL_REGISTRY["nuextract3"]
        settings = launch_settings(config, header=VLLM_HEADER)
        assert settings["image"] == config.image
        assert settings["env"] == config.env

    def test_header_image_python_env_taken_together(self):
        config = ModelConfig(script="s", model_id="m", size="1B")
        settings = launch_settings(config, header=VLLM_HEADER)
        assert settings == {
            "flavor": "l4x1",
            "image": VLLM_HEADER["image"],
            "python": VLLM_HEADER["python"],
            "env": VLLM_HEADER["env"],
        }

    def test_header_without_image_ignores_python_and_env(self):
        config = ModelConfig(script="s", model_id="m", size="1B")
        header = {"python": "/usr/bin/python3", "env": {"A": "1"}, "flavor": "a10g-small"}
        assert launch_settings(config, header=header) == {"flavor": "l4x1"}

    def test_flavor_override_wins(self):
        config = ModelConfig(script="s", model_id="m", size="1B")
        settings = launch_settings(config, flavor_override="a100-large", header=VLLM_HEADER)
        assert settings["flavor"] == "a100-large"


class TestPollJobs:
    @patch("ocr_bench.run.time.sleep")
    def test_polls_until_complete(self, mock_sleep):
        mock_api = MagicMock()
        mock_info = MagicMock()
        mock_info.status.stage = "COMPLETED"
        mock_api.inspect_job.return_value = mock_info

        jobs = [
            JobRun(model_slug="glm-ocr", job_id="j1", job_url="url1"),
            JobRun(model_slug="dots-ocr", job_id="j2", job_url="url2"),
        ]

        result = poll_jobs(jobs, interval=1, api=mock_api)
        assert all(j.status == "completed" for j in result)

    @patch("ocr_bench.run.time.sleep")
    def test_handles_error_status(self, mock_sleep):
        mock_api = MagicMock()
        mock_info = MagicMock()
        mock_info.status.stage = "ERROR"
        mock_api.inspect_job.return_value = mock_info

        jobs = [JobRun(model_slug="glm-ocr", job_id="j1", job_url="url1")]
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

        jobs = [JobRun(model_slug="glm-ocr", job_id="j1", job_url="url1")]
        result = poll_jobs(jobs, interval=1, api=mock_api)
        assert result[0].status == "completed"
        assert mock_sleep.call_count == 2


class TestFailedJobs:
    def test_all_completed_none_failed(self):
        jobs = [JobRun("a", "1", "u", "completed"), JobRun("b", "2", "u", "completed")]
        assert failed_jobs(jobs) == []

    def test_returns_non_completed_in_order(self):
        jobs = [
            JobRun("a", "1", "u", "completed"),
            JobRun("b", "2", "u", "error"),
            JobRun("c", "3", "u", "canceled"),
        ]
        assert [j.model_slug for j in failed_jobs(jobs)] == ["b", "c"]

    def test_running_counts_as_not_completed(self):
        # A never-polled job keeps status "running" — not a success.
        assert failed_jobs([JobRun("a", "1", "u")]) == [JobRun("a", "1", "u")]


class TestCLIParser:
    def test_run_subcommand_parses(self):
        from ocr_bench.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["run", "input/ds", "output/repo", "--max-samples", "50"])
        assert args.command == "run"
        assert args.input_dataset == "input/ds"
        assert args.output_repo == "output/repo"
        assert args.max_samples == 50

    def test_run_list_models_without_repo_arguments(self):
        from ocr_bench.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["run", "--list-models"])
        assert args.list_models is True
        assert args.input_dataset is None
        assert args.output_repo is None

    def test_run_dry_run(self):
        from ocr_bench.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["run", "in", "out", "--dry-run"])
        assert args.dry_run is True

    def test_run_models_flag(self):
        from ocr_bench.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["run", "in", "out", "--models", "glm-ocr", "dots-ocr"])
        assert args.models == ["glm-ocr", "dots-ocr"]
