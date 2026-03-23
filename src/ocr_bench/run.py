"""OCR model orchestration — launch HF Jobs for multiple OCR models."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import structlog
from huggingface_hub import HfApi, get_token

logger = structlog.get_logger()


@dataclass
class ModelConfig:
    """Configuration for a single OCR model."""

    script: str
    model_id: str
    size: str
    default_flavor: str = "l4x1"
    default_args: list[str] = field(default_factory=list)


MODEL_REGISTRY: dict[str, ModelConfig] = {
    "glm-ocr": ModelConfig(
        script="https://huggingface.co/datasets/uv-scripts/ocr/raw/main/glm-ocr.py",
        model_id="zai-org/GLM-OCR",
        size="0.9B",
        default_flavor="l4x1",
    ),
    "deepseek-ocr": ModelConfig(
        script="https://huggingface.co/datasets/uv-scripts/ocr/raw/main/deepseek-ocr-vllm.py",
        model_id="deepseek-ai/DeepSeek-OCR",
        size="4B",
        default_flavor="l4x1",
        default_args=["--prompt-mode", "free"],
    ),
    "lighton-ocr-2": ModelConfig(
        script="https://huggingface.co/datasets/uv-scripts/ocr/raw/main/lighton-ocr2.py",
        model_id="lightonai/LightOnOCR-2-1B",
        size="1B",
        default_flavor="a100-large",
    ),
    "dots-ocr": ModelConfig(
        script="https://huggingface.co/datasets/uv-scripts/ocr/raw/main/dots-ocr.py",
        model_id="rednote-hilab/dots.ocr",
        size="1.7B",
        default_flavor="l4x1",
    ),
    "firered-ocr": ModelConfig(
        script="https://huggingface.co/datasets/uv-scripts/ocr/raw/main/firered-ocr.py",
        model_id="FireRedTeam/FireRed-OCR",
        size="2.1B",
        default_flavor="l4x1",
    ),
    "qianfan-ocr": ModelConfig(
        script="https://huggingface.co/datasets/uv-scripts/ocr/raw/main/qianfan-ocr.py",
        model_id="baidu/Qianfan-OCR",
        size="4.7B",
        default_flavor="l4x1",
    ),
    "dots-mocr": ModelConfig(
        script="https://huggingface.co/datasets/uv-scripts/ocr/raw/main/dots-mocr.py",
        model_id="rednote-hilab/dots.mocr",
        size="3B",
        default_flavor="l4x1",
    ),
}

DEFAULT_MODELS = ["glm-ocr", "deepseek-ocr", "lighton-ocr-2", "dots-ocr", "firered-ocr"]


@dataclass
class JobRun:
    """Tracks a launched HF Job."""

    model_slug: str
    job_id: str
    job_url: str
    status: str = "running"


def list_models() -> list[str]:
    """Return sorted list of available model slugs."""
    return sorted(MODEL_REGISTRY.keys())


def build_script_args(
    input_dataset: str,
    output_repo: str,
    config_name: str,
    *,
    max_samples: int | None = None,
    shuffle: bool = False,
    seed: int = 42,
    extra_args: list[str] | None = None,
) -> list[str]:
    """Build the script_args list for run_uv_job."""
    args = [
        input_dataset,
        output_repo,
        "--config",
        config_name,
        "--create-pr",
    ]
    if max_samples is not None:
        args += ["--max-samples", str(max_samples)]
    if shuffle:
        args.append("--shuffle")
    if seed != 42:
        args += ["--seed", str(seed)]
    if extra_args:
        args += extra_args
    return args


def launch_ocr_jobs(
    input_dataset: str,
    output_repo: str,
    *,
    models: list[str] | None = None,
    max_samples: int | None = None,
    split: str = "train",
    shuffle: bool = False,
    seed: int = 42,
    flavor_override: str | None = None,
    timeout: str = "4h",
    api: HfApi | None = None,
) -> list[JobRun]:
    """Launch HF Jobs for each model. Returns list of JobRun tracking objects."""
    if api is None:
        api = HfApi()

    token = get_token()
    if not token:
        raise RuntimeError("No HF token found. Log in with `hf login` or set HF_TOKEN.")

    selected = models or DEFAULT_MODELS
    for slug in selected:
        if slug not in MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model: {slug}. Available: {', '.join(MODEL_REGISTRY.keys())}"
            )

    jobs: list[JobRun] = []
    for slug in selected:
        config = MODEL_REGISTRY[slug]
        flavor = flavor_override or config.default_flavor
        script_args = build_script_args(
            input_dataset,
            output_repo,
            slug,
            max_samples=max_samples,
            shuffle=shuffle,
            seed=seed,
            extra_args=config.default_args or None,
        )

        logger.info("launching_job", model=slug, flavor=flavor, script=config.script)
        job = api.run_uv_job(
            script=config.script,
            script_args=script_args,
            flavor=flavor,
            secrets={"HF_TOKEN": token},
            timeout=timeout,
        )
        jobs.append(JobRun(model_slug=slug, job_id=job.id, job_url=job.url))
        logger.info("job_launched", model=slug, job_id=job.id, url=job.url)

    return jobs


_TERMINAL_STAGES = frozenset({"COMPLETED", "ERROR", "CANCELED", "DELETED"})


def poll_jobs(
    jobs: list[JobRun],
    *,
    interval: int = 30,
    api: HfApi | None = None,
) -> list[JobRun]:
    """Poll until all jobs complete or fail. Updates status in-place and returns the list."""
    if api is None:
        api = HfApi()

    pending = {j.job_id: j for j in jobs if j.status == "running"}

    while pending:
        time.sleep(interval)
        still_running: dict[str, JobRun] = {}
        for job_id, job_run in pending.items():
            info = api.inspect_job(job_id=job_id)
            stage = info.status.stage
            if stage in _TERMINAL_STAGES:
                job_run.status = stage.lower()
                logger.info("job_finished", model=job_run.model_slug, status=job_run.status)
            else:
                still_running[job_id] = job_run
        pending = still_running
        if pending:
            slugs = [j.model_slug for j in pending.values()]
            logger.info("jobs_pending", models=slugs)

    return jobs
