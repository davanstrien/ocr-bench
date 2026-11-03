"""OCR model orchestration — launch HF Jobs for multiple OCR models."""

from __future__ import annotations

import re
import time
import tomllib
from dataclasses import dataclass, field
from typing import Any

import structlog
from huggingface_hub import HfApi, get_token
from huggingface_hub.utils import get_session

logger = structlog.get_logger()


@dataclass
class ModelConfig:
    """Configuration for a single OCR model.

    ``image`` / ``python`` / ``env`` override the recipe's own launch settings.
    Leave them ``None`` to use the image, interpreter and environment named in
    the recipe's ``[tool.hf-jobs]`` header (see :func:`launch_settings`); set
    them only where ocr-bench must run a model differently from its recipe.
    """

    script: str
    model_id: str
    size: str
    default_flavor: str = "l4x1"
    default_args: list[str] = field(default_factory=list)
    image: str | None = None
    python: str | None = None
    env: dict[str, str] | None = None


# Image-mode invocation for models needing prebuilt CUDA kernels (Qwen3.5 /
# flashinfer). The default uv-script image has no ``nvcc`` so flashinfer's JIT
# compile fails at vLLM warmup; the vllm/vllm-openai image ships them prebuilt.
# ``python`` points at that image's interpreter and ``env`` puts its site-packages
# on the path so ``uv run`` reuses them instead of rebuilding.
_VLLM_OPENAI_IMAGE = "vllm/vllm-openai:latest"
_VLLM_OPENAI_PYTHON = "/usr/bin/python3"
_VLLM_OPENAI_ENV = {"PYTHONPATH": "/usr/local/lib/python3.12/dist-packages"}


MODEL_REGISTRY: dict[str, ModelConfig] = {
    "falcon-ocr": ModelConfig(
        script="https://huggingface.co/datasets/uv-scripts/ocr/raw/main/falcon-ocr.py",
        model_id="tiiuae/Falcon-OCR",
        size="0.3B",
        default_flavor="l4x1",
    ),
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
    "ovis-ocr2": ModelConfig(
        script="https://huggingface.co/datasets/uv-scripts/ocr/raw/main/ovis-ocr2.py",
        model_id="ATH-MaaS/OvisOCR2",
        size="0.9B",
        default_flavor="l4x1",
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
    # Image-mode models (Qwen3.5 / flashinfer) — need the vllm/vllm-openai image.
    "nuextract3": ModelConfig(
        script="https://huggingface.co/datasets/uv-scripts/ocr/raw/main/nuextract3.py",
        model_id="numind/NuExtract3",
        size="4B",
        default_flavor="a100-large",
        image=_VLLM_OPENAI_IMAGE,
        python=_VLLM_OPENAI_PYTHON,
        env=_VLLM_OPENAI_ENV,
    ),
    "paddleocr-vl-1.6": ModelConfig(
        script="https://huggingface.co/datasets/uv-scripts/ocr/raw/main/paddleocr-vl-1.6.py",
        model_id="PaddlePaddle/PaddleOCR-VL-1.6",
        size="0.9B",
        default_flavor="a100-large",
        image=_VLLM_OPENAI_IMAGE,
        python=_VLLM_OPENAI_PYTHON,
        env=_VLLM_OPENAI_ENV,
    ),
    "deepseek-ocr-2": ModelConfig(
        script="https://huggingface.co/datasets/uv-scripts/ocr/raw/main/deepseek-ocr2-vllm.py",
        model_id="deepseek-ai/DeepSeek-OCR-2",
        size="3.4B",
        default_flavor="l4x1",
        # Free OCR (no <|grounding|>) so the judge sees clean text, matching
        # the deepseek-ocr (v1) entry.
        default_args=["--prompt-mode", "free"],
        # The script's vLLM nightly cu129 wheels need CUDA 13 runtime libs
        # (libnvrtc.so.13) that the default uv image lacks.
        image=_VLLM_OPENAI_IMAGE,
        python=_VLLM_OPENAI_PYTHON,
        env=_VLLM_OPENAI_ENV,
    ),
    "unlimited-ocr": ModelConfig(
        script="https://huggingface.co/datasets/uv-scripts/ocr/raw/main/unlimited-ocr-vllm.py",
        model_id="baidu/Unlimited-OCR",
        size="3.3B",
        default_flavor="l4x1",
        # --strip-grounding: judge sees clean text, not <|det|> layout boxes.
        default_args=["--strip-grounding"],
        # Baidu's dedicated vLLM image (vllm + torch from the image, per the
        # script's own docstring); use the :unlimited-ocr-cu129 tag on Hopper.
        image="vllm/vllm-openai:unlimited-ocr",
        python=_VLLM_OPENAI_PYTHON,
        env=_VLLM_OPENAI_ENV,
    ),
    "olmocr-2": ModelConfig(
        script="https://huggingface.co/datasets/uv-scripts/ocr/raw/main/olmocr2-vllm.py",
        model_id="allenai/olmOCR-2-7B-1025-FP8",
        size="8.3B",
        default_flavor="l4x1",
    ),
    # Classical (non-VLM) baselines
    "tesseract": ModelConfig(
        script="https://huggingface.co/datasets/uv-scripts/ocr/raw/main/tesseract-ocr.py",
        model_id="tesseract-5",
        size="n/a",
        default_flavor="cpu-upgrade",
    ),
    "pp-ocrv6": ModelConfig(
        script="https://huggingface.co/datasets/uv-scripts/ocr/raw/main/pp-ocrv6.py",
        model_id="PaddlePaddle/PP-OCRv6_medium",
        size="34.5M",
        default_flavor="t4-small",
        default_args=["--model-tier", "medium"],
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


# PEP 723 inline-metadata block: the specification's reference regex. The body
# is greedy, so a block ends at the last ``# ///`` line of its comment run, which
# lets a TOML multi-line string contain a ``# ///`` line.
_PEP723_BLOCK = re.compile(
    r"(?m)^# /// (?P<type>[a-zA-Z0-9-]+)$\s(?P<content>(^#(| .*)$\s)+)^# ///$"
)


def parse_hf_jobs_header(script_text: str) -> dict[str, Any]:
    """Return a UV script's ``[tool.hf-jobs]`` table, or ``{}`` when it has none.

    The table sits in the script's PEP 723 ``script`` block. ``hf jobs uv run``
    reads it for launch settings such as the Docker image; ``HfApi.run_uv_job``
    does not, so :func:`launch_settings` applies it for ocr-bench. Raises
    ``ValueError`` for more than one ``script`` block, as PEP 723 requires.
    """
    text = script_text.replace("\r\n", "\n").replace("\r", "\n")
    blocks = [m for m in _PEP723_BLOCK.finditer(text) if m.group("type") == "script"]
    if len(blocks) > 1:
        raise ValueError("multiple PEP 723 `script` blocks")
    if not blocks:
        return {}
    lines = blocks[0].group("content").splitlines(keepends=True)
    toml_text = "".join(line[2:] if line.startswith("# ") else line[1:] for line in lines)
    table = tomllib.loads(toml_text).get("tool", {}).get("hf-jobs", {})
    return table if isinstance(table, dict) else {}


def fetch_hf_jobs_header(script_url: str) -> dict[str, Any]:
    """Download a recipe and return its ``[tool.hf-jobs]`` table (``{}`` if absent).

    Raises ``RuntimeError`` when the script cannot be fetched or parsed, so a
    launch stops before any job starts rather than running on the wrong image.
    """
    try:
        response = get_session().get(script_url, follow_redirects=True, timeout=30)
        response.raise_for_status()
        return parse_hf_jobs_header(response.text)
    except Exception as exc:
        raise RuntimeError(
            f"Could not read the [tool.hf-jobs] header of {script_url}: {exc}"
        ) from exc


def launch_settings(
    config: ModelConfig,
    *,
    flavor_override: str | None = None,
    header: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the ``run_uv_job`` keyword arguments that pick hardware and runtime.

    - ``flavor``: ``flavor_override``, else the registry's ``default_flavor``.
    - When the registry sets ``image``, its ``image``/``python``/``env`` are
      used as a set and the script header is ignored (a deliberate override).
    - Otherwise each runtime key comes from the script header, as with
      ``hf jobs uv run``: ``image`` and ``python`` from the header unless the
      registry sets ``python``; ``env`` is the header's merged with the
      registry's, the registry winning. Recipes that moved vLLM out of their
      dependencies into a header image need this to start at all.
    """
    settings: dict[str, Any] = {"flavor": flavor_override or config.default_flavor}
    if config.image:
        settings["image"] = config.image
        if config.python:
            settings["python"] = config.python
        if config.env:
            settings["env"] = dict(config.env)
        return settings
    header = header or {}
    if header.get("image"):
        settings["image"] = header["image"]
    python = config.python or header.get("python")
    if python:
        settings["python"] = python
    env = {**dict(header.get("env") or {}), **(config.env or {})}
    if env:
        settings["env"] = env
    return settings


def build_script_args(
    input_dataset: str,
    output_repo: str,
    config_name: str,
    *,
    split: str = "train",
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
        "--split",
        split,
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

    # Resolve every model's launch settings before starting any job, so an
    # unreadable script header stops the whole run instead of leaving it partial.
    settings_by_slug: dict[str, dict[str, Any]] = {}
    for slug in selected:
        config = MODEL_REGISTRY[slug]
        header = {} if config.image else fetch_hf_jobs_header(config.script)
        settings_by_slug[slug] = launch_settings(
            config, flavor_override=flavor_override, header=header
        )

    jobs: list[JobRun] = []
    for slug in selected:
        config = MODEL_REGISTRY[slug]
        settings = settings_by_slug[slug]
        script_args = build_script_args(
            input_dataset,
            output_repo,
            slug,
            split=split,
            max_samples=max_samples,
            shuffle=shuffle,
            seed=seed,
            extra_args=config.default_args or None,
        )

        logger.info(
            "launching_job",
            model=slug,
            flavor=settings["flavor"],
            script=config.script,
            image=settings.get("image"),
        )
        job = api.run_uv_job(
            script=config.script,
            script_args=script_args,
            secrets={"HF_TOKEN": token},
            timeout=timeout,
            **settings,
        )
        jobs.append(JobRun(model_slug=slug, job_id=job.id, job_url=job.url))
        logger.info("job_launched", model=slug, job_id=job.id, url=job.url)

    return jobs


_TERMINAL_STAGES = frozenset({"COMPLETED", "ERROR", "CANCELED", "DELETED"})


def failed_jobs(jobs: list[JobRun]) -> list[JobRun]:
    """Return the jobs that did not finish in the COMPLETED state.

    ``poll_jobs`` records the terminal stage lowercased ("completed", "error",
    "canceled", "deleted"); anything other than "completed" is a failure. A
    still-"running" job (one that was never polled) also counts as not
    completed.
    """
    return [j for j in jobs if j.status != "completed"]


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
