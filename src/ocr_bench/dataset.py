"""Dataset loading — flat, config-per-model, PR-based. OCR column discovery."""

from __future__ import annotations

import json

import structlog
from datasets import Dataset, get_dataset_config_names, load_dataset
from huggingface_hub import HfApi

logger = structlog.get_logger()


class DatasetError(Exception):
    """Raised when dataset loading or column discovery fails."""


# ---------------------------------------------------------------------------
# OCR column discovery
# ---------------------------------------------------------------------------


def discover_ocr_columns(dataset: Dataset) -> dict[str, str]:
    """Discover OCR output columns and their model names from a dataset.

    Strategy:
      1. Parse ``inference_info`` JSON from the first row (list or single entry).
      2. Fallback: heuristic column-name matching (``markdown``, ``ocr``, ``text``).
      3. Disambiguate duplicate model names by appending the column name.

    Returns:
        Mapping of ``column_name → model_name``.

    Raises:
        DatasetError: If no OCR columns can be found.
    """
    columns: dict[str, str] = {}

    try:
        if "inference_info" not in dataset.column_names:
            raise KeyError("no inference_info column")
        info_raw = dataset["inference_info"][0]  # column access avoids image decode
        if info_raw:
            info = json.loads(info_raw)
            if not isinstance(info, list):
                info = [info]
            for entry in info:
                col = entry.get("column_name", "")
                model = entry.get("model_id", entry.get("model_name", "unknown"))
                if col and col in dataset.column_names:
                    columns[col] = model
    except (json.JSONDecodeError, TypeError, KeyError) as exc:
        logger.warning("could_not_parse_inference_info", error=str(exc))

    # Fallback: heuristic
    if not columns:
        for col in dataset.column_names:
            lower = col.lower()
            if "markdown" in lower or "ocr" in lower or col == "text":
                columns[col] = col

    if not columns:
        raise DatasetError(f"No OCR columns found. Available columns: {dataset.column_names}")

    # Disambiguate duplicates
    model_counts: dict[str, int] = {}
    for model in columns.values():
        model_counts[model] = model_counts.get(model, 0) + 1

    disambiguated: dict[str, str] = {}
    for col, model in columns.items():
        if model_counts[model] > 1:
            short = model.split("/")[-1] if "/" in model else model
            disambiguated[col] = f"{short} ({col})"
        else:
            disambiguated[col] = model

    return disambiguated


# ---------------------------------------------------------------------------
# PR-based config discovery
# ---------------------------------------------------------------------------


def discover_pr_configs(
    repo_id: str,
    merge: bool = False,
    api: HfApi | None = None,
) -> tuple[list[str], dict[str, str]]:
    """Discover dataset configs from open PRs on a Hub dataset repo.

    PR titles must end with ``[config_name]`` to be detected.

    Args:
        repo_id: HF dataset repo id.
        merge: If True, merge each discovered PR before loading.
        api: Optional pre-configured HfApi instance.

    Returns:
        Tuple of (config_names, {config_name: pr_revision}).
    """
    if api is None:
        api = HfApi()

    config_names: list[str] = []
    revisions: dict[str, str] = {}

    discussions = api.get_repo_discussions(repo_id, repo_type="dataset")
    for disc in discussions:
        if not disc.is_pull_request or disc.status != "open":
            continue
        title = disc.title
        if "[" in title and title.endswith("]"):
            config = title[title.rindex("[") + 1 : -1].strip()
            if config:
                if merge:
                    api.merge_pull_request(repo_id, disc.num, repo_type="dataset")
                    logger.info("merged_pr", pr=disc.num, config=config)
                else:
                    revisions[config] = f"refs/pr/{disc.num}"
                config_names.append(config)

    return config_names, revisions


def discover_configs(repo_id: str) -> list[str]:
    """List non-default configs from the main branch of a Hub dataset.

    Returns:
        Config names excluding "default", or empty list if none found.
    """
    try:
        configs = get_dataset_config_names(repo_id)
    except Exception as exc:
        logger.info("no_configs_on_main", repo=repo_id, reason=str(exc))
        return []
    return [c for c in configs if c != "default"]


# ---------------------------------------------------------------------------
# Config-per-model loading
# ---------------------------------------------------------------------------


def load_config_dataset(
    repo_id: str,
    config_names: list[str],
    split: str = "train",
    pr_revisions: dict[str, str] | None = None,
) -> tuple[Dataset, dict[str, str]]:
    """Load multiple configs from a Hub dataset and merge into one.

    Each config becomes a column whose name is the config name and whose value
    is the OCR text (from the first column matching heuristics, or ``markdown``).

    Args:
        repo_id: HF dataset repo id.
        config_names: List of config names to load.
        split: Dataset split to load.
        pr_revisions: Optional mapping of config_name → revision for PR-based loading.

    Returns:
        Tuple of (unified Dataset, {column_name: model_id}).
    """
    if not config_names:
        raise DatasetError("No config names provided")

    pr_revisions = pr_revisions or {}
    unified: Dataset | None = None
    ocr_columns: dict[str, str] = {}

    for config in config_names:
        revision = pr_revisions.get(config)
        kwargs: dict = {"path": repo_id, "name": config, "split": split}
        if revision:
            kwargs["revision"] = revision

        ds = load_dataset(**kwargs)

        # Find the OCR text column in this config
        text_col = _find_text_column(ds)
        if text_col is None:
            logger.warning("no_text_column_in_config", config=config)
            continue

        # Extract model_id from inference_info if available
        model_id = _extract_model_id(ds, config)
        ocr_columns[config] = model_id

        # Build unified dataset using Arrow-level ops (no per-row image decode)
        text_values = ds[text_col]  # column access — no image decoding
        if unified is None:
            # First config: keep all columns except text_col, add text as config name
            drop = [text_col] if text_col != config else []
            unified = ds.remove_columns(drop) if drop else ds
            if config != text_col:
                unified = unified.add_column(config, text_values)
            # Also rename text_col to config if they differ and text_col was kept
        else:
            if len(ds) != len(unified):
                logger.warning(
                    "config_length_mismatch",
                    config=config,
                    expected=len(unified),
                    got=len(ds),
                )
                text_values = text_values[: len(unified)]
            unified = unified.add_column(config, text_values)

    if unified is None:
        raise DatasetError("No configs loaded successfully")

    # Disambiguate configs that resolve to the same model_id (mirrors the flat
    # discover_ocr_columns path). Downstream the ELO keys models by these values,
    # so two configs of the same model run with different settings (e.g.
    # `nuextract3` vs `nuextract3-rep`) would silently collapse into one
    # leaderboard row. On collision, label by config name; keep the bare
    # model_id when unique.
    model_counts: dict[str, int] = {}
    for model_id in ocr_columns.values():
        model_counts[model_id] = model_counts.get(model_id, 0) + 1
    if any(n > 1 for n in model_counts.values()):
        for config, model_id in list(ocr_columns.items()):
            if model_counts[model_id] > 1:
                short = model_id.split("/")[-1] if "/" in model_id else model_id
                ocr_columns[config] = f"{short} ({config})"
        logger.warning(
            "duplicate_model_ids",
            note="configs sharing a model_id were labelled by config name to keep them distinct",
        )

    return unified, ocr_columns


def _extract_model_id(ds: Dataset, config: str) -> str:
    """Extract model_id from inference_info in first row, falling back to config name.

    Takes the *last* entry in the inference_info list, since OCR scripts append
    new entries — the last one is the model that actually produced this config.
    """
    if "inference_info" not in ds.column_names:
        return config
    try:
        info_raw = ds["inference_info"][0]  # column access avoids image decode
        if info_raw:
            info = json.loads(info_raw)
            if isinstance(info, list):
                info = info[-1]
            return info.get("model_id", info.get("model_name", config))
    except (json.JSONDecodeError, TypeError, KeyError, IndexError):
        pass
    return config


def _find_text_column(ds: Dataset) -> str | None:
    """Find the likely OCR text column in a dataset.

    Priority:
      1. ``inference_info[0]["column_name"]`` if present and exists in dataset.
      2. First column matching ``markdown`` (case-insensitive).
      3. First column matching ``ocr`` (case-insensitive).
      4. Column named exactly ``text``.
    """
    # Try inference_info first (column access avoids image decoding)
    if "inference_info" in ds.column_names:
        try:
            info_raw = ds["inference_info"][0]
            if info_raw:
                info = json.loads(info_raw)
                if isinstance(info, list):
                    info = info[0]
                col_name = info.get("column_name", "")
                if col_name and col_name in ds.column_names:
                    return col_name
        except (json.JSONDecodeError, TypeError, KeyError, IndexError):
            pass

    # Prioritized heuristic: markdown > ocr > text
    for pattern in ["markdown", "ocr"]:
        for col in ds.column_names:
            if pattern in col.lower():
                return col
    if "text" in ds.column_names:
        return "text"
    return None


# ---------------------------------------------------------------------------
# Flat dataset loading
# ---------------------------------------------------------------------------


def load_flat_dataset(
    repo_id: str,
    split: str = "train",
    columns: list[str] | None = None,
) -> tuple[Dataset, dict[str, str]]:
    """Load a flat dataset from Hub and discover OCR columns.

    Args:
        repo_id: HF dataset repo id.
        split: Dataset split.
        columns: If given, use these as OCR columns (maps col→col).

    Returns:
        Tuple of (Dataset, {column_name: model_name}).
    """
    ds = load_dataset(repo_id, split=split)

    if columns:
        # Validate columns exist
        for col in columns:
            if col not in ds.column_names:
                raise DatasetError(f"Column '{col}' not found. Available: {ds.column_names}")
        ocr_columns = {col: col for col in columns}
    else:
        ocr_columns = discover_ocr_columns(ds)

    return ds, ocr_columns
