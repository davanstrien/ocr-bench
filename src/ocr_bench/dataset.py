"""Dataset loading — flat, config-per-model, PR-based. OCR column discovery."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Literal

import structlog
from datasets import Dataset, get_dataset_config_names, load_dataset
from huggingface_hub import HfApi

logger = structlog.get_logger()


class DatasetError(Exception):
    """Raised when dataset loading or column discovery fails."""


# ---------------------------------------------------------------------------
# Cross-config row alignment (issue #5)
# ---------------------------------------------------------------------------
#
# ``load_config_dataset`` merges one config per model by *positional* index —
# row i of config A is assumed to be the same source page as row i of config B.
# That holds only when every run used the same source dataset, ``--seed`` and
# ``--max-samples``. These passthrough columns, carried unchanged through the
# OCR scripts, let us *verify* the assumption instead of trusting it: if two
# configs disagree on a shared key, their rows are not aligned and merging them
# would score model A's page against model B's transcription of a different page.
ALIGNMENT_KEYS: tuple[str, ...] = ("b_number", "page_index", "source_row", "id")

AlignmentStatus = Literal["ok", "misaligned", "partial", "unverified", "n/a"]


@dataclass
class AlignmentResult:
    """Outcome of checking row alignment across configs.

    ``status``:
      - ``ok``: every non-reference config shared a passthrough column with the
        reference and all matched.
      - ``misaligned``: a shared column disagrees — ``config``/``column``/
        ``index`` locate the first mismatch against ``reference_config``.
      - ``partial``: some configs were verified against the reference but others
        shared no identifying passthrough keys, so their alignment is unchecked. The
        unchecked configs are listed in ``unverified_configs``. A single passing
        config must NOT let the whole set read as verified.
      - ``unverified``: no config shared identifying passthrough keys with the
        reference, so alignment is positional-only and could not be checked.
      - ``n/a``: fewer than two configs (nothing to compare).
    """

    status: AlignmentStatus
    reference_config: str | None = None
    config: str | None = None
    column: str | None = None
    index: int | None = None
    shared_keys: list[str] = field(default_factory=list)
    verified_configs: list[str] = field(default_factory=list)
    unverified_configs: list[str] = field(default_factory=list)

    def config_status(self, config_name: str) -> str:
        """Per-config alignment status, for the audit's per-config display."""
        if config_name == self.reference_config:
            return "reference"
        if self.status == "misaligned" and config_name == self.config:
            return "misaligned"
        if config_name in self.unverified_configs:
            return "unverified"
        if config_name in self.verified_configs:
            return "ok"
        return "n/a"

    def describe(self) -> str:
        """One-line human summary for CLI/report output."""
        if self.status == "ok":
            return f"ok (matched on {', '.join(self.shared_keys)})"
        if self.status == "partial":
            return (
                f"PARTIAL — verified {', '.join(self.verified_configs)} on "
                f"{', '.join(self.shared_keys)}; unverified (no identifying keys): "
                f"{', '.join(self.unverified_configs)}"
            )
        if self.status == "misaligned":
            return (
                f"MISALIGNED — '{self.config}' vs '{self.reference_config}' "
                f"at row {self.index} (column '{self.column}')"
            )
        if self.status == "unverified":
            return "unverified (no identifying shared passthrough keys — positional only)"
        return "n/a (single config)"


def shared_alignment_keys(ds_a: Dataset, ds_b: Dataset) -> list[str]:
    """Passthrough columns present in *both* datasets, in canonical order."""
    return [k for k in ALIGNMENT_KEYS if k in ds_a.column_names and k in ds_b.column_names]


def _alignment_values_equal(a: object, b: object) -> bool:
    """Equality for passthrough values, treating two NaNs as the same missing value."""
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b


def _first_value_mismatch(ref: list, cur: list) -> int | None:
    """First index where two column value lists differ, or None if identical.

    A length difference counts as a mismatch at the first excess index.
    """
    n = min(len(ref), len(cur))
    for i in range(n):
        if not _alignment_values_equal(ref[i], cur[i]):
            return i
    if len(ref) != len(cur):
        return n
    return None


def _alignment_value_missing(value: object) -> bool:
    """Whether a passthrough value cannot contribute to row identity."""
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return isinstance(value, str) and not value.strip()


def _alignment_value_key(value: object) -> object:
    """Convert a possibly nested Arrow value into a hashable identity key."""
    try:
        hash(value)
    except TypeError:
        return json.dumps(value, sort_keys=True, default=str)
    return value


def alignment_keys_identify_rows(ds: Dataset, keys: list[str]) -> bool:
    """True when the shared key tuple uniquely identifies every dataset row.

    Equal constant/null columns cannot prove positional alignment: swapping two
    pages leaves those columns unchanged. Verification therefore requires every
    row to have at least one non-missing key value and the combined key tuple to
    be unique across the config.
    """
    if not keys:
        return False
    columns = [ds[key] for key in keys]
    identities: list[tuple[object, ...]] = []
    for values in zip(*columns):
        if all(_alignment_value_missing(value) for value in values):
            return False
        identities.append(tuple(_alignment_value_key(value) for value in values))
    return bool(identities) and len(identities) == len(set(identities))


def find_alignment_mismatch(
    ref_ds: Dataset, cur_ds: Dataset, keys: list[str]
) -> tuple[str, int] | None:
    """Return (column, row_index) of the first alignment mismatch, or None.

    Compares ``cur_ds`` against ``ref_ds`` on each shared passthrough ``key``
    using column access (no image decode).
    """
    for key in keys:
        idx = _first_value_mismatch(ref_ds[key], cur_ds[key])
        if idx is not None:
            return key, idx
    return None


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


@dataclass
class LoadedConfig:
    """A single loaded config, ready for merging or auditing."""

    config: str
    model_id: str
    ds: Dataset
    text_col: str | None


def _load_configs(
    repo_id: str,
    config_names: list[str],
    split: str,
    pr_revisions: dict[str, str],
) -> list[LoadedConfig]:
    """Load each config separately (column access only — no image decode).

    Shared by the merge path (``load_config_dataset``) and the read-only audit,
    so both see identical text-column and model-id resolution.
    """
    loaded: list[LoadedConfig] = []
    for config in config_names:
        revision = pr_revisions.get(config)
        kwargs: dict = {"path": repo_id, "name": config, "split": split}
        if revision:
            kwargs["revision"] = revision
        ds = load_dataset(**kwargs)
        loaded.append(
            LoadedConfig(
                config=config,
                model_id=_extract_model_id(ds, config),
                ds=ds,
                text_col=_find_text_column(ds),
            )
        )
    return loaded


def check_config_alignment(loaded: list[LoadedConfig]) -> AlignmentResult:
    """Verify row-for-row alignment across configs via shared passthrough keys.

    The first config is the reference. Returns an :class:`AlignmentResult`; this
    function never raises, so it is reusable by the read-only audit. Callers that
    must fail hard (the merge path) inspect ``status``.
    """
    if len(loaded) < 2:
        ref_config = loaded[0].config if loaded else None
        return AlignmentResult(status="n/a", reference_config=ref_config)

    ref = loaded[0]
    used: set[str] = set()
    verified: list[str] = []
    unverified: list[str] = []
    for lc in loaded[1:]:
        keys = shared_alignment_keys(ref.ds, lc.ds)
        if not keys:
            # No shared passthrough column with the reference — this config's
            # alignment can't be checked. It must NOT be silently folded into an
            # overall "ok" just because a sibling config passed.
            unverified.append(lc.config)
            continue
        hit = find_alignment_mismatch(ref.ds, lc.ds, keys)
        if hit is not None:
            column, index = hit
            return AlignmentResult(
                status="misaligned",
                reference_config=ref.config,
                config=lc.config,
                column=column,
                index=index,
                shared_keys=keys,
                verified_configs=verified,
                unverified_configs=unverified,
            )
        if not (
            alignment_keys_identify_rows(ref.ds, keys)
            and alignment_keys_identify_rows(lc.ds, keys)
        ):
            # Matching constant/null/repeated values do not identify rows, so a
            # permutation can pass equality while pairing different pages.
            unverified.append(lc.config)
            continue
        used.update(keys)
        verified.append(lc.config)

    ordered = [k for k in ALIGNMENT_KEYS if k in used]
    if not verified:
        # Nothing could be checked against the reference at all.
        return AlignmentResult(
            status="unverified", reference_config=ref.config, unverified_configs=unverified
        )
    if unverified:
        # Some verified, some not — an honest "partial", never a blanket "ok".
        return AlignmentResult(
            status="partial",
            reference_config=ref.config,
            shared_keys=ordered,
            verified_configs=verified,
            unverified_configs=unverified,
        )
    return AlignmentResult(
        status="ok",
        reference_config=ref.config,
        shared_keys=ordered,
        verified_configs=verified,
    )


def load_config_dataset(
    repo_id: str,
    config_names: list[str],
    split: str = "train",
    pr_revisions: dict[str, str] | None = None,
) -> tuple[Dataset, dict[str, str]]:
    """Load multiple configs from a Hub dataset and merge into one.

    Each config becomes a column whose name is the config name and whose value
    is the OCR text (from the first column matching heuristics, or ``markdown``).

    Before merging, row alignment is verified across configs on shared
    passthrough columns (see :data:`ALIGNMENT_KEYS`). A mismatch raises
    :class:`DatasetError`; when no passthrough columns are shared, a warning
    notes that alignment is positional-only.

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
    loaded = _load_configs(repo_id, config_names, split, pr_revisions)

    usable: list[LoadedConfig] = []
    for lc in loaded:
        if lc.text_col is None:
            logger.warning("no_text_column_in_config", config=lc.config)
            continue
        usable.append(lc)

    if not usable:
        raise DatasetError("No configs loaded successfully")

    # Every config must cover the same rows. A row-count mismatch is the silent-
    # corruption path issue #5 warns about: the positional merge would pair one
    # model's row i against another model's row i even though they are different
    # pages. Fail loudly instead of truncating the longer config.
    ref_n = len(usable[0].ds)
    for lc in usable[1:]:
        if len(lc.ds) != ref_n:
            raise DatasetError(
                f"Row-count mismatch in {repo_id}: config '{lc.config}' has "
                f"{len(lc.ds)} rows but reference config '{usable[0].config}' has "
                f"{ref_n}. Configs merge by position, so differing lengths misalign "
                f"the pages — re-run the OCR jobs at matching --max-samples (and the "
                f"same source split) so every config covers the same rows."
            )

    # Verify alignment BEFORE merging — a misalignment means we'd score one
    # model's page against another model's transcription of a different page.
    alignment = check_config_alignment(usable)
    if alignment.status == "misaligned":
        raise DatasetError(
            f"Row alignment mismatch in {repo_id}: config '{alignment.config}' "
            f"disagrees with reference config '{alignment.reference_config}' at row "
            f"{alignment.index} (passthrough column '{alignment.column}'). The configs "
            f"were not produced from the same source rows in the same order — re-run "
            f"the OCR jobs with matching --seed/--max-samples, or re-derive the configs."
        )
    if alignment.status in ("unverified", "partial"):
        # Not fatal — positional merge still proceeds — but the caller should
        # know which configs could not be alignment-checked.
        logger.warning(
            "alignment_unverified",
            repo_id=repo_id,
            status=alignment.status,
            reference=alignment.reference_config,
            unverified_configs=alignment.unverified_configs,
            note="these configs share no identifying passthrough keys with the reference; "
            "their positional alignment is unverified",
        )

    unified: Dataset | None = None
    ocr_columns: dict[str, str] = {}

    for lc in usable:
        config, text_col = lc.config, lc.text_col
        if text_col is None:  # filtered into `usable` above; narrows the type
            continue
        ocr_columns[config] = lc.model_id

        # Build unified dataset using Arrow-level ops (no per-row image decode).
        # Row counts are already verified equal above, so no truncation is needed.
        text_values = lc.ds[text_col]  # column access — no image decoding
        if unified is None:
            # First config: keep all columns except text_col, add text as config name
            drop = [text_col] if text_col != config else []
            unified = lc.ds.remove_columns(drop) if drop else lc.ds
            if config != text_col:
                unified = unified.add_column(config, text_values)
            # Also rename text_col to config if they differ and text_col was kept
        else:
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
    duplicates = sorted(mid for mid, n in model_counts.items() if n > 1)
    if duplicates:
        # Capture the colliding config → model_id mapping before relabeling so
        # the warning is actionable (which configs/model_ids collided, in which
        # repo) when running multiple datasets/config sweeps.
        collided = {
            config: model_id
            for config, model_id in ocr_columns.items()
            if model_counts[model_id] > 1
        }
        for config, model_id in list(ocr_columns.items()):
            if model_counts[model_id] > 1:
                short = model_id.split("/")[-1] if "/" in model_id else model_id
                ocr_columns[config] = f"{short} ({config})"
        logger.warning(
            "duplicate_model_ids",
            repo_id=repo_id,
            model_ids=duplicates,
            collided_configs=collided,
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
