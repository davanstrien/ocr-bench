"""Share a judged evaluation as a scores-only board in a separate repo.

A results repo written by ``judge`` holds page text: each comparison row has
both OCR outputs and the judge's reason. ``share`` reads that repo and writes
ratings only to a different dataset repo. It never changes the visibility of
an existing repo.

Three checks run on the final file contents, before any Hub call:

1. Structure. Each file is one that ``share`` writes. Each row has exactly the
   allowed columns, and each string value is one the board is meant to show: a
   model name, a slice label, a status or a setting. The README must be exactly
   the card rendered from those checked rows and settings.
2. Repo ids. No file names the results repo or the source dataset.
3. Page text. No aligned 32-character chunk of a compared output or a judge
   reason occurs in any file. A copied fragment of 63 or more characters holds
   such a chunk. A chunk with fewer than 4 distinct letters or digits is skipped:
   it is markup or layout (a markdown or HTML table rule, dot leaders, a run of
   spaces), and the board's own tables use the same markup.

If a check fails, :class:`PrivacyError` is raised and nothing is written.

Slice labels and the alias are published as they are. Choose a slice column of
labels that may be public (a document type, a decade).
"""

from __future__ import annotations

import datetime
import json
import re
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import structlog
from datasets import Image, load_dataset
from huggingface_hub import CommitOperationAdd, CommitOperationDelete, DatasetCard, HfApi
from huggingface_hub.errors import RepositoryNotFoundError
from huggingface_hub.utils import HFValidationError, validate_repo_id

from ocr_bench import __version__
from ocr_bench.dataset import (
    DatasetError,
    evaluation_input_fingerprints,
    fingerprint_identity_columns,
    load_config_dataset,
)
from ocr_bench.elo import (
    MIN_PAGES_TO_RESOLVE,
    ComparisonResult,
    Leaderboard,
    _unswap_winner,
    compute_elo,
)
from ocr_bench.judge import sample_indices
from ocr_bench.publish import (
    build_leaderboard_rows,
    load_existing_comparisons,
    load_existing_metadata,
)

logger = structlog.get_logger()

SHARE_TAG = "ocr-bench-share"
BOARD_COLUMNS = (
    "rank",
    "model",
    "params",
    "elo",
    "elo_low",
    "elo_high",
    "pages",
    "wins",
    "losses",
    "ties",
    "win_pct",
    "status",
    "failed_outputs",
)
# Error-placeholder counts are known per model for the whole sample only, so a
# slice board does not repeat them. Its Pages column shows the pages it rests on.
SLICE_BOARD_COLUMNS = (
    "slice",
    "rank",
    "model",
    "params",
    "elo",
    "elo_low",
    "elo_high",
    "pages",
    "wins",
    "losses",
    "ties",
    "win_pct",
)
SETTINGS_KEYS = (
    "alias",
    "judge_models",
    "judge_spec_hashes",
    "criteria",
    "prompt_hash",
    "judge_text_mode",
    "max_ocr_text_len",
    "judge_image_dim",
    "min_chars",
    "pages",
    "seed",
    "max_samples",
    "adaptive_strategy",
    "comparisons",
    "auto_tied",
    "slice_column",
    "slices",
    "slice_pages",
    "pages_not_in_a_slice_board",
    "method",
    "source_fingerprint",
    "judged_at",
    "shared_at",
    "ocr_bench_version",
    "repo",
    "sampled_pages",
)
SHARE_FILES = ("README.md", "overall.jsonl", "slices.jsonl", "board.json")
# The results repo folders judge writes, in the order it pushes them.
RESULT_FOLDERS = ("comparisons", "data", "metadata")
# Everything str.splitlines() splits on. A label or alias must hold none of them.
LINE_BREAKS = "\n\r\x0b\x0c\x1c\x1d\x1e\x85\u2028\u2029"
# Markdown that could make a label render as a link, an image, HTML or code.
MARKDOWN_SPECIAL = "\\`*_[]<>|"
STATUSES = ("ranked", "degraded", "failed")
# A model name that is a Hub repo id, with an optional " (label)" after it,
# gets a link to the repo's model page.
REPO_ID = r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}/[A-Za-z0-9][A-Za-z0-9_.-]{0,95}"
HUB_MODEL_ID = re.compile(rf"(?P<repo>{REPO_ID})(?P<label> \([^()]+\))?")

# A slice board needs as many pages as a resolved ranking does. A smaller slice
# is left out, which also keeps a per-page column (a page id) from being published.
MIN_SLICE_PAGES = MIN_PAGES_TO_RESOLVE
MAX_SLICES = 50
MAX_LABEL_LENGTH = 64

CHUNK = 32
MIN_DISTINCT_ALNUM = 4

N_BOOTSTRAP = 1000
METHOD = "Bradley-Terry MLE, page-level bootstrap 95% CIs (1000 resamples, seed 42)"
# Ratings are rounded to integers, so a refit can differ by one at a .5 boundary.
ELO_TOLERANCE = 1


class PrivacyError(DatasetError):
    """A file that share would publish fails a privacy check. Nothing was written."""


class ShareError(DatasetError):
    """The results repo or the destination repo cannot be used for a share."""


@dataclass
class PrivateEvaluation:
    """What share reads from a results repo. It holds page text: never publish it."""

    leaderboard: list[dict]
    comparisons: list[ComparisonResult]
    metadata: dict


@dataclass
class SharePlan:
    """The files to publish, plus the private values they are checked against."""

    alias: str
    files: dict[str, bytes]
    overall_rows: list[dict]
    slice_rows: list[dict]
    settings: dict
    allowed_strings: frozenset[str]
    source_dataset: str
    private_ids: list[str] = field(repr=False, default_factory=list)
    private_texts: list[str] = field(repr=False, default_factory=list)


# --- reading -----------------------------------------------------------------


def load_private_evaluation(results_repo: str, api: HfApi | None = None) -> PrivateEvaluation:
    """Read the leaderboard, comparisons and latest metadata row of a results repo.

    All three are read at one revision, after checking that the last judge run
    finished publishing (see :func:`check_publish_order`).
    """
    api = api or HfApi()
    try:
        revision = api.dataset_info(results_repo).sha
    except Exception as exc:
        raise ShareError(f"Could not read {results_repo} ({type(exc).__name__}).") from exc
    check_publish_order(api, results_repo, revision)
    try:
        leaderboard = [
            dict(row) for row in load_dataset(results_repo, split="train", revision=revision)
        ]
    except Exception as exc:
        # The loader's message can quote file contents, so only its type is shown.
        raise ShareError(
            f"Could not read the leaderboard of {results_repo} ({type(exc).__name__}). "
            "Is it a results repo written by `ocr-bench judge`?"
        ) from exc
    comparisons = load_existing_comparisons(results_repo, revision=revision)
    if not comparisons:
        raise ShareError(f"{results_repo} has no comparisons to share.")
    metadata_rows = load_existing_metadata(results_repo, revision=revision)
    if not metadata_rows:
        raise ShareError(f"{results_repo} has no metadata rows.")
    return PrivateEvaluation(leaderboard, comparisons, metadata_rows[-1])


def check_publish_order(api: HfApi, results_repo: str, revision: str | None) -> None:
    """Refuse a results repo whose last judge run did not finish publishing.

    Judge pushes the comparisons, then the leaderboard, then the metadata row, in
    separate commits; a checkpoint pushes only the comparisons. Every run appends a
    metadata row, so after a finished run the metadata folder changed no earlier
    than the comparisons folder. If a run stopped before its metadata push, the
    latest row describes an earlier run, and its settings and page fingerprint
    would be published with newer comparisons.

    The leaderboard is not ordered here: an unchanged leaderboard is not uploaded
    again, so its folder can look older after a finished run. The refit check
    (:func:`check_board_matches`) ties it to the comparisons instead. The check
    cannot tell apart two judge runs on the same repo at the same time.
    """
    try:
        infos = api.get_paths_info(
            results_repo, list(RESULT_FOLDERS), expand=True, revision=revision, repo_type="dataset"
        )
        commits = [
            commit.commit_id
            for commit in api.list_repo_commits(
                results_repo, repo_type="dataset", revision=revision
            )
        ]
    except Exception as exc:
        raise ShareError(
            f"Could not read the commit history of {results_repo} ({type(exc).__name__})."
        ) from exc
    last: dict[str, str] = {}
    for info in infos:
        commit = getattr(info, "last_commit", None)
        if commit is not None:
            last[info.path] = commit.oid
    if "comparisons" not in last:
        raise ShareError(
            f"{results_repo} has no comparisons yet, so there is nothing to share. "
            "Run `ocr-bench judge` first."
        )
    missing = [folder for folder in RESULT_FOLDERS if folder not in last]
    if missing:
        raise ShareError(
            f"Could not find the last commit of the {', '.join(missing)} folder in "
            f"{results_repo}. Is it a results repo written by `ocr-bench judge`?"
        )
    newest_first = {commit: position for position, commit in enumerate(commits)}
    if any(last[folder] not in newest_first for folder in RESULT_FOLDERS):
        raise ShareError(f"Could not order the commits of {results_repo}.")
    if newest_first[last["metadata"]] > newest_first[last["comparisons"]]:
        raise ShareError(
            f"The last judge run on {results_repo} did not finish publishing: its "
            "comparisons are newer than its metadata. Run the same judge command "
            "again to finish it, then share."
        )


def reference_config(metadata: dict) -> str | None:
    """The source config whose rows judge numbered, or None for a flat dataset.

    Judge merges configs by position and keeps the rows of the first config that
    has an OCR text column. ``source_output_columns`` lists those configs.
    """
    configs = json.loads(metadata.get("source_configs") or "[]")
    if not configs:
        return None
    usable = json.loads(metadata.get("source_output_columns") or "{}")
    for config in configs:
        if config in usable:
            return config
    raise ShareError("The results metadata names no source config with OCR output.")


def load_slice_values(metadata: dict, slice_column: str) -> list:
    """Read a column of the judged source rows, after proving they are the judged pages.

    ``sample_idx`` in a comparison is a row position in the reference config.
    The page fingerprint that judge stored covers the row ids and image bytes of
    the judged sample, so recomputing it on the reference config shows that row
    i is still the page judge compared. This reads every column of one config,
    images included, or of every config when judge's merge has to be rebuilt
    (see :func:`_as_judged`).
    """
    expected = metadata.get("source_fingerprint") or ""
    if not expected:
        raise ShareError(
            "The results repo has no source fingerprint, so share cannot check that "
            "slice labels are on the judged pages. Judge again with this version."
        )
    source = metadata["source_dataset"]
    split = metadata.get("source_split") or "train"
    config = reference_config(metadata)
    revision = json.loads(metadata.get("source_revisions") or "{}").get(config, "main")
    try:
        if config is None:
            ds = load_dataset(source, split=split)
        else:
            ds = load_dataset(source, name=config, split=split, revision=revision)
    except Exception as exc:
        raise ShareError(
            f"Could not read the judged source config {config or 'default'} ({type(exc).__name__})."
        ) from exc
    if slice_column not in ds.column_names:
        raise ShareError(
            f"Column '{slice_column}' is not in the judged source config {config or 'default'}."
        )
    judged = sample_indices(len(ds), metadata.get("max_samples"), metadata.get("seed", 42))
    try:
        view = _as_judged(ds, metadata)
    except Exception as exc:
        raise ShareError(f"Could not rebuild the judged pages ({type(exc).__name__}).") from exc
    # Check the columns the fingerprint reads, after judge's merge: a config named
    # "image" replaces the page image with that config's OCR text there.
    identity = fingerprint_identity_columns(view.column_names)
    if not any(isinstance(view.features.get(column), Image) for column in identity):
        # Without image bytes the fingerprint covers only the row ids, and ids do
        # not show that a row is still the page that was judged.
        raise ShareError(
            f"The judged pages from {config or 'default'} have no image column that "
            "the page fingerprint reads, so share cannot check that slice labels are "
            "on the judged pages. Share without --slice-column."
        )
    try:
        found, _ = evaluation_input_fingerprints(view, {}, judged)
    except Exception as exc:
        # A fingerprint error can name a local image path, so only its type is shown.
        raise ShareError(
            f"Could not fingerprint the judged source pages ({type(exc).__name__})."
        ) from exc
    if found != expected:
        raise ShareError(
            "The source pages are not the pages judge compared: their fingerprint "
            "changed. Slice labels could be on the wrong pages. Judge again on the "
            "current source, or share without --slice-column."
        )
    return list(ds[slice_column])


def _as_judged(ds, metadata: dict):
    """The columns judge fingerprinted, from the reference config ``ds``.

    Judge fingerprints the merged dataset: the reference config without its OCR
    column, plus one column per usable config, named after the config. Usually
    the identity columns (ids and the image) come from the reference config, so
    reading it is enough. When a config name or an OCR column is itself an
    identity column, judge's merge is rebuilt from every config.
    """
    configs = json.loads(metadata.get("source_configs") or "[]")
    if not configs:
        return ds
    outputs = json.loads(metadata.get("source_output_columns") or "{}")
    config = reference_config(metadata)
    merged_names = [name for name in ds.column_names if name != outputs[config]]
    merged_names = list(dict.fromkeys([*merged_names, *(c for c in configs if c in outputs)]))
    identity = fingerprint_identity_columns(merged_names)
    if identity and all(col in ds.column_names and col not in outputs for col in identity):
        return ds.select_columns(identity)
    revisions = json.loads(metadata.get("source_revisions") or "{}")
    merged, _ = load_config_dataset(
        metadata["source_dataset"],
        configs,
        split=metadata.get("source_split") or "train",
        pr_revisions={c: r for c, r in revisions.items() if r.startswith("refs/pr/")},
    )
    return merged


# --- fitting -----------------------------------------------------------------


def split_models(leaderboard: list[dict]) -> tuple[list[str], list[str]]:
    """(models judge fitted, failed models). Judge fits the board without failed models."""
    fitted = [row["model"] for row in leaderboard if row.get("status") != "failed"]
    failed = [row["model"] for row in leaderboard if row.get("status") == "failed"]
    return fitted, failed


def in_grid(comparisons: list[ComparisonResult], models: Iterable[str]) -> list[ComparisonResult]:
    """Comparisons between two fitted models. Other rows are history only."""
    grid = set(models)
    return [c for c in comparisons if c.model_a in grid and c.model_b in grid]


def check_board_matches(board: Leaderboard, leaderboard: list[dict], results_repo: str) -> None:
    """Refuse to share when a refit of the comparisons differs from the published board.

    Wins, losses and ties must be equal and ratings must agree to within rounding.
    They differ when a later judge run stopped after a checkpoint, so the
    comparisons hold verdicts that the leaderboard does not include.
    """
    for row in leaderboard:
        if row.get("status") == "failed":
            continue
        model = row["model"]
        refit = board.elo.get(model)
        counts = (board.wins.get(model), board.losses.get(model), board.ties.get(model))
        if (
            refit is None
            or abs(round(refit) - row["elo"]) > ELO_TOLERANCE
            or counts != (row["wins"], row["losses"], row["ties"])
        ):
            raise ShareError(
                f"The leaderboard of {results_repo} does not match its comparisons "
                f"({model}). A judge run probably stopped after a checkpoint. Run judge "
                "again to completion, then share."
            )


def _format_size(total: int) -> str:
    """A parameter count in the registry's style: 0.85B, 3.3B, 34.5M."""
    if total >= 10_000_000_000:
        return f"{total / 1e9:.3g}B"
    if total >= 100_000_000:
        return f"{total / 1e9:.2g}B"
    return f"{total / 1e6:.3g}M"


def hub_model_sizes(models: Iterable[str], api: HfApi | None = None) -> dict[str, str]:
    """Sizes from the Hub's safetensors metadata, for model names that are repo ids.

    A name with a " (label)" after the id uses the id. Only public repos are read.
    A model that is not on the Hub, is private or gated, has no safetensors files,
    or cannot be read gets no size.
    """
    api = api or HfApi()
    sizes: dict[str, str] = {}
    for name in models:
        match = HUB_MODEL_ID.fullmatch(name)
        if not match:
            continue
        try:
            # No token: a private model's size is not published.
            info = api.model_info(match["repo"], expand=["safetensors"], token=False)
        except Exception as error:  # an optional column: never stop a share for it
            logger.info("share.size_not_found", model=name, error=type(error).__name__)
            continue
        if info.safetensors and info.safetensors.total:
            sizes[name] = _format_size(info.safetensors.total)
    return sizes


def _fill_sizes(rows: list[dict], sizes: dict[str, str]) -> None:
    """The Hub's count where there is one, so every size is measured the same way."""
    for row in rows:
        row["params"] = sizes.get(row["model"]) or row["params"]


def _overall_rows(board: Leaderboard, failed_models: list[str], failed_outputs: dict) -> list[dict]:
    rows = []
    rank = 0
    for row in build_leaderboard_rows(board, failed_models, failed_outputs):
        failed = row["status"] == "failed"
        if not failed:
            rank += 1
        rows.append(
            {
                "rank": None if failed else rank,
                "model": row["model"],
                "params": row["params"] or None,
                "elo": row["elo"],
                "elo_low": row.get("elo_low"),
                "elo_high": row.get("elo_high"),
                "pages": board.pages_by_model.get(row["model"], 0),
                "wins": row["wins"],
                "losses": row["losses"],
                "ties": row["ties"],
                "win_pct": row["win_pct"],
                "status": row["status"],
                "failed_outputs": row["failed_outputs"],
            }
        )
    return rows


def _slice_rows(board: Leaderboard, slice_name: str) -> list[dict]:
    return [
        {
            "slice": slice_name,
            "rank": rank,
            "model": row["model"],
            "params": row["params"] or None,
            "elo": row["elo"],
            "elo_low": row.get("elo_low"),
            "elo_high": row.get("elo_high"),
            "pages": board.pages_by_model.get(row["model"], 0),
            "wins": row["wins"],
            "losses": row["losses"],
            "ties": row["ties"],
            "win_pct": row["win_pct"],
        }
        for rank, row in enumerate(build_leaderboard_rows(board), start=1)
    ]


def slice_groups(values: list, sample_ids: Iterable[int]) -> tuple[dict[str, set[int]], int]:
    """Group judged pages by slice label. Returns (groups, unlabelled page count)."""
    groups: dict[str, set[int]] = {}
    unlabelled = 0
    for idx in sample_ids:
        value = values[idx]
        label = "" if value is None else str(value).strip()
        if not label:
            unlabelled += 1
            continue
        groups.setdefault(label, set()).add(idx)
    if len(groups) > MAX_SLICES:
        raise ShareError(
            f"The slice column has {len(groups)} different values on the judged pages "
            f"(at most {MAX_SLICES}). Is it a per-page column?"
        )
    if any(len(label) > MAX_LABEL_LENGTH or _has_line_break(label) for label in groups):
        raise ShareError(
            f"Slice labels must be {MAX_LABEL_LENGTH} characters or fewer, on one line. "
            "Use a column of short labels."
        )
    return groups, unlabelled


def rankable_models(
    comparisons: list[ComparisonResult], models: list[str]
) -> tuple[list[str], list[ComparisonResult]]:
    """Models a slice board can rank, and the comparisons between them.

    A model needs MIN_SLICE_PAGES pages in the slice. The models left must be
    rankable together (see :func:`rankable_together`). Returns no models when
    they are not.
    """
    kept = list(models)
    while True:
        # Dropping a model removes pages from the models it was compared with,
        # so the floor is applied again until nothing changes.
        kept_set = set(kept)
        between = [c for c in comparisons if c.model_a in kept_set and c.model_b in kept_set]
        pages: dict[str, set[int]] = {}
        for c in between:
            pages.setdefault(c.model_a, set()).add(c.sample_idx)
            pages.setdefault(c.model_b, set()).add(c.sample_idx)
        still = [m for m in kept if len(pages.get(m, ())) >= MIN_SLICE_PAGES]
        if still == kept:
            break
        kept = still
    if len(kept) < 2 or not rankable_together(kept, between):
        return [], []
    return kept, between


def rankable_together(models: list[str], comparisons: list[ComparisonResult]) -> bool:
    """Whether Bradley-Terry gives every model a finite rating from these comparisons.

    Draw an arrow from each model to every model it beat or tied. The fit has a
    finite rating for every model only when each model can reach every other
    along the arrows. It fails when two groups were never compared (targeted
    runs judge only some pairs), and when one group beat or lost to the other in
    every comparison: a model that won all its comparisons gets an unbounded
    rating and a zero-width interval.
    """
    beat: dict[str, set[str]] = {model: set() for model in models}
    beaten_by: dict[str, set[str]] = {model: set() for model in models}
    for c in comparisons:
        if c.model_a not in beat or c.model_b not in beat:
            continue
        winner = _unswap_winner(c.winner, c.swapped)
        if winner in ("A", "tie"):
            beat[c.model_a].add(c.model_b)
            beaten_by[c.model_b].add(c.model_a)
        if winner in ("B", "tie"):
            beat[c.model_b].add(c.model_a)
            beaten_by[c.model_a].add(c.model_b)
    return _reaches_all(models, beat) and _reaches_all(models, beaten_by)


def _reaches_all(models: list[str], arrows: dict[str, set[str]]) -> bool:
    reached, todo = {models[0]}, [models[0]]
    while todo:
        for other in arrows[todo.pop()] - reached:
            reached.add(other)
            todo.append(other)
    return reached == set(models)


def _has_line_break(text: str) -> bool:
    return any(char in LINE_BREAKS for char in text)


def check_judged_sample(values: list, metadata: dict, sample_ids: set[int]) -> None:
    """Refuse when a comparison is on a row outside the sample judge draws.

    The fingerprint proves the drawn sample is the judged pages, so this binds
    the comparisons to those pages.
    """
    if sample_ids and max(sample_ids) >= len(values):
        raise ShareError(
            f"The source has {len(values)} rows, but a comparison is on row "
            f"{max(sample_ids)}. The source changed after judging."
        )
    drawn = set(sample_indices(len(values), metadata.get("max_samples"), metadata.get("seed", 42)))
    if not sample_ids <= drawn:
        raise ShareError(
            "Some comparisons are on pages outside the judged sample. The comparisons "
            "and the metadata are not from the same judge run."
        )


# --- files -------------------------------------------------------------------


def _jsonl(rows: list[dict]) -> bytes:
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()


def _md(value: object) -> str:
    """A value as plain text in a markdown table cell."""
    text = "".join(" " if char in LINE_BREAKS else char for char in str(value))
    return "".join("\\" + char if char in MARKDOWN_SPECIAL else char for char in text)


def _model_cell(name: str) -> str:
    """The model name, linked to its Hub page when it is a Hub repo id."""
    match = HUB_MODEL_ID.fullmatch(name)
    if not match:
        return _md(name)
    repo = match["repo"]
    return f"[{_md(repo)}](https://huggingface.co/{repo}){_md(match['label'] or '')}"


def _table(rows: list[dict]) -> list[str]:
    lines = [
        "| Rank | Model | Size | Rating | 95% interval | Pages | Wins | Losses | Ties |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        status = row.get("status", "ranked")
        if status == "failed":
            lines.append(
                f"| — | {_model_cell(row['model'])} | {_md(row['params'] or '')} | FAILED "
                "| — | — | — | — | — |"
            )
            continue
        ci = f"{row['elo_low']}–{row['elo_high']}" if row.get("elo_low") is not None else "—"
        mark = " ⚠" if status == "degraded" else ""
        lines.append(
            f"| {row['rank']} | {_model_cell(row['model'])}{mark} | {_md(row['params'] or '')} "
            f"| {row['elo']} | {ci} | {row['pages']} | {row['wins']} | {row['losses']} "
            f"| {row['ties']} |"
        )
    return lines


def _label(value: object) -> str:
    """A slice label or column name as inline code, or escaped text if it holds a backtick."""
    text = "".join(" " if char in LINE_BREAKS else char for char in str(value))
    return _md(text) if "`" in text else f"`{text}`"


def build_card(settings: dict, overall_rows: list[dict], slice_rows: list[dict]) -> str:
    """The README of the share repo, rendered only from the checked rows and settings."""
    repo = settings["repo"]
    ranked = [row for row in overall_rows if row["status"] != "failed"]
    lines = [
        "---",
        "tags:",
        "  - ocr-bench",
        f"  - {SHARE_TAG}",
        "  - leaderboard",
        "configs:",
        "  - config_name: default",
        "    data_files: overall.jsonl",
    ]
    if slice_rows:
        lines += ["  - config_name: slices", "    data_files: slices.jsonl"]
    lines += [
        "---",
        "",
        f"# OCR leaderboard: {_md(settings['alias'])}",
        "",
        f"{len(ranked)} OCR models ranked on {settings['pages']} pages. For each page, a "
        "vision-language model compared two transcriptions at a time and picked the more "
        "faithful one, or called a tie. "
        "[ocr-bench](https://github.com/davanstrien/ocr-bench) ran the comparisons and "
        "fitted the ratings. The pages, the transcriptions and the judge's reasons stay "
        "private: this repo holds scores only.",
        "",
        "**How to read it.** A higher rating is better; 1500 is the average rating on each "
        "board. The 95% interval is where the rating stays when the pages are resampled: "
        "when two intervals overlap, this data does not clearly separate those models. "
        "Size is the parameter count the Hub reports for the model's weights, where it "
        "has one (all experts, for a mixture-of-experts model). "
        "Pages is the number of pages a model was compared on. Wins, losses and ties are "
        "raw counts that depend on how often a model was compared, so compare ratings, "
        "not wins.",
        "",
        f"## All pages ({settings['pages']})",
        "",
        *_table(overall_rows),
    ]
    degraded = [row for row in overall_rows if row["status"] == "degraded"]
    if degraded:
        counts = "; ".join(f"{_md(row['model'])}: {row['failed_outputs']}" for row in degraded)
        lines += [
            "",
            "⚠ marks a model that returned an error message (such as `[OCR ERROR]`) instead "
            "of text on some pages. Those pages were skipped for it, so its rating rests on "
            f"the pages in the Pages column. Error outputs in the source dataset: {counts}.",
        ]
    if any(row["status"] == "failed" for row in overall_rows):
        lines += ["", "FAILED: every output was an error message, so the model has no rating."]
    if slice_rows:
        column = _label(settings["slice_column"])
        lines += [
            "",
            f"## By {column}",
            "",
            f"One board for each value of the {column} column, fitted only on the pages with "
            f"that value. A model needs {MIN_SLICE_PAGES} of those pages to appear.",
        ]
        for name, pages in zip(settings["slices"], settings["slice_pages"], strict=True):
            rows = [row for row in slice_rows if row["slice"] == name]
            lines += ["", f"### {_label(name)} (ranked on {pages} pages)", "", *_table(rows)]
        if settings["pages_not_in_a_slice_board"]:
            lines += [
                "",
                f"{settings['pages_not_in_a_slice_board']} judged pages are in none of these "
                "boards because they have no label, their label has fewer than "
                f"{MIN_SLICE_PAGES} pages, or they were only compared with models left out of "
                "their board.",
            ]
    judges = ", ".join(_md(name) for name in settings["judge_models"]) or "not recorded"
    criteria = settings["criteria"]
    if criteria == "custom":
        judge_line = f"- **Judge**: {judges}, with a custom prompt"
    else:
        # A link, not a paraphrase: judge reasons echo the prompt's wording, and the
        # page-text check refuses a card that repeats them.
        prompts = (
            "https://github.com/davanstrien/ocr-bench/blob/"
            f"v{settings['ocr_bench_version']}/src/ocr_bench/judge.py"
        )
        judge_line = (
            f"- **Judge**: {judges}, with the `{_md(criteria)}` criteria ([the prompts]({prompts}))"
        )
    if settings["sampled_pages"]:
        sample = (
            f"{settings['sampled_pages']} pages sampled (seed {settings['seed']}); "
            f"{settings['pages']} had at least one comparison"
        )
    elif settings["max_samples"]:
        sample = (
            f"{settings['pages']} pages with at least one comparison, from a sample of at "
            f"most {settings['max_samples']} (seed {settings['seed']})"
        )
    else:
        sample = f"{settings['pages']} pages with at least one comparison"
    lines += [
        "",
        "## How this board was made",
        "",
        judge_line,
        f"- **Pages**: {sample}",
        f"- **Comparisons**: {settings['comparisons']:,}; {settings['auto_tied']} of them had "
        "identical text and were tied without a judge call",
        "- **Ratings**: a Bradley-Terry fit on an Elo-like scale; 95% intervals from "
        "resampling pages",
        f"- **Judged**: {str(settings['judged_at'])[:10] or 'not recorded'}",
        "",
        "The full settings (prompt hash, text and image caps, page-set fingerprint) are in "
        "`board.json`.",
        "",
        "## Use the data",
        "",
        "In Python, with `from datasets import load_dataset`: "
        f'`load_dataset("{repo}")` gives the overall board'
        + (
            f'; `load_dataset("{repo}", "slices")` gives the boards by label.'
            if slice_rows
            else "."
        ),
        "",
        "<details>",
        "<summary>Query with DuckDB</summary>",
        "",
        "Each row is one model on one board. `elo` is the rating, and `elo_low` and "
        "`elo_high` are its 95% interval.",
        "",
        "```sql",
        "-- a private board needs a Hub token first:",
        "-- CREATE SECRET hf (TYPE huggingface, PROVIDER credential_chain);",
        f"SELECT rank, model, elo, elo_low, elo_high, pages "
        f"FROM 'hf://datasets/{repo}/overall.jsonl' ORDER BY rank;",
    ]
    if slice_rows:
        lines += [
            "-- the top model for each label",
            f"SELECT slice, model, elo, elo_low, elo_high "
            f"FROM 'hf://datasets/{repo}/slices.jsonl' WHERE rank = 1 ORDER BY slice;",
        ]
    lines += [
        "```",
        "",
        "</details>",
        "",
        f"*Generated by [ocr-bench](https://github.com/davanstrien/ocr-bench) "
        f"{settings['ocr_bench_version']}*",
    ]
    return "\n".join(lines) + "\n"


def _settings_json(settings: dict) -> bytes:
    return (json.dumps(settings, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode()


def build_share_files(
    settings: dict, overall_rows: list[dict], slice_rows: list[dict]
) -> dict[str, bytes]:
    files = {
        "README.md": build_card(settings, overall_rows, slice_rows).encode(),
        "overall.jsonl": _jsonl(overall_rows),
        "board.json": _settings_json(settings),
    }
    if slice_rows:
        files["slices.jsonl"] = _jsonl(slice_rows)
    return files


# --- checks ------------------------------------------------------------------


def _strings(value: object) -> Iterator[str]:
    """Every string in a decoded JSON value: dict keys and string leaves."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def _check_value(path: str, key: str, value: object, allowed: frozenset[str]) -> None:
    if value is None or isinstance(value, bool | int | float):
        return
    if isinstance(value, str) and value in allowed:
        return
    if isinstance(value, list) and all(
        (isinstance(v, str) and v in allowed) or (isinstance(v, int) and not isinstance(v, bool))
        for v in value
    ):
        return
    raise PrivacyError(
        f"{path}: the value of '{key}' is not one the board is meant to show. Nothing was written."
    )


def _rows(files: dict[str, bytes], path: str, columns: tuple[str, ...], allowed) -> list[dict]:
    rows = []
    for line in files.get(path, b"").decode().split("\n"):
        if not line:
            continue
        row = json.loads(line)
        if tuple(row) != columns:
            raise PrivacyError(
                f"{path}: a row has columns {sorted(row)}, not the allowed columns. "
                "Nothing was written."
            )
        for key, value in row.items():
            _check_value(path, key, value, allowed)
        rows.append(row)
    return rows


def check_structure(files: dict[str, bytes], allowed: frozenset[str]) -> None:
    """Allow-list check: known files, exact columns, allowed values, and a README
    that is exactly the card rendered from them."""
    unknown = sorted(set(files) - set(SHARE_FILES))
    if unknown:
        raise PrivacyError(f"Unexpected files {unknown}. Nothing was written.")
    overall = _rows(files, "overall.jsonl", BOARD_COLUMNS, allowed)
    slices = _rows(files, "slices.jsonl", SLICE_BOARD_COLUMNS, allowed)
    settings = json.loads(files["board.json"].decode())
    if sorted(settings) != sorted(SETTINGS_KEYS):
        raise PrivacyError("board.json: unexpected keys. Nothing was written.")
    for key, value in settings.items():
        _check_value("board.json", key, value, allowed)
    if files["board.json"] != _settings_json(settings):
        raise PrivacyError("board.json is not in the form share writes. Nothing was written.")
    if files["README.md"] != build_card(settings, overall, slices).encode():
        raise PrivacyError(
            "README.md is not the card rendered from the checked boards. Nothing was written."
        )


def _spans(text: str, needle: str) -> Iterator[tuple[int, int]]:
    start = text.find(needle)
    while start != -1:
        yield start, start + len(needle)
        start = text.find(needle, start + 1)


def check_no_private_ids(
    files: dict[str, bytes], private_ids: Iterable[str], board_repo: str = ""
) -> None:
    """Refuse when any file names the results repo or the source dataset.

    A mention inside the board's own repo id is allowed: that id is the published
    repo name, and a name such as ``me/scans-board`` contains a source id such as
    ``me/scans``. Every other mention, also one that overlaps the board id, is refused.
    """
    ids = [repo_id.lower() for repo_id in private_ids if repo_id]
    own = re.escape(board_repo.lower())
    for path, data in files.items():
        for view in _file_views(path, data):
            text = view.lower()
            # Whole ids only: a board me/a is not a mention inside me/a-results.
            allowed = (
                [m.span() for m in re.finditer(rf"(?<![\w.-]){own}(?![\w.-])", text)]
                if board_repo
                else []
            )
            for repo_id in ids:
                for found in _spans(text, repo_id):
                    if not any(lo <= found[0] and found[1] <= hi for lo, hi in allowed):
                        raise PrivacyError(
                            f"{path} names the results repo or the source dataset. "
                            "Nothing was written."
                        )


def _chunks(text: str) -> Iterator[str]:
    for start in range(0, len(text) - CHUNK + 1, CHUNK):
        piece = text[start : start + CHUNK]
        if len({char for char in piece if char.isalnum()}) >= MIN_DISTINCT_ALNUM:
            yield piece


def _windows(text: str) -> set[str]:
    return {text[i : i + CHUNK] for i in range(len(text) - CHUNK + 1)}


def _file_views(path: str, data: bytes) -> list[str]:
    """The texts to search in a file: its raw text plus every decoded JSON string.

    JSON escapes newlines and quotes, so a copied fragment can be absent from
    the raw text and present in a decoded value.
    """
    text = data.decode("utf-8")
    views = [text]
    if path.endswith(".jsonl"):
        for line in text.split("\n"):
            if line:
                views.extend(_strings(json.loads(line)))
    elif path.endswith(".json"):
        views.extend(_strings(json.loads(text)))
    return views


def check_no_page_text(files: dict[str, bytes], private_texts: Iterable[str]) -> None:
    """Refuse when any file holds a fragment of a compared output or judge reason."""
    windows = {
        path: set().union(*(_windows(view) for view in _file_views(path, data)))
        for path, data in files.items()
    }
    for text in private_texts:
        for piece in _chunks(text):
            for path, found in windows.items():
                if piece in found:
                    raise PrivacyError(
                        f"{path} holds text from a compared page or a judge reason. "
                        "Nothing was written."
                    )


def private_texts(comparisons: Iterable[ComparisonResult]) -> list[str]:
    """Every OCR output and judge reason in the comparisons, without repeats."""
    texts: dict[str, None] = {}
    for comparison in comparisons:
        for text in (comparison.text_a, comparison.text_b, comparison.reason):
            if text:
                texts[text] = None
    return list(texts)


def check_plan(plan: SharePlan) -> None:
    """All privacy checks, on the exact bytes that would be written."""
    check_structure(plan.files, plan.allowed_strings)
    check_no_private_ids(plan.files, plan.private_ids, plan.settings["repo"])
    check_no_page_text(plan.files, plan.private_texts)


# --- plan and push -----------------------------------------------------------


def _check_dest(dest: str) -> str:
    # A bare name would be created under the token's namespace, then not found again
    # under the bare name, leaving an empty repo behind. The card shows the id, so it
    # must be a valid Hub repo id; validate_repo_id lets a trailing newline through.
    try:
        validate_repo_id(dest)
        # validate_repo_id's \w also matches non-ASCII letters, which the Hub refuses.
        valid = dest.count("/") == 1 and dest == dest.strip() and dest.isascii()
    except HFValidationError:
        valid = False
    if not valid:
        raise ShareError(
            f"Give the board repo as namespace/name (for example me/board), not {dest!r}."
        )
    return dest


def _check_alias(alias: str) -> str:
    alias = alias.strip()
    if not alias or len(alias) > MAX_LABEL_LENGTH or _has_line_break(alias):
        raise ShareError(f"--alias must be 1–{MAX_LABEL_LENGTH} characters on one line.")
    return alias


def _sampled_pages(metadata: dict) -> int | None:
    """The size of the judged sample, from the coverage judge records (0.0.6 and later)."""
    try:
        coverage = json.loads(metadata.get("model_coverage") or "{}")
    except json.JSONDecodeError:
        return None
    counts = [entry.get("sampled_pages") for entry in coverage.values() if isinstance(entry, dict)]
    counts = [count for count in counts if isinstance(count, int) and count > 0]
    return max(counts) if counts else None


def _public_criteria(criteria: str) -> str:
    """A --criteria-file run is recorded as ``custom:<file name>``; publish ``custom``."""
    return "custom" if criteria.startswith("custom:") else criteria


def plan_share(
    evaluation: PrivateEvaluation,
    *,
    results_repo: str,
    alias: str,
    slice_column: str | None = None,
    slice_values: Callable[[dict, str], list] | None = None,
    now: str | None = None,
    board_repo: str,
    model_sizes: Callable[[list[str]], dict[str, str]] | None = None,
) -> SharePlan:
    """Fit the boards, build the files and check them. Makes no Hub writes.

    ``board_repo`` is the repo the board is for. The card names it in its data
    snippets, and ``push_share`` writes the plan to that repo only.
    ``model_sizes`` gives each model's size (the CLI passes ``hub_model_sizes``).
    A model it gives no size for keeps the built-in registry's size, if any.
    """
    board_repo = _check_dest(board_repo)
    alias = _check_alias(alias)
    metadata = evaluation.metadata
    if board_repo.lower() in (
        results_repo.lower(),
        str(metadata.get("source_dataset", "")).lower(),
    ):
        raise ShareError(
            f"{board_repo} is the results repo or the source dataset. Share to a separate repo."
        )
    models, failed_models = split_models(evaluation.leaderboard)
    comparisons = in_grid(evaluation.comparisons, models)
    if not comparisons:
        raise ShareError(f"{results_repo} has no comparisons between its leaderboard models.")
    failed_outputs = json.loads(metadata.get("failed_outputs") or "{}")
    if not rankable_together(models, comparisons):
        raise ShareError(
            "The comparisons cannot rank every leaderboard model against every other: "
            "two groups of models were never compared, or one model won (or lost) "
            "every comparison. Bradley-Terry gives no finite rating then, so share "
            "publishes no board. Judge the missing pairs, or leave such a model out "
            "with judge --configs."
        )

    board = compute_elo(comparisons, models, n_bootstrap=N_BOOTSTRAP)
    check_board_matches(board, evaluation.leaderboard, results_repo)
    overall_rows = _overall_rows(board, failed_models, failed_outputs)
    sample_ids = {c.sample_idx for c in comparisons}

    slice_rows: list[dict] = []
    slice_names: list[str] = []
    slice_pages: list[int] = []
    not_in_a_slice = 0
    if slice_column:
        values = (slice_values or load_slice_values)(metadata, slice_column)
        check_judged_sample(values, metadata, sample_ids)
        groups, not_in_a_slice = slice_groups(values, sample_ids)
        for name in sorted(groups):
            pages = groups[name]
            if len(pages) < MIN_SLICE_PAGES:
                not_in_a_slice += len(pages)
                continue
            slice_models, slice_comparisons = rankable_models(
                [c for c in comparisons if c.sample_idx in pages], models
            )
            if not slice_models:
                not_in_a_slice += len(pages)
                continue
            slice_board = compute_elo(slice_comparisons, slice_models, n_bootstrap=N_BOOTSTRAP)
            slice_rows += _slice_rows(slice_board, name)
            slice_names.append(name)
            # Pages whose only comparisons involve a model left out of this board
            # are not on it.
            ranked_pages = {c.sample_idx for c in slice_comparisons}
            slice_pages.append(len(ranked_pages))
            not_in_a_slice += len(pages) - len(ranked_pages)

    if model_sizes:
        sizes = model_sizes(sorted({row["model"] for row in overall_rows}))
        _fill_sizes(overall_rows, sizes)
        _fill_sizes(slice_rows, sizes)

    judge_models = json.loads(metadata.get("judge_models") or "[]")
    judge_spec_hashes = json.loads(metadata.get("judge_spec_hashes") or "[]")
    settings = {
        "alias": alias,
        "judge_models": judge_models,
        "judge_spec_hashes": judge_spec_hashes,
        "criteria": _public_criteria(metadata.get("criteria") or "default"),
        "prompt_hash": metadata.get("prompt_hash") or "",
        "judge_text_mode": metadata.get("judge_text_mode") or "normalized",
        "max_ocr_text_len": metadata.get("max_ocr_text_len"),
        "judge_image_dim": metadata.get("judge_image_dim"),
        "min_chars": metadata.get("min_chars"),
        "pages": len(sample_ids),
        "seed": metadata.get("seed"),
        "max_samples": metadata.get("max_samples"),
        "adaptive_strategy": metadata.get("adaptive_strategy") or "balanced",
        "comparisons": len(comparisons),
        "auto_tied": sum(1 for c in comparisons if c.agreement == "auto"),
        "slice_column": slice_column,
        "slices": slice_names,
        "slice_pages": slice_pages,
        "pages_not_in_a_slice_board": not_in_a_slice if slice_column else 0,
        "method": METHOD,
        "source_fingerprint": metadata.get("source_fingerprint") or "",
        "judged_at": metadata.get("timestamp") or "",
        "shared_at": now or datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "ocr_bench_version": __version__,
        "repo": board_repo,
        "sampled_pages": _sampled_pages(metadata),
    }
    allowed = frozenset(
        {
            *models,
            *failed_models,
            *slice_names,
            *STATUSES,
            *(row["params"] for row in overall_rows if row["params"]),
            *judge_models,
            *judge_spec_hashes,
            *(v for v in settings.values() if isinstance(v, str)),
        }
    )
    plan = SharePlan(
        alias=alias,
        files=build_share_files(settings, overall_rows, slice_rows),
        overall_rows=overall_rows,
        slice_rows=slice_rows,
        settings=settings,
        allowed_strings=allowed,
        source_dataset=metadata.get("source_dataset", ""),
        private_ids=[results_repo, metadata.get("source_dataset", "")],
        private_texts=private_texts(evaluation.comparisons),
    )
    check_plan(plan)
    return plan


def write_local(plan: SharePlan, out_dir: Path) -> list[Path]:
    """Write the checked files to a local folder (``--dry-run``)."""
    check_plan(plan)
    out_dir.mkdir(parents=True, exist_ok=True)
    # A file from an earlier run (slices.jsonl after a run without slices) would
    # otherwise sit next to the new board.
    for name in SHARE_FILES:
        if name not in plan.files and (out_dir / name).is_file():
            (out_dir / name).unlink()
    written = []
    for path, data in plan.files.items():
        target = out_dir / path
        target.write_bytes(data)
        written.append(target)
    return written


@dataclass
class PushResult:
    repo_id: str
    created: bool
    private: bool


def _is_share_card(readme: str) -> bool:
    try:
        tags = DatasetCard(readme).data.to_dict().get("tags") or []
    except Exception:
        return False
    return isinstance(tags, list) and SHARE_TAG in tags


def _revision(api: HfApi, dest: str) -> str:
    revision = api.repo_info(dest, repo_type="dataset").sha
    if not revision:
        raise ShareError(f"Could not read the current revision of {dest}.")
    return revision


def _destination_files(
    api: HfApi, dest: str, alias: str, replace: bool
) -> tuple[str, list[str]] | None:
    """(revision, files) of an existing destination, or None when it is missing.

    Refuses a repo with files that share does not write, or whose card lacks the
    ``ocr-bench-share`` tag. Refuses a board of another alias unless ``replace``.
    """
    try:
        revision = _revision(api, dest)
    except RepositoryNotFoundError:
        return None
    files = api.list_repo_files(dest, repo_type="dataset", revision=revision)
    content = [path for path in files if path != ".gitattributes"]
    if not content:
        return revision, files
    foreign = sorted(set(content) - set(SHARE_FILES))
    readme = ""
    if "README.md" in content:
        readme = Path(
            api.hf_hub_download(dest, "README.md", repo_type="dataset", revision=revision)
        ).read_text()
    if foreign or not _is_share_card(readme):
        raise ShareError(
            f"{dest} has files that `ocr-bench share` did not write. Choose a new or empty repo."
        )
    if "board.json" in content and not replace:
        path = api.hf_hub_download(dest, "board.json", repo_type="dataset", revision=revision)
        current = json.loads(Path(path).read_text()).get("alias")
        if current != alias:
            raise ShareError(
                f"{dest} holds the board of '{current}'. Pass --replace to replace it "
                f"with '{alias}', or choose another repo."
            )
    return revision, files


def push_share(
    plan: SharePlan,
    dest: str,
    *,
    results_repo: str,
    public: bool = False,
    replace: bool = False,
    api: HfApi | None = None,
) -> PushResult:
    """Check the files again, then write them to ``dest`` in one commit.

    A missing ``dest`` is created private, or public with ``public``. An existing
    one keeps its visibility. The commit names the revision that was checked as its
    parent, so it fails if another writer changed the repo in between.
    """
    check_plan(plan)
    _check_dest(dest)
    if dest.lower() in (results_repo.lower(), plan.source_dataset.lower()):
        raise ShareError(
            f"{dest} is the results repo or the source dataset. Share to a separate repo."
        )
    if plan.settings["repo"] != dest:
        raise ShareError(
            f"The board was built for {plan.settings['repo']}, not {dest}. "
            f"Build it again with board_repo={dest!r}."
        )
    api = api or HfApi()
    existing = _destination_files(api, dest, plan.alias, replace)
    created = existing is None
    if existing is None:
        api.create_repo(dest, repo_type="dataset", private=not public, exist_ok=False)
        # Check the new repo too: another writer could commit before this read.
        existing = _destination_files(api, dest, plan.alias, replace)
        if existing is None:
            raise ShareError(f"{dest} was not found after it was created.")
    revision, present = existing
    operations: list[CommitOperationAdd | CommitOperationDelete] = [
        CommitOperationAdd(path_in_repo=path, path_or_fileobj=data)
        for path, data in plan.files.items()
    ]
    operations += [
        CommitOperationDelete(path_in_repo=path)
        for path in SHARE_FILES
        if path in present and path not in plan.files
    ]
    api.create_commit(
        dest,
        repo_type="dataset",
        operations=operations,
        commit_message=f"ocr-bench share: {plan.alias}",
        parent_commit=revision,
    )
    is_private = bool(api.repo_info(dest, repo_type="dataset").private)
    logger.info("shared_board", repo=dest, files=sorted(plan.files), private=is_private)
    return PushResult(repo_id=dest, created=created, private=is_private)
