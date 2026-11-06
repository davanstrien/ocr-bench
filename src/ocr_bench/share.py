"""Share a judged evaluation as a scores-only board in a separate repo.

A results repo written by ``judge`` holds page text: each comparison row has
both OCR outputs and the judge's reason. ``share`` reads that repo and writes
ratings only to a different dataset repo. It never changes the visibility of
an existing repo.

Two checks run on the final file contents, before any Hub call:

1. Structure. Each file is one that ``share`` writes, each row has exactly the
   allowed columns, and each string value is one the board is meant to show:
   a model name, a slice name, a status or a setting.
2. Page text. No aligned 32-character chunk of a compared output or a judge
   reason occurs in any file. A copied fragment of 63 or more characters always
   holds such a chunk, so it is always found. A chunk with fewer than 4 distinct
   letters or digits is skipped: it is markup or layout (a markdown or HTML table
   rule, dot leaders, a run of spaces), not page text, and the board's own tables
   use the same markup.

If a check fails, :class:`PrivacyError` is raised and nothing is written.
"""

from __future__ import annotations

import datetime
import json
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import structlog
from datasets import load_dataset
from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi
from huggingface_hub.errors import RepositoryNotFoundError

from ocr_bench import __version__
from ocr_bench.dataset import DatasetError
from ocr_bench.elo import MIN_PAGES_TO_RESOLVE, ComparisonResult, Leaderboard, compute_elo
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
SLICE_BOARD_COLUMNS = ("slice", *BOARD_COLUMNS)
SETTINGS_KEYS = (
    "alias",
    "judge_models",
    "criteria",
    "prompt_hash",
    "judge_text_mode",
    "max_ocr_text_len",
    "judge_image_dim",
    "min_chars",
    "pages",
    "comparisons",
    "auto_tied",
    "slice_column",
    "slices",
    "pages_not_in_a_slice_board",
    "method",
    "source_fingerprint",
    "judged_at",
    "shared_at",
    "ocr_bench_version",
)
SHARE_FILES = ("README.md", "overall.jsonl", "slices.jsonl", "board.json")
STATUSES = ("ranked", "degraded", "failed")

# A slice board needs as many pages as a resolved ranking does. A smaller slice
# is left out, which also keeps a per-page column (a title) from being published.
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
    """The files to publish, plus the private texts they are checked against."""

    alias: str
    files: dict[str, bytes]
    overall_rows: list[dict]
    slice_rows: list[dict]
    settings: dict
    allowed_strings: frozenset[str]
    source_dataset: str
    private_texts: list[str] = field(repr=False, default_factory=list)


# --- reading -----------------------------------------------------------------


def load_private_evaluation(results_repo: str) -> PrivateEvaluation:
    """Read the leaderboard, comparisons and latest metadata row of a results repo."""
    try:
        leaderboard = [dict(row) for row in load_dataset(results_repo, split="train")]
    except Exception as exc:
        raise ShareError(
            f"Could not read the leaderboard of {results_repo}. Is it a results repo "
            f"written by `ocr-bench judge`? ({exc})"
        ) from exc
    comparisons = load_existing_comparisons(results_repo)
    if not comparisons:
        raise ShareError(f"{results_repo} has no comparisons to share.")
    metadata_rows = load_existing_metadata(results_repo)
    if not metadata_rows:
        raise ShareError(f"{results_repo} has no metadata rows.")
    return PrivateEvaluation(leaderboard, comparisons, metadata_rows[-1])


def load_slice_values(metadata: dict, slice_column: str) -> list:
    """Read one column of the judged source dataset, in row order.

    ``sample_idx`` in a comparison is a row position in the first loaded
    source config, and every config has the same rows in the same order (judge
    checks this). So the column is read from the first config that has it.
    """
    source = metadata["source_dataset"]
    split = metadata.get("source_split") or "train"
    configs = json.loads(metadata.get("source_configs") or "[]")
    revisions = json.loads(metadata.get("source_revisions") or "{}")
    errors = []
    for config in configs or [None]:
        try:
            ds = load_dataset(
                source,
                name=config,
                split=split,
                revision=revisions.get(config, "main") if config else None,
                columns=[slice_column],
            )
        except Exception as exc:
            errors.append(f"{config or 'default'}: {exc}")
            continue
        if slice_column in ds.column_names:
            return list(ds[slice_column])
    raise ShareError(
        f"Column '{slice_column}' was not found in {source} (split {split}). " + "; ".join(errors)
    )


# --- fitting -----------------------------------------------------------------


def board_models(leaderboard: list[dict]) -> list[str]:
    """Every model of the judged grid, failed ones included, as judge fitted them."""
    return [row["model"] for row in leaderboard]


def in_grid(comparisons: list[ComparisonResult], models: Iterable[str]) -> list[ComparisonResult]:
    """Comparisons between two models of the grid. Out-of-grid rows are history only."""
    grid = set(models)
    return [c for c in comparisons if c.model_a in grid and c.model_b in grid]


def check_board_matches(board: Leaderboard, leaderboard: list[dict], results_repo: str) -> None:
    """Refuse to share when a refit of the comparisons differs from the published board.

    This happens when a later judge run stopped after a checkpoint, so the
    comparisons hold verdicts the leaderboard does not include yet.
    """
    for row in leaderboard:
        if row.get("elo") is None:
            continue
        refit = board.elo.get(row["model"])
        if refit is None or abs(round(refit) - row["elo"]) > ELO_TOLERANCE:
            raise ShareError(
                f"The leaderboard of {results_repo} does not match its comparisons "
                f"({row['model']}: board {row['elo']}, refit "
                f"{'none' if refit is None else round(refit)}). A judge run probably "
                "stopped after a checkpoint. Run judge again to completion, then share."
            )


def _public_rows(
    board: Leaderboard,
    *,
    failed_models: list[str],
    failed_outputs: dict[str, int],
    slice_name: str | None = None,
) -> list[dict]:
    rows = []
    rank = 0
    for row in build_leaderboard_rows(board, failed_models, failed_outputs):
        failed = row["status"] == "failed"
        if not failed:
            rank += 1
        public = {
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
        if slice_name is not None:
            public = {"slice": slice_name, **public}
        rows.append(public)
    return rows


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
    too_long = [label for label in groups if len(label) > MAX_LABEL_LENGTH]
    if too_long:
        raise ShareError(
            f"{len(too_long)} slice labels are longer than {MAX_LABEL_LENGTH} characters. "
            "Use a column of short labels."
        )
    return groups, unlabelled


def check_judged_sample(values: list, metadata: dict, sample_ids: set[int]) -> None:
    """Refuse when the source rows differ from the rows judge sampled."""
    n_rows = len(values)
    if sample_ids and max(sample_ids) >= n_rows:
        raise ShareError(
            f"The source dataset has {n_rows} rows, but a comparison is on row "
            f"{max(sample_ids)}. The source changed after judging."
        )
    expected = set(sample_indices(n_rows, metadata.get("max_samples"), metadata.get("seed", 42)))
    if not sample_ids <= expected:
        raise ShareError(
            "The judged pages are not the sample that judge draws from the source "
            "dataset now. The source changed after judging, so slice labels could be "
            "on the wrong pages."
        )


# --- files -------------------------------------------------------------------


def _jsonl(rows: list[dict]) -> bytes:
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()


def _md(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _table(rows: list[dict]) -> list[str]:
    lines = [
        "| Rank | Model | Params | ELO | 95% CI | Pages | Wins | Losses | Ties |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        if row["status"] == "failed":
            lines.append(
                f"| — | {_md(row['model'])} | {row['params'] or ''} | FAILED | — | — | — | — | — |"
            )
            continue
        ci = f"{row['elo_low']}–{row['elo_high']}" if row.get("elo_low") is not None else "—"
        mark = " ⚠" if row["status"] == "degraded" else ""
        lines.append(
            f"| {row['rank']} | {_md(row['model'])}{mark} | {row['params'] or ''} "
            f"| {row['elo']} | {ci} | {row['pages']} | {row['wins']} | {row['losses']} "
            f"| {row['ties']} |"
        )
    return lines


def build_card(settings: dict, overall_rows: list[dict], slice_rows: list[dict]) -> str:
    """The README of the share repo. Built only from checked rows and settings."""
    alias = settings["alias"]
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
        f"# OCR board: {_md(alias)}",
        "",
        "Pairwise VLM-as-judge ranking of OCR models on a private evaluation set. "
        "This repo holds scores only: the pages, the OCR outputs and the judge's "
        "reasons are not published.",
        "",
        f"## Overall ({settings['pages']} pages)",
        "",
        *_table(overall_rows),
    ]
    if any(row["status"] == "degraded" for row in overall_rows):
        lines += [
            "",
            "⚠ The model returned error placeholders on some pages. Those pages were "
            "not judged for it, so its rank leaves them out.",
        ]
    if slice_rows:
        lines += ["", f"## By {_md(settings['slice_column'])}"]
        for name in settings["slices"]:
            rows = [row for row in slice_rows if row["slice"] == name]
            pages = max(row["pages"] for row in rows)
            lines += ["", f"### {_md(name)} ({pages} pages)", "", *_table(rows)]
        if settings["pages_not_in_a_slice_board"]:
            lines += [
                "",
                f"{settings['pages_not_in_a_slice_board']} judged pages are in no slice "
                f"board: they have no label, or their slice has fewer than "
                f"{MIN_SLICE_PAGES} judged pages.",
            ]
    judges = ", ".join(_md(name) for name in settings["judge_models"]) or "not recorded"
    lines += [
        "",
        "## Settings",
        "",
        f"- **Judge**: {judges}",
        f"- **Judge criteria**: {_md(settings['criteria'])}",
        f"- **Judge prompt hash**: `{settings['prompt_hash'] or 'not recorded'}`",
        f"- **Judge text mode**: {settings['judge_text_mode']}",
        f"- **OCR text cap**: {settings['max_ocr_text_len']} characters per output",
        f"- **Judge image cap**: {settings['judge_image_dim']} px on the longer side",
        f"- **Comparisons**: {settings['comparisons']} "
        f"({settings['auto_tied']} identical-output ties)",
        f"- **Method**: {settings['method']}",
        f"- **Source fingerprint**: `{settings['source_fingerprint'] or 'not recorded'}` "
        "(identifies the exact evaluation data without publishing it)",
        f"- **Judged**: {settings['judged_at']}; **shared**: {settings['shared_at']}",
        "",
        "`load_dataset(repo)` gives the overall board"
        + ('; `load_dataset(repo, "slices")` gives the slice boards.' if slice_rows else ".")
        + " `board.json` has the settings.",
        "",
        f"*Generated by [ocr-bench](https://github.com/davanstrien/ocr-bench) "
        f"{settings['ocr_bench_version']}*",
    ]
    return "\n".join(lines) + "\n"


def build_share_files(
    settings: dict, overall_rows: list[dict], slice_rows: list[dict]
) -> dict[str, bytes]:
    files = {
        "README.md": build_card(settings, overall_rows, slice_rows).encode(),
        "overall.jsonl": _jsonl(overall_rows),
        "board.json": (
            json.dumps(settings, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
        ).encode(),
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
    if isinstance(value, list) and all(isinstance(v, str) and v in allowed for v in value):
        return
    raise PrivacyError(
        f"{path}: the value of '{key}' is not one the board is meant to show. Nothing was written."
    )


def check_structure(files: dict[str, bytes], allowed: frozenset[str]) -> None:
    """Allow-list check: known files, exact columns, only allowed string values."""
    unknown = sorted(set(files) - set(SHARE_FILES))
    if unknown:
        raise PrivacyError(f"Unexpected files {unknown}. Nothing was written.")
    for path, columns in (("overall.jsonl", BOARD_COLUMNS), ("slices.jsonl", SLICE_BOARD_COLUMNS)):
        if path not in files:
            continue
        for line in files[path].decode().splitlines():
            row = json.loads(line)
            if tuple(row) != columns:
                raise PrivacyError(
                    f"{path}: a row has columns {sorted(row)}, not the allowed columns. "
                    "Nothing was written."
                )
            for key, value in row.items():
                _check_value(path, key, value, allowed)
    settings = json.loads(files["board.json"].decode())
    if sorted(settings) != sorted(SETTINGS_KEYS):
        raise PrivacyError("board.json: unexpected keys. Nothing was written.")
    for key, value in settings.items():
        _check_value("board.json", key, value, allowed)


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
        for line in text.splitlines():
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
    """Both privacy checks, on the exact bytes that would be written."""
    check_structure(plan.files, plan.allowed_strings)
    check_no_page_text(plan.files, plan.private_texts)


# --- plan and push -----------------------------------------------------------


def _check_alias(alias: str) -> str:
    alias = alias.strip()
    if not alias or len(alias) > MAX_LABEL_LENGTH or "\n" in alias:
        raise ShareError(f"--alias must be 1–{MAX_LABEL_LENGTH} characters on one line.")
    return alias


def plan_share(
    evaluation: PrivateEvaluation,
    *,
    results_repo: str,
    alias: str,
    slice_column: str | None = None,
    slice_values: Callable[[dict, str], list] = load_slice_values,
    now: str | None = None,
) -> SharePlan:
    """Fit the boards, build the files and check them. Makes no Hub writes."""
    alias = _check_alias(alias)
    metadata = evaluation.metadata
    models = board_models(evaluation.leaderboard)
    comparisons = in_grid(evaluation.comparisons, models)
    if not comparisons:
        raise ShareError(f"{results_repo} has no comparisons between its leaderboard models.")
    failed_models = json.loads(metadata.get("failed_models") or "[]")
    failed_outputs = json.loads(metadata.get("failed_outputs") or "{}")

    board = compute_elo(comparisons, models, n_bootstrap=N_BOOTSTRAP)
    check_board_matches(board, evaluation.leaderboard, results_repo)
    overall_rows = _public_rows(board, failed_models=failed_models, failed_outputs=failed_outputs)
    sample_ids = {c.sample_idx for c in comparisons}

    slice_rows: list[dict] = []
    slice_names: list[str] = []
    not_in_a_slice = 0
    if slice_column:
        values = slice_values(metadata, slice_column)
        check_judged_sample(values, metadata, sample_ids)
        groups, not_in_a_slice = slice_groups(values, sample_ids)
        for name in sorted(groups):
            pages = groups[name]
            if len(pages) < MIN_SLICE_PAGES:
                not_in_a_slice += len(pages)
                continue
            slice_comparisons = [c for c in comparisons if c.sample_idx in pages]
            slice_models = [
                m for m in models if any(m in (c.model_a, c.model_b) for c in slice_comparisons)
            ]
            slice_board = compute_elo(slice_comparisons, slice_models, n_bootstrap=N_BOOTSTRAP)
            slice_rows += _public_rows(
                slice_board,
                failed_models=[],
                failed_outputs=failed_outputs,
                slice_name=name,
            )
            slice_names.append(name)

    judge_models = json.loads(metadata.get("judge_models") or "[]")
    settings = {
        "alias": alias,
        "judge_models": judge_models,
        "criteria": metadata.get("criteria") or "default",
        "prompt_hash": metadata.get("prompt_hash") or "",
        "judge_text_mode": metadata.get("judge_text_mode") or "normalized",
        "max_ocr_text_len": metadata.get("max_ocr_text_len"),
        "judge_image_dim": metadata.get("judge_image_dim"),
        "min_chars": metadata.get("min_chars"),
        "pages": len(sample_ids),
        "comparisons": len(comparisons),
        "auto_tied": sum(1 for c in comparisons if c.agreement == "auto"),
        "slice_column": slice_column,
        "slices": slice_names,
        "pages_not_in_a_slice_board": not_in_a_slice if slice_column else 0,
        "method": METHOD,
        "source_fingerprint": metadata.get("source_fingerprint") or "",
        "judged_at": metadata.get("timestamp") or "",
        "shared_at": now or datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "ocr_bench_version": __version__,
    }
    allowed = frozenset(
        {
            *models,
            *slice_names,
            *STATUSES,
            *(row["params"] for row in overall_rows if row["params"]),
            *judge_models,
            *(v for v in settings.values() if isinstance(v, str)),
        }
    )
    files = build_share_files(settings, overall_rows, slice_rows)
    plan = SharePlan(
        alias=alias,
        files=files,
        overall_rows=overall_rows,
        slice_rows=slice_rows,
        settings=settings,
        allowed_strings=allowed,
        source_dataset=metadata.get("source_dataset", ""),
        private_texts=private_texts(evaluation.comparisons),
    )
    check_plan(plan)
    return plan


def write_local(plan: SharePlan, out_dir: Path) -> list[Path]:
    """Write the checked files to a local folder (``--dry-run``)."""
    check_plan(plan)
    out_dir.mkdir(parents=True, exist_ok=True)
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


def _check_destination(api: HfApi, dest: str) -> bool:
    """Return whether dest exists. Refuse a repo that share did not write."""
    try:
        files = api.list_repo_files(dest, repo_type="dataset")
    except RepositoryNotFoundError:
        return False
    if not set(files) - {".gitattributes"}:
        return True
    readme = ""
    if "README.md" in files:
        readme = Path(api.hf_hub_download(dest, "README.md", repo_type="dataset")).read_text()
    if f"- {SHARE_TAG}" not in readme:
        raise ShareError(
            f"{dest} already has files and was not written by `ocr-bench share`. "
            "Choose a new or empty repo."
        )
    return True


def push_share(
    plan: SharePlan,
    dest: str,
    *,
    results_repo: str,
    private: bool = False,
    api: HfApi | None = None,
) -> PushResult:
    """Check the files again, then write them to ``dest`` in one commit.

    A missing ``dest`` is created (public unless ``private``). An existing one
    keeps its visibility. ``dest`` must not be the results repo or the source.
    """
    check_plan(plan)
    if dest in (results_repo, plan.source_dataset):
        raise ShareError(
            f"{dest} is the results repo or the source dataset. Share to a separate repo."
        )
    api = api or HfApi()
    existed = _check_destination(api, dest)
    if not existed:
        api.create_repo(dest, repo_type="dataset", private=private, exist_ok=False)
    stale = []
    if existed:
        present = set(api.list_repo_files(dest, repo_type="dataset"))
        stale = [path for path in SHARE_FILES if path in present and path not in plan.files]
    operations: list[CommitOperationAdd | CommitOperationDelete] = [
        CommitOperationAdd(path_in_repo=path, path_or_fileobj=data)
        for path, data in plan.files.items()
    ]
    operations += [CommitOperationDelete(path_in_repo=path) for path in stale]
    api.create_commit(
        dest,
        repo_type="dataset",
        operations=operations,
        commit_message=f"ocr-bench share: {plan.alias}",
    )
    info = api.repo_info(dest, repo_type="dataset")
    logger.info("shared_board", repo=dest, files=sorted(plan.files), private=bool(info.private))
    return PushResult(repo_id=dest, created=not existed, private=bool(info.private))
