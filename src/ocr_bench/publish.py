"""Hub publishing — push comparisons, leaderboard, and metadata configs to HF Hub."""

from __future__ import annotations

import datetime
import json
from dataclasses import dataclass

import structlog
from datasets import Dataset, load_dataset
from huggingface_hub import HfApi

from ocr_bench.elo import ComparisonResult, Leaderboard
from ocr_bench.run import MODEL_REGISTRY

logger = structlog.get_logger()


@dataclass
class EvalMetadata:
    """Metadata for an evaluation run, stored alongside results on Hub."""

    source_dataset: str
    judge_models: list[str]
    seed: int
    max_samples: int
    total_comparisons: int
    valid_comparisons: int
    from_prs: bool = False
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.datetime.now(datetime.UTC).isoformat()


def load_existing_comparisons(repo_id: str) -> list[ComparisonResult]:
    """Load existing comparisons from a Hub results repo.

    The stored winner is already unswapped (canonical), so ``swapped=False``.
    Returns an empty list if the repo or config doesn't exist.
    """
    try:
        ds = load_dataset(repo_id, name="comparisons", split="train")
    except Exception as exc:
        logger.info("no_existing_comparisons", repo=repo_id, reason=str(exc))
        return []

    results = []
    for row in ds:
        results.append(
            ComparisonResult(
                sample_idx=row["sample_idx"],
                model_a=row["model_a"],
                model_b=row["model_b"],
                winner=row["winner"],
                reason=row.get("reason", ""),
                agreement=row.get("agreement", "1/1"),
                swapped=False,
                text_a=row.get("text_a", ""),
                text_b=row.get("text_b", ""),
                col_a=row.get("col_a", ""),
                col_b=row.get("col_b", ""),
            )
        )
    logger.info("loaded_existing_comparisons", repo=repo_id, n=len(results))
    return results


def load_existing_metadata(repo_id: str) -> list[dict]:
    """Load existing metadata rows from a Hub results repo.

    Returns an empty list if the repo or config doesn't exist.
    """
    try:
        ds = load_dataset(repo_id, name="metadata", split="train")
        return [dict(row) for row in ds]
    except Exception as exc:
        logger.info("no_existing_metadata", repo=repo_id, reason=str(exc))
        return []


def _get_model_sizes() -> dict[str, str]:
    """Build model_id → size lookup from the model registry."""
    return {cfg.model_id: cfg.size for cfg in MODEL_REGISTRY.values()}


def build_leaderboard_rows(board: Leaderboard) -> list[dict]:
    """Convert a Leaderboard into rows suitable for a Hub dataset."""
    sizes = _get_model_sizes()
    rows = []
    for model, elo in board.ranked:
        total = board.wins[model] + board.losses[model] + board.ties[model]
        row = {
            "model": model,
            "elo": round(elo),
            "params": sizes.get(model, ""),
            "wins": board.wins[model],
            "losses": board.losses[model],
            "ties": board.ties[model],
            "win_pct": round(board.wins[model] / total * 100) if total > 0 else 0,
        }
        if board.elo_ci and model in board.elo_ci:
            lo, hi = board.elo_ci[model]
            row["elo_low"] = round(lo)
            row["elo_high"] = round(hi)
        rows.append(row)
    return rows


def build_metadata_row(metadata: EvalMetadata) -> dict:
    """Convert EvalMetadata into a single row for a Hub dataset."""
    return {
        "source_dataset": metadata.source_dataset,
        "judge_models": json.dumps(metadata.judge_models),
        "seed": metadata.seed,
        "max_samples": metadata.max_samples,
        "total_comparisons": metadata.total_comparisons,
        "valid_comparisons": metadata.valid_comparisons,
        "from_prs": metadata.from_prs,
        "timestamp": metadata.timestamp,
    }


def publish_results(
    repo_id: str,
    board: Leaderboard,
    metadata: EvalMetadata,
    existing_metadata: list[dict] | None = None,
) -> None:
    """Push evaluation results to Hub as a dataset with multiple configs.

    Configs:
      - (default): Leaderboard table — ``load_dataset("repo")`` returns this.
      - ``leaderboard``: Same table, named config (backward compat for viewer).
      - ``comparisons``: Full comparison log from the board (caller merges
        existing + new before ``compute_elo``, so ``board.comparison_log``
        is already the complete set).
      - ``metadata``: Append-only run log. New row is appended to
        ``existing_metadata``.
    """
    # Comparisons
    if board.comparison_log:
        comp_ds = Dataset.from_list(board.comparison_log)
        comp_ds.push_to_hub(repo_id, config_name="comparisons")
        logger.info("published_comparisons", repo=repo_id, n=len(board.comparison_log))

    # Leaderboard — dual push: default config + named config
    rows = build_leaderboard_rows(board)
    lb_ds = Dataset.from_list(rows)
    lb_ds.push_to_hub(repo_id)
    lb_ds.push_to_hub(repo_id, config_name="leaderboard")
    logger.info("published_leaderboard", repo=repo_id, n=len(rows))

    # Metadata — append-only
    meta_row = build_metadata_row(metadata)
    all_meta = (existing_metadata or []) + [meta_row]
    Dataset.from_list(all_meta).push_to_hub(repo_id, config_name="metadata")
    logger.info("published_metadata", repo=repo_id, n=len(all_meta))

    # README — auto-generated dataset card with leaderboard
    readme = _build_readme(repo_id, rows, board, metadata)
    api = HfApi()
    api.upload_file(
        path_or_fileobj=readme.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
    )
    logger.info("published_readme", repo=repo_id)


def _build_readme(
    repo_id: str,
    rows: list[dict],
    board: Leaderboard,
    metadata: EvalMetadata,
) -> str:
    """Build a dataset card README with the leaderboard table."""
    has_ci = bool(board.elo_ci)
    source_short = metadata.source_dataset.split("/")[-1]
    judges = json.loads(
        metadata.judge_models
        if isinstance(metadata.judge_models, str)
        else json.dumps(metadata.judge_models)
    )
    judge_str = ", ".join(j.split("/")[-1] for j in judges) if judges else "N/A"
    n_comparisons = len(board.comparison_log)

    lines = [
        "---",
        "license: mit",
        "tags:",
        "  - ocr-bench",
        "  - leaderboard",
        "source_datasets:",
        f"  - {metadata.source_dataset}",
        "configs:",
        "  - config_name: default",
        "    data_files:",
        "      - split: train",
        "        path: data/train-*.parquet",
        "  - config_name: comparisons",
        "    data_files:",
        "      - split: train",
        "        path: comparisons/train-*.parquet",
        "  - config_name: leaderboard",
        "    data_files:",
        "      - split: train",
        "        path: leaderboard/train-*.parquet",
        "  - config_name: metadata",
        "    data_files:",
        "      - split: train",
        "        path: metadata/train-*.parquet",
        "---",
        "",
        f"# OCR Bench Results: {source_short}",
        "",
        "VLM-as-judge pairwise evaluation of OCR models. "
        "Rankings depend on document type — there is no single best OCR model.",
        "",
        "## Leaderboard",
        "",
    ]

    # Table header
    if has_ci:
        lines.append("| Rank | Model | Params | ELO | 95% CI | Wins | Losses | Ties | Win% |")
        lines.append("|------|-------|--------|-----|--------|------|--------|------|------|")
    else:
        lines.append("| Rank | Model | Params | ELO | Wins | Losses | Ties | Win% |")
        lines.append("|------|-------|--------|-----|------|--------|------|------|")

    for rank, row in enumerate(rows, 1):
        model = row["model"]
        elo = row["elo"]
        params = row.get("params", "")
        if has_ci and "elo_low" in row:
            ci = f"{row['elo_low']}\u2013{row['elo_high']}"
            lines.append(
                f"| {rank} | {model} | {params} | {elo} | {ci} "
                f"| {row['wins']} | {row['losses']} | {row['ties']} "
                f"| {row['win_pct']}% |"
            )
        else:
            lines.append(
                f"| {rank} | {model} | {params} | {elo} "
                f"| {row['wins']} | {row['losses']} | {row['ties']} "
                f"| {row['win_pct']}% |"
            )

    lines += [
        "",
        "## Details",
        "",
        f"- **Source dataset**: [`{metadata.source_dataset}`]"
        f"(https://huggingface.co/datasets/{metadata.source_dataset})",
        f"- **Judge**: {judge_str}",
        f"- **Comparisons**: {n_comparisons}",
        "- **Method**: Bradley-Terry MLE with bootstrap 95% CIs",
        "",
        "## Configs",
        "",
        f"- `load_dataset(\"{repo_id}\")` — leaderboard table",
        f"- `load_dataset(\"{repo_id}\", name=\"comparisons\")` "
        "— full pairwise comparison log",
        f"- `load_dataset(\"{repo_id}\", name=\"metadata\")` "
        "— evaluation run history",
        "",
        "*Generated by [ocr-bench](https://github.com/davanstrien/ocr-bench)*",
    ]

    return "\n".join(lines) + "\n"
