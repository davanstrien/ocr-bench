"""Hub publishing — push comparisons, leaderboard, and metadata configs to HF Hub."""

from __future__ import annotations

import datetime
import json
from dataclasses import dataclass

import structlog
from datasets import Dataset

from ocr_bench.elo import Leaderboard

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


def build_leaderboard_rows(board: Leaderboard) -> list[dict]:
    """Convert a Leaderboard into rows suitable for a Hub dataset."""
    rows = []
    for model, elo in board.ranked:
        total = board.wins[model] + board.losses[model] + board.ties[model]
        row = {
            "model": model,
            "elo": round(elo),
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
) -> None:
    """Push evaluation results to Hub as a dataset with three configs.

    Configs:
      - ``comparisons``: Raw comparison log from the leaderboard.
      - ``leaderboard``: Ranked model table with ELO, wins, losses, ties.
      - ``metadata``: Single-row dataset with eval run metadata.
    """
    # Comparisons
    if board.comparison_log:
        comp_ds = Dataset.from_list(board.comparison_log)
        comp_ds.push_to_hub(repo_id, config_name="comparisons")
        logger.info("published_comparisons", repo=repo_id, n=len(board.comparison_log))

    # Leaderboard
    rows = build_leaderboard_rows(board)
    Dataset.from_list(rows).push_to_hub(repo_id, config_name="leaderboard")
    logger.info("published_leaderboard", repo=repo_id, n=len(rows))

    # Metadata
    meta_row = build_metadata_row(metadata)
    Dataset.from_list([meta_row]).push_to_hub(repo_id, config_name="metadata")
    logger.info("published_metadata", repo=repo_id)
