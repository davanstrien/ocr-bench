"""Hub publishing — push comparisons, leaderboard, and metadata configs to HF Hub."""

from __future__ import annotations

import datetime
import json
from dataclasses import dataclass, field

import structlog
from datasets import Dataset, load_dataset
from huggingface_hub import HfApi

from ocr_bench.elo import ComparisonResult, Leaderboard, compute_elo
from ocr_bench.run import MODEL_REGISTRY

logger = structlog.get_logger()


@dataclass
class EvalMetadata:
    """Metadata for an evaluation run, stored alongside results on Hub.

    The comparison counts describe this run's judge effort:

    - ``total_comparisons``: pairs actually sent to a judge (judge calls).
    - ``valid_comparisons``: judged pairs that returned a usable verdict —
      excludes judge failures and auto-ties.
    - ``auto_tied``: identical-output pairs scored as ties *without* a judge
      call. Not judge calls, so excluded from the two counts above, but they
      still enter the leaderboard as ordinary ties.

    So the comparison log powering the leaderboard is ``valid_comparisons +
    auto_tied`` for a fresh run (existing comparisons add to it on incremental
    runs).
    """

    source_dataset: str
    judge_models: list[str]
    seed: int
    max_samples: int
    total_comparisons: int
    valid_comparisons: int
    auto_tied: int = 0
    # Global comparison budget for the run (--max-comparisons); None = uncapped.
    # ``budget_exhausted`` records whether the run stopped because it hit the cap
    # (as opposed to converging or exhausting the samples).
    max_comparisons: int | None = None
    budget_exhausted: bool = False
    from_prs: bool = False
    # model → count of error-sentinel outputs excluded from judging (issue #46).
    failed_outputs: dict[str, int] = field(default_factory=dict)
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
        "auto_tied": metadata.auto_tied,
        "max_comparisons": metadata.max_comparisons,
        "budget_exhausted": metadata.budget_exhausted,
        "from_prs": metadata.from_prs,
        "failed_outputs": json.dumps(metadata.failed_outputs),
        "timestamp": metadata.timestamp,
    }


def _align_metadata_rows(rows: list[dict]) -> list[dict]:
    """Give every metadata row the same keys (union), filling gaps with None.

    ``Dataset.from_list`` infers its schema from the *first* row only, so a
    newer row carrying columns that older rows lack (e.g. the budget fields
    added here) would be silently dropped whenever an older row comes first.
    Taking the union of keys keeps the append-only metadata log
    forward-compatible as new fields are introduced.
    """
    keys: dict[str, None] = {}
    for row in rows:
        keys.update(dict.fromkeys(row))
    return [{k: row.get(k) for k in keys} for row in rows]


def publish_checkpoint(
    repo_id: str,
    results: list[ComparisonResult],
    model_names: list[str],
) -> None:
    """Push ONLY the comparisons config as a mid-run checkpoint.

    Append-only and cheap: unlike :func:`publish_results` this writes no
    leaderboard, README, or metadata — those churn the repo and are written
    once at the final publish. The point of a checkpoint is durability: a run
    killed between checkpoints loses at most the comparisons judged since the
    last one, and a relaunch WITHOUT ``--full-rejudge`` picks the checkpointed
    comparisons back up (see ``load_existing_comparisons`` + ``skip_samples`` in
    ``cli.cmd_judge``).

    ``results`` must be the *full* accumulated set (existing + new so far);
    ``push_to_hub`` replaces the config's data, so passing the whole set each
    time keeps the published comparisons config complete and monotonic.

    Reuses :func:`compute_elo` with bootstrapping disabled purely to build the
    canonicalised comparison rows the same way the final publish does — the
    returned ELO/CIs are discarded — so checkpointed and final comparison logs
    are identical.
    """
    board = compute_elo(results, model_names, n_bootstrap=0)
    if not board.comparison_log:
        return
    comp_ds = Dataset.from_list(board.comparison_log)
    comp_ds.push_to_hub(repo_id, config_name="comparisons")
    logger.info("published_checkpoint", repo=repo_id, n=len(board.comparison_log))


def publish_results(
    repo_id: str,
    board: Leaderboard,
    metadata: EvalMetadata,
    existing_metadata: list[dict] | None = None,
    license_id: str | None = None,
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

    # Metadata — append-only. Align all rows to the union of keys so a newer
    # row's columns (auto_tied, budget fields, failed_outputs) aren't dropped
    # when an older row written before those fields existed comes first.
    meta_row = build_metadata_row(metadata)
    all_meta = _align_metadata_rows((existing_metadata or []) + [meta_row])
    Dataset.from_list(all_meta).push_to_hub(repo_id, config_name="metadata")
    logger.info("published_metadata", repo=repo_id, n=len(all_meta))

    # README — auto-generated dataset card with leaderboard
    readme = _build_readme(repo_id, rows, board, metadata, license_id=license_id)
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
    license_id: str | None = None,
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
    # Break the leaderboard's comparison log down by how each verdict was
    # reached: judged pairs vs identical-output auto-ties (agreement "auto").
    n_comparisons = len(board.comparison_log)
    n_auto = sum(1 for r in board.comparison_log if r.get("agreement") == "auto")
    n_judged = n_comparisons - n_auto
    if n_auto:
        comparisons_str = f"{n_judged} judged + {n_auto} auto-tied ({n_comparisons} total)"
    else:
        comparisons_str = str(n_comparisons)

    # Models that emitted error sentinels instead of transcriptions. These
    # outputs were excluded from judging, so a high count means the run failed
    # on this corpus — the card must not let it read as "ranked low" (issue #46).
    failed = metadata.failed_outputs
    if isinstance(failed, str):
        failed = json.loads(failed) if failed else {}
    failed_outputs: dict[str, int] = {
        model: count for model, count in (failed or {}).items() if count
    }

    # The card license describes the published results DATA (which embeds
    # OCR text derived from the source dataset), not this tool — so there is
    # no correct default; it's declared per-run via --license or set on the
    # Hub repo by the publisher.
    lines = ["---"]
    if license_id:
        lines.append(f"license: {license_id}")
    lines += [
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
        # Escape pipes so arbitrary model names can't break the table
        model = str(row["model"]).replace("|", "\\|")
        if row["model"] in failed_outputs:
            # Flag rows whose model produced excluded error sentinels so a
            # failed run is never mistaken for a genuinely low-ranked model.
            model = f"{model} ⚠"
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

    if failed_outputs:
        lines += [
            "",
            "## ⚠ Failed outputs",
            "",
            "The models below emitted error sentinels (e.g. `[OCR ERROR]`, "
            "`[OCR FAILED]`) instead of transcriptions on some pages — usually a "
            "crashed or misconfigured run, **not** poor OCR quality. Those outputs "
            "were **excluded from judging**, so a high count means the model did "
            "not produce comparable output on this corpus. Do not read a flagged "
            "model's rank as a quality signal.",
            "",
            "| Model | Excluded outputs |",
            "|-------|------------------|",
        ]
        for model, count in sorted(failed_outputs.items(), key=lambda kv: -kv[1]):
            safe_model = str(model).replace("|", "\\|")
            lines.append(f"| {safe_model} | {count} |")

    lines += [
        "",
        "## Details",
        "",
        f"- **Source dataset**: [`{metadata.source_dataset}`]"
        f"(https://huggingface.co/datasets/{metadata.source_dataset})",
        f"- **Judge**: {judge_str}",
        f"- **Comparisons**: {comparisons_str}",
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
