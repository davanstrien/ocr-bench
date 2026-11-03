"""Results viewer — data loading and helpers for OCR bench results."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog
from datasets import load_dataset

if TYPE_CHECKING:
    from PIL import Image

logger = structlog.get_logger()


def _latest_revision(repo_id: str) -> str | None:
    """Resolve the results repo's latest commit SHA.

    Pinning ``load_dataset`` to this explicit revision (rather than the bare
    ``main`` pointer) guarantees the viewer serves the just-published results
    instead of a stale cached snapshot — the freshness half of the incident in
    issue #37. Returns ``None`` if it can't be resolved, in which case callers
    fall back to ``load_dataset``'s default resolution.
    """
    try:
        from huggingface_hub import HfApi

        return HfApi().dataset_info(repo_id).sha
    except Exception as exc:
        logger.warning("could_not_resolve_revision", repo=repo_id, error=str(exc))
        return None


def load_results(repo_id: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load leaderboard and comparisons from a Hub results dataset.

    Resolves the repo's latest revision up front and pins every load to it, so
    a warm cache never serves stale results. Tries the default config first
    (new repos), then falls back to the named ``leaderboard`` config (old repos).

    Returns:
        (leaderboard_rows, comparison_rows)
    """
    revision = _latest_revision(repo_id)

    try:
        leaderboard_ds = load_dataset(repo_id, split="train", revision=revision)
        leaderboard_rows = [dict(row) for row in leaderboard_ds]
    except Exception:
        leaderboard_ds = load_dataset(
            repo_id, name="leaderboard", split="train", revision=revision
        )
        leaderboard_rows = [dict(row) for row in leaderboard_ds]

    try:
        comparisons_ds = load_dataset(
            repo_id, name="comparisons", split="train", revision=revision
        )
    except Exception:
        logger.warning("no_comparisons_config", repo=repo_id)
        return leaderboard_rows, []
    comparison_rows = [dict(row) for row in comparisons_ds]

    return leaderboard_rows, comparison_rows


def _load_source_metadata(repo_id: str) -> dict[str, Any]:
    """Load the latest run metadata used to resolve source images."""
    revision = _latest_revision(repo_id)
    try:
        meta_ds = load_dataset(repo_id, name="metadata", split="train", revision=revision)
        if len(meta_ds) > 0:
            return dict(meta_ds[-1])
    except Exception as exc:
        logger.warning("could_not_load_metadata", repo=repo_id, error=str(exc))
    return {}


class ImageLoader:
    """Lazy image loader — fetches images from source dataset by sample_idx."""

    def __init__(
        self,
        source_dataset: str,
        from_prs: bool = False,
        source_split: str = "train",
        source_configs: list[str] | None = None,
        source_revisions: dict[str, str] | None = None,
    ):
        self._source = source_dataset
        self._from_prs = from_prs
        self._source_split = source_split
        # The configs the run judged, from its metadata. Images must come from
        # one of these: another open PR may cover different pages.
        self._source_configs = list(source_configs or [])
        # Where the run read each config ("refs/pr/N" or "main"), if recorded.
        self._source_revisions = dict(source_revisions or {})
        self._cache: dict[int, Any] = {}
        self._image_col: str | None = None
        self._config_name: str | None = None
        self._pr_revision: str | None = None
        self._available = True
        self._init_done = False

    def _init_source(self) -> None:
        """Lazy init: discover image column and PR revision on first call."""
        if self._init_done:
            return
        self._init_done = True

        try:
            if self._from_prs and self._source_revisions:
                # Use the revisions the run recorded: PRs may since have been
                # merged, closed or superseded.
                configs = self._source_configs or list(self._source_revisions)
                for config in configs:
                    revision = self._source_revisions.get(config, "main")
                    if revision != "main":
                        self._config_name, self._pr_revision = config, revision
                        break
                else:
                    self._config_name = configs[0] if configs else None
            elif self._from_prs:
                # Older metadata without revisions: rediscover open PRs.
                from ocr_bench.dataset import discover_pr_configs

                _, revisions = discover_pr_configs(
                    self._source, only=self._source_configs or None
                )
                # Prefer the run's own configs, in its order; a selected config
                # without an open PR was loaded from main.
                for config in self._source_configs or list(revisions):
                    self._config_name = config
                    self._pr_revision = revisions.get(config)
                    if self._pr_revision:
                        break

            # Probe for image column by loading 1 row
            probe = load_dataset(**self._load_kwargs(f"{self._source_split}[:1]"))
            for col in probe.column_names:
                if col == "image" or "image" in col.lower():
                    self._image_col = col
                    break
            if not self._image_col:
                logger.info("no_image_column_in_source", source=self._source)
                self._available = False
        except Exception as exc:
            logger.warning("image_loader_init_failed", source=self._source, error=str(exc))
            self._available = False

    def _load_kwargs(self, split: str) -> dict[str, Any]:
        """load_dataset arguments for the resolved config and revision."""
        kwargs: dict[str, Any] = {"path": self._source, "split": split}
        if self._config_name:
            kwargs["name"] = self._config_name
        if self._pr_revision:
            kwargs["revision"] = self._pr_revision
        return kwargs

    def get(self, sample_idx: int) -> Image.Image | None:
        """Fetch image for a sample index. Returns None on failure."""
        self._init_source()
        if not self._available or self._image_col is None:
            return None
        if sample_idx in self._cache:
            return self._cache[sample_idx]
        try:
            row = load_dataset(
                **self._load_kwargs(f"{self._source_split}[{sample_idx}:{sample_idx + 1}]")
            )
            img = row[0][self._image_col]
            self._cache[sample_idx] = img
            return img
        except Exception as exc:
            logger.debug("image_load_failed", sample_idx=sample_idx, error=str(exc))
            return None


def _filter_comparisons(
    comparisons: list[dict[str, Any]],
    winner_filter: str,
    model_filter: str,
) -> list[dict[str, Any]]:
    """Filter comparison rows by winner and model."""
    filtered = comparisons
    if winner_filter and winner_filter != "All":
        filtered = [c for c in filtered if c.get("winner") == winner_filter]
    if model_filter and model_filter != "All":
        filtered = [
            c
            for c in filtered
            if c.get("model_a") == model_filter or c.get("model_b") == model_filter
        ]
    return filtered


def _winner_badge(winner: str) -> str:
    """Return a badge string for the winner."""
    if winner == "A":
        return "Winner: A"
    elif winner == "B":
        return "Winner: B"
    else:
        return "Tie"


def _model_label(model: str, col: str) -> str:
    """Format model name with optional column name. Avoids empty parens."""
    if col:
        return f"{model} ({col})"
    return model


def _build_pair_summary(comparisons: list[dict[str, Any]]) -> str:
    """Build a win/loss summary string for each model pair."""
    from collections import Counter

    pair_counts: dict[tuple[str, str], Counter[str]] = {}
    for c in comparisons:
        ma = c.get("model_a", "")
        mb = c.get("model_b", "")
        winner = c.get("winner", "tie")
        key = (ma, mb) if ma <= mb else (mb, ma)
        if key not in pair_counts:
            pair_counts[key] = Counter()
        # Track from perspective of first model in sorted pair
        if winner == "A":
            actual_winner = ma
        elif winner == "B":
            actual_winner = mb
        else:
            actual_winner = "tie"

        if actual_winner == key[0]:
            pair_counts[key]["W"] += 1
        elif actual_winner == key[1]:
            pair_counts[key]["L"] += 1
        else:
            pair_counts[key]["T"] += 1

    if not pair_counts:
        return ""

    parts = []
    for (ma, mb), counts in sorted(pair_counts.items()):
        short_a = ma.split("/")[-1] if "/" in ma else ma
        short_b = mb.split("/")[-1] if "/" in mb else mb
        wins, losses, ties = counts["W"], counts["L"], counts["T"]
        parts.append(f"**{short_a}** vs **{short_b}**: {wins}W {losses}L {ties}T")
    return " | ".join(parts)

