"""Decision helpers for opt-in targeted and size-aware adaptive judging."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Literal, cast

from ocr_bench.elo import Leaderboard
from ocr_bench.run import MODEL_REGISTRY

ModelPair = tuple[str, str]
PairStatus = Literal["resolved", "unresolved", "prefer-smaller"]

_SIZE_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([KMB])\s*$", re.IGNORECASE)
_SIZE_MULTIPLIERS = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000}


def normalize_model_pair(model_a: str, model_b: str) -> ModelPair:
    """Return a stable, order-independent model-pair key."""
    return (model_a, model_b) if model_a <= model_b else (model_b, model_a)


def parse_parameter_count(size: str) -> int | None:
    """Parse registry display sizes such as ``34.5M`` and ``8.3B``.

    Unknown values (including ``n/a``) return ``None`` so custom/classical
    models are excluded from size-aware decisions rather than failing a run.
    """
    match = _SIZE_RE.fullmatch(size or "")
    if match is None:
        return None
    value, suffix = match.groups()
    return round(float(value) * _SIZE_MULTIPLIERS[suffix.upper()])


def model_parameter_counts() -> dict[str, int]:
    """Build model-id → numeric parameter count from the built-in registry."""
    counts: dict[str, int] = {}
    for config in MODEL_REGISTRY.values():
        count = parse_parameter_count(config.size)
        if count is not None:
            counts[config.model_id] = count
    return counts


def comparison_pair_counts(comparisons: Iterable[object]) -> Counter[ModelPair]:
    """Count direct comparisons per model pair from results or stored rows."""
    counts: Counter[ModelPair] = Counter()
    for comparison in comparisons:
        if isinstance(comparison, Mapping):
            row = cast(Mapping[str, object], comparison)
            model_a = row.get("model_a")
            model_b = row.get("model_b")
        else:
            model_a = getattr(comparison, "model_a", None)
            model_b = getattr(comparison, "model_b", None)
        if isinstance(model_a, str) and isinstance(model_b, str):
            counts[normalize_model_pair(model_a, model_b)] += 1
    return counts


@dataclass(frozen=True)
class AdjacentPairDecision:
    """Stopping decision for one adjacent pair in the current ranking."""

    higher_model: str
    lower_model: str
    status: PairStatus
    direct_comparisons: int
    smaller_model: str | None = None
    larger_model: str | None = None
    size_ratio: float | None = None

    @property
    def pair(self) -> ModelPair:
        return normalize_model_pair(self.higher_model, self.lower_model)


def classify_adjacent_pairs(
    board: Leaderboard,
    pair_counts: Mapping[ModelPair, int],
    *,
    size_tie_ratio: float | None = None,
    size_tie_min_samples: int = 10,
    parameter_counts: Mapping[str, int] | None = None,
) -> list[AdjacentPairDecision]:
    """Classify adjacent ranks for adaptive sampling and practical preference.

    Non-overlapping marginal 95% CIs are statistically resolved. An overlapping
    pair may be marked ``prefer-smaller`` when the parameter ratio reaches the
    configured threshold and the pair has enough direct comparisons. This is a
    deployment preference, not an equivalence test or a change to ELO/rank.
    """
    sizes = parameter_counts if parameter_counts is not None else model_parameter_counts()
    decisions: list[AdjacentPairDecision] = []
    ranked = board.ranked

    for (higher_model, _), (lower_model, _) in zip(ranked, ranked[1:]):
        pair = normalize_model_pair(higher_model, lower_model)
        direct = pair_counts.get(pair, 0)
        higher_ci = board.elo_ci.get(higher_model)
        lower_ci = board.elo_ci.get(lower_model)

        if higher_ci is not None and lower_ci is not None and lower_ci[1] < higher_ci[0]:
            decisions.append(AdjacentPairDecision(higher_model, lower_model, "resolved", direct))
            continue

        if (
            higher_ci is not None
            and lower_ci is not None
            and size_tie_ratio is not None
            and direct >= size_tie_min_samples
        ):
            higher_size = sizes.get(higher_model)
            lower_size = sizes.get(lower_model)
            if higher_size and lower_size and higher_size != lower_size:
                smaller_model, larger_model = (
                    (higher_model, lower_model)
                    if higher_size < lower_size
                    else (lower_model, higher_model)
                )
                ratio = max(higher_size, lower_size) / min(higher_size, lower_size)
                if ratio >= size_tie_ratio:
                    decisions.append(
                        AdjacentPairDecision(
                            higher_model,
                            lower_model,
                            "prefer-smaller",
                            direct,
                            smaller_model=smaller_model,
                            larger_model=larger_model,
                            size_ratio=ratio,
                        )
                    )
                    continue

        decisions.append(AdjacentPairDecision(higher_model, lower_model, "unresolved", direct))

    return decisions


def unresolved_pairs(decisions: Iterable[AdjacentPairDecision]) -> set[ModelPair]:
    """Return adjacent pairs that should receive another targeted batch."""
    return {decision.pair for decision in decisions if decision.status == "unresolved"}


def practical_preferences(
    decisions: Iterable[AdjacentPairDecision],
) -> dict[str, list[AdjacentPairDecision]]:
    """Group size-aware practical preferences by the smaller model."""
    grouped: dict[str, list[AdjacentPairDecision]] = defaultdict(list)
    for decision in decisions:
        if decision.status == "prefer-smaller" and decision.smaller_model is not None:
            grouped[decision.smaller_model].append(decision)
    return dict(grouped)
