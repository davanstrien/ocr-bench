"""Bradley-Terry MLE rating computation for pairwise comparisons."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy.optimize import minimize

INITIAL_ELO: float = 1500.0

Winner = Literal["A", "B", "tie"]


@dataclass
class ComparisonResult:
    """Result of a single pairwise comparison, ready for ELO computation."""

    sample_idx: int
    model_a: str
    model_b: str
    winner: Winner
    reason: str = ""
    agreement: str = "1/1"
    swapped: bool = False
    text_a: str = ""
    text_b: str = ""
    col_a: str = ""
    col_b: str = ""


@dataclass
class Leaderboard:
    """ELO leaderboard computed from pairwise comparison results."""

    elo: dict[str, float] = field(default_factory=dict)
    wins: dict[str, int] = field(default_factory=dict)
    losses: dict[str, int] = field(default_factory=dict)
    ties: dict[str, int] = field(default_factory=dict)
    comparison_log: list[dict[str, object]] = field(default_factory=list)
    elo_ci: dict[str, tuple[float, float]] = field(default_factory=dict)

    @property
    def ranked(self) -> list[tuple[str, float]]:
        """Models sorted by ELO rating, descending."""
        return sorted(self.elo.items(), key=lambda x: x[1], reverse=True)

    def win_pct(self, model: str) -> float | None:
        """Win percentage for a model, or None if no comparisons."""
        total = self.wins[model] + self.losses[model] + self.ties[model]
        if total == 0:
            return None
        return self.wins[model] / total * 100


def _unswap_winner(winner: Winner, swapped: bool) -> Winner:
    """Unswap winner if positions were randomized."""
    if swapped:
        if winner == "A":
            return "B"
        elif winner == "B":
            return "A"
    return winner


def _build_win_matrix(
    results: list[ComparisonResult],
) -> tuple[dict[tuple[str, str], float], set[str]]:
    """Count wins per ordered pair. Ties count as 0.5 for each side.

    Returns (win_counts, models_seen) where win_counts[(i, j)] = fractional
    wins of i over j.
    """
    win_counts: dict[tuple[str, str], float] = defaultdict(float)
    models_seen: set[str] = set()

    for r in results:
        winner = _unswap_winner(r.winner, r.swapped)
        models_seen.add(r.model_a)
        models_seen.add(r.model_b)

        if winner == "A":
            win_counts[(r.model_a, r.model_b)] += 1.0
        elif winner == "B":
            win_counts[(r.model_b, r.model_a)] += 1.0
        else:
            win_counts[(r.model_a, r.model_b)] += 0.5
            win_counts[(r.model_b, r.model_a)] += 0.5

    return win_counts, models_seen


def _bt_mle(
    win_counts: dict[tuple[str, str], float],
    model_names: list[str],
) -> dict[str, float]:
    """Fit Bradley-Terry model via maximum likelihood estimation.

    Returns theta (strength) per model. Uses scipy L-BFGS-B on the
    negative log-likelihood with log-parameterization for positivity.
    """
    n = len(model_names)
    if n == 0:
        return {}
    if n == 1:
        return {model_names[0]: 1.0}

    idx = {name: i for i, name in enumerate(model_names)}

    # Collect all pairs with nonzero games
    pairs: list[tuple[int, int, float, float]] = []
    for i_name in model_names:
        for j_name in model_names:
            if i_name >= j_name:
                continue
            w_ij = win_counts.get((i_name, j_name), 0.0)
            w_ji = win_counts.get((j_name, i_name), 0.0)
            if w_ij + w_ji > 0:
                pairs.append((idx[i_name], idx[j_name], w_ij, w_ji))

    if not pairs:
        return {name: 1.0 for name in model_names}

    def neg_log_likelihood(log_theta: np.ndarray) -> float:
        nll = 0.0
        for i, j, w_ij, w_ji in pairs:
            diff = log_theta[i] - log_theta[j]
            # log(theta_i / (theta_i + theta_j)) = diff - log(1 + exp(diff))
            # log(theta_j / (theta_i + theta_j)) = -diff - log(1 + exp(-diff))
            # Use log-sum-exp for numerical stability
            log_p_ij = diff - np.logaddexp(0.0, diff)
            log_p_ji = -diff - np.logaddexp(0.0, -diff)
            nll -= w_ij * log_p_ij + w_ji * log_p_ji
        return nll

    def gradient(log_theta: np.ndarray) -> np.ndarray:
        grad = np.zeros(n)
        for i, j, w_ij, w_ji in pairs:
            diff = log_theta[i] - log_theta[j]
            p_ij = 1.0 / (1.0 + np.exp(-diff))  # sigmoid(diff)
            total = w_ij + w_ji
            # d(NLL)/d(log_theta_i)
            grad[i] -= w_ij - total * p_ij
            grad[j] -= w_ji - total * (1.0 - p_ij)
        return grad

    # Pin first model at 0 to fix the scale
    x0 = np.zeros(n)
    result = minimize(
        neg_log_likelihood,
        x0,
        jac=gradient,
        method="L-BFGS-B",
    )

    log_theta = result.x
    # Center: subtract geometric mean (= mean of log_theta)
    log_theta -= log_theta.mean()
    theta = np.exp(log_theta)

    return {name: float(theta[idx[name]]) for name in model_names}


def _theta_to_elo(theta: dict[str, float], center: float = 1500.0) -> dict[str, float]:
    """Convert BT theta values to ELO scale.

    ELO_i = 400 * log10(theta_i / theta_ref) + center
    where theta_ref is the geometric mean of all theta values.
    """
    if not theta:
        return {}

    values = list(theta.values())
    log_geo_mean = sum(math.log(v) for v in values) / len(values)
    geo_mean = math.exp(log_geo_mean)

    return {
        name: 400.0 * math.log10(t / geo_mean) + center
        for name, t in theta.items()
    }


def _bootstrap_ci(
    results: list[ComparisonResult],
    model_names: list[str],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict[str, tuple[float, float]]:
    """Compute bootstrap confidence intervals for ELO ratings.

    Resamples comparisons with replacement, fits BT-MLE each time,
    returns percentile-based CIs.
    """
    if not results or not model_names:
        return {}

    rng = random.Random(seed)
    n = len(results)
    elo_samples: dict[str, list[float]] = {name: [] for name in model_names}

    for _ in range(n_bootstrap):
        boot = rng.choices(results, k=n)
        win_counts, _ = _build_win_matrix(boot)
        theta = _bt_mle(win_counts, model_names)
        elos = _theta_to_elo(theta)
        for name in model_names:
            elo_samples[name].append(elos.get(name, 1500.0))

    alpha = (1.0 - ci) / 2.0
    lo_pct = alpha * 100
    hi_pct = (1.0 - alpha) * 100

    cis: dict[str, tuple[float, float]] = {}
    for name in model_names:
        samples = sorted(elo_samples[name])
        lo_idx = int(len(samples) * lo_pct / 100)
        hi_idx = min(int(len(samples) * hi_pct / 100), len(samples) - 1)
        cis[name] = (samples[lo_idx], samples[hi_idx])

    return cis


def rankings_resolved(board: Leaderboard) -> bool:
    """Check if all adjacent ranks have non-overlapping 95% CIs.

    Returns True when the ranking order is statistically resolved — i.e. for
    every pair of adjacent models in the ranking, the higher-ranked model's
    CI lower bound exceeds the lower-ranked model's CI upper bound.
    """
    if not board.elo_ci:
        return False
    ranked = board.ranked
    if len(ranked) < 2:
        return False
    for i in range(len(ranked) - 1):
        model_hi, _ = ranked[i]
        model_lo, _ = ranked[i + 1]
        if model_hi not in board.elo_ci or model_lo not in board.elo_ci:
            return False
        lo_of_higher, _ = board.elo_ci[model_hi]
        _, hi_of_lower = board.elo_ci[model_lo]
        if hi_of_lower >= lo_of_higher:
            return False  # CIs overlap
    return True


def compute_elo(
    results: list[ComparisonResult],
    model_names: list[str],
    n_bootstrap: int = 1000,
) -> Leaderboard:
    """Compute ELO ratings from pairwise comparison results using Bradley-Terry MLE.

    Handles position-bias unswapping: if a result has swapped=True,
    the winner is flipped before updating ratings.

    Bootstrap confidence intervals are computed when n_bootstrap > 0.
    """
    board = Leaderboard(
        elo={m: INITIAL_ELO for m in model_names},
        wins={m: 0 for m in model_names},
        losses={m: 0 for m in model_names},
        ties={m: 0 for m in model_names},
    )

    # Tally wins/losses/ties and build comparison log
    for r in results:
        winner = _unswap_winner(r.winner, r.swapped)

        if winner == "A":
            board.wins[r.model_a] += 1
            board.losses[r.model_b] += 1
        elif winner == "B":
            board.losses[r.model_a] += 1
            board.wins[r.model_b] += 1
        else:
            board.ties[r.model_a] += 1
            board.ties[r.model_b] += 1

        board.comparison_log.append(
            {
                "sample_idx": r.sample_idx,
                "model_a": r.model_a,
                "model_b": r.model_b,
                "winner": winner,
                "reason": r.reason,
                "agreement": r.agreement,
                "text_a": r.text_a,
                "text_b": r.text_b,
                "col_a": r.col_a,
                "col_b": r.col_b,
            }
        )

    # Fit BT-MLE
    win_counts, _ = _build_win_matrix(results)
    theta = _bt_mle(win_counts, model_names)
    board.elo = _theta_to_elo(theta)

    # Bootstrap CIs
    if n_bootstrap > 0 and results:
        board.elo_ci = _bootstrap_ci(results, model_names, n_bootstrap=n_bootstrap)

    return board
