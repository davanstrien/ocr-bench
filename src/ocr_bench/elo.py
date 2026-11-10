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
ShownFirst = Literal["A", "B", ""]


# A page-level bootstrap has few distinct resamples on a handful of pages, and
# identical pages give zero-width intervals. Do not call a ranking resolved below this.
MIN_PAGES_TO_RESOLVE = 10

# The paired bootstrap test behind the rank ranges calls two equally good models
# apart too often on few pages: in simulation about 9% of the time at 10 pages,
# 7% at 20 and 6% at 30 and more, against the 5% it is set to. A model on fewer
# pages than this is not called apart from any model.
MIN_PAGES_FOR_RANK_RANGE = 20

# A percentile bound from a few replicates is close to the smallest one, so the
# rank ranges are left out below this number of replicates.
MIN_REPLICATES_FOR_RANK_RANGE = 100


@dataclass
class ComparisonResult:
    """Result of a single pairwise comparison, ready for ELO computation."""

    sample_idx: int
    model_a: str
    model_b: str
    winner: Winner
    reason: str = ""
    agreement: str = "1/1"
    # In-memory only: True while ``winner`` is still in the order the judge saw
    # (Output A = model_b). Stored rows hold the un-swapped winner, so a result
    # loaded from the Hub always has ``swapped=False``.
    swapped: bool = False
    # Which model's output the judge saw first, as "Output A": "A" (model_a) or
    # "B" (model_b), the same letters as ``winner``, so ``winner == shown_first``
    # means the first output shown won. "" when no judge saw the pair (an
    # auto-tie) or the row predates this field. Stored with the comparison;
    # never used to un-swap.
    shown_first: ShownFirst = ""
    text_a: str = ""
    text_b: str = ""
    col_a: str = ""
    col_b: str = ""
    # Whether the judge read a cut version of the output. None on a stored row
    # that does not record it.
    truncated_a: bool | None = False
    truncated_b: bool | None = False
    # Hash of the source/judge/sampling provenance for safe incremental resume.
    # Historical rows lack it and must not be silently reused.
    provenance_hash: str = ""


@dataclass
class Leaderboard:
    """ELO leaderboard computed from pairwise comparison results."""

    elo: dict[str, float] = field(default_factory=dict)
    wins: dict[str, int] = field(default_factory=dict)
    losses: dict[str, int] = field(default_factory=dict)
    ties: dict[str, int] = field(default_factory=dict)
    comparison_log: list[dict[str, object]] = field(default_factory=list)
    elo_ci: dict[str, tuple[float, float]] = field(default_factory=dict)
    pages_by_model: dict[str, int] = field(default_factory=dict)
    # model → (best rank, worst rank) it could hold, from the paired bootstrap
    # of rating differences (see ``_rank_ranges``). Empty without a bootstrap.
    rank_range: dict[str, tuple[int, int]] = field(default_factory=dict)

    def has_enough_pages_to_resolve(self, model: str) -> bool:
        """False when a model's interval rests on too few pages to stop on.

        Counts are per model, because targeted judging can give one model far
        fewer pages than the rest of the board. Boards with no recorded page
        counts are not gated.
        """
        if not self.pages_by_model:
            return True
        return self.pages_by_model.get(model, 0) >= MIN_PAGES_TO_RESOLVE

    @property
    def ranked(self) -> list[tuple[str, float]]:
        """Models sorted by ELO descending, with model name as a stable tie-break."""
        return sorted(self.elo.items(), key=lambda item: (-item[1], item[0]))

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

    return {name: 400.0 * math.log10(t / geo_mean) + center for name, t in theta.items()}


def _bootstrap_replicates(
    results: list[ComparisonResult],
    model_names: list[str],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> list[dict[str, float]]:
    """Fit the ratings on ``n_bootstrap`` resamples of the pages.

    Resamples whole pages with replacement and fits BT-MLE each time.
    Comparisons from one page share that page's image and difficulty, so a page
    is the independent unit. Each replicate maps the models it holds to their
    ratings, shifted onto the scale of the fit on all pages.
    """
    if not results or not model_names:
        return []

    rng = random.Random(seed)
    results_by_page: dict[int, list[ComparisonResult]] = {}
    for result in results:
        results_by_page.setdefault(result.sample_idx, []).append(result)
    pages = [results_by_page[page_idx] for page_idx in sorted(results_by_page)]
    full_win_counts, _ = _build_win_matrix(results)
    full_elos = _theta_to_elo(_bt_mle(full_win_counts, model_names))

    replicates: list[dict[str, float]] = []
    for _ in range(n_bootstrap):
        boot = [result for page in rng.choices(pages, k=len(pages)) for result in page]
        # A model absent from every resampled page has no evidence in this
        # replicate, so it gets no rating rather than an arbitrary one.
        models_seen = {name for result in boot for name in (result.model_a, result.model_b)}
        replicate_models = [name for name in model_names if name in models_seen]
        win_counts, _ = _build_win_matrix(boot)
        theta = _bt_mle(win_counts, replicate_models)
        elos = _theta_to_elo(theta)
        # Each fit is centred on its own models. Without this shift, a replicate
        # that omits a model sits on a different scale from the others.
        scale_shift = sum(full_elos[name] - elos[name] for name in replicate_models) / len(
            replicate_models
        )
        replicates.append({name: elos[name] + scale_shift for name in replicate_models})
    return replicates


def _percentile_bounds(samples: list[float], ci: float = 0.95) -> tuple[float, float]:
    """The lower and upper percentile of ``samples`` for a ``ci`` interval."""
    ordered = sorted(samples)
    alpha = (1.0 - ci) / 2.0
    lo_pct = alpha * 100
    hi_pct = (1.0 - alpha) * 100
    lo_idx = int(len(ordered) * lo_pct / 100)
    hi_idx = min(int(len(ordered) * hi_pct / 100), len(ordered) - 1)
    return ordered[lo_idx], ordered[hi_idx]


def _intervals(
    replicates: list[dict[str, float]], model_names: list[str], ci: float = 0.95
) -> dict[str, tuple[float, float]]:
    """Percentile interval of each model's rating over the bootstrap replicates."""
    cis: dict[str, tuple[float, float]] = {}
    for name in model_names:
        samples = [replicate[name] for replicate in replicates if name in replicate]
        if samples:
            cis[name] = _percentile_bounds(samples, ci)
    return cis


def _bootstrap_ci(
    results: list[ComparisonResult],
    model_names: list[str],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict[str, tuple[float, float]]:
    """Compute bootstrap confidence intervals for ELO ratings.

    Percentile intervals over page-level resamples (see
    :func:`_bootstrap_replicates`).
    """
    replicates = _bootstrap_replicates(results, model_names, n_bootstrap=n_bootstrap, seed=seed)
    return _intervals(replicates, model_names, ci)


def _clearly_above(
    higher: str, lower: str, replicates: list[dict[str, float]], ci: float = 0.95
) -> bool:
    """Whether ``higher`` is rated above ``lower`` on this data, beyond resampling noise.

    Both ratings come from the same resampled pages in each replicate, so their
    difference varies less than the two ratings do on their own: a page that is
    hard for one model is often hard for the other. The difference is clear
    when its percentile interval lies above zero. Two models whose own
    intervals overlap can still be clearly apart.
    """
    differences = [
        replicate[higher] - replicate[lower]
        for replicate in replicates
        if higher in replicate and lower in replicate
    ]
    if not differences:
        return False
    low, _ = _percentile_bounds(differences, ci)
    return low > 0


def _rank_ranges(
    elo: dict[str, float],
    replicates: list[dict[str, float]],
    pages_by_model: dict[str, int],
    ci: float = 0.95,
) -> dict[str, tuple[int, int]]:
    """The best and worst rank each model could hold.

    A model's best rank is one more than the number of models clearly above it;
    its worst rank is the number of models minus those clearly below it (see
    :func:`_clearly_above`). A range of one rank means the model is clearly
    apart from every other model. The range says how many models are clearly
    above and below, not which: two models with overlapping ranges can be
    clearly apart.

    A model on fewer than ``MIN_PAGES_FOR_RANK_RANGE`` pages is not called
    clearly apart from any model, so it widens every other range by one rank.
    With fewer than ``MIN_REPLICATES_FOR_RANK_RANGE`` replicates there are no
    ranges.

    The tests are one per pair at the 95% level, with no correction for the
    number of pairs: about 1 pair in 20 of equally good models is called apart
    by chance, so a large board is likely to hold such a pair. The range still
    holds a model's true rank far more often than 95% of the time.
    """
    if len(replicates) < MIN_REPLICATES_FOR_RANK_RANGE:
        return {}
    names = sorted(elo, key=lambda name: (-elo[name], name))
    enough_pages = {name: pages_by_model.get(name, 0) >= MIN_PAGES_FOR_RANK_RANGE for name in names}
    ranges: dict[str, tuple[int, int]] = {}
    for position, name in enumerate(names, start=1):
        clearly_above_it = 0
        clearly_below_it = 0
        for other in names:
            if other == name or not (enough_pages[name] and enough_pages[other]):
                continue
            if _clearly_above(other, name, replicates, ci):
                clearly_above_it += 1
            elif _clearly_above(name, other, replicates, ci):
                clearly_below_it += 1
        best = min(1 + clearly_above_it, position)
        worst = max(len(names) - clearly_below_it, position)
        ranges[name] = (best, worst)
    return ranges


def rankings_resolved(board: Leaderboard) -> bool:
    """Check if all adjacent ranks have non-overlapping 95% CIs.

    Retained as a public statistical-only helper; the CLI uses the richer
    adaptive classifier when optional practical stopping rules are enabled.

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
        if not (
            board.has_enough_pages_to_resolve(model_hi)
            and board.has_enough_pages_to_resolve(model_lo)
        ):
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

        # Canonicalise the reason text so A/B references match model_a/model_b
        reason = r.reason
        if r.swapped and reason:
            # Swap "Output A"↔"Output B" (and bare A/B) so the stored reason
            # uses A/B consistently with model_a/model_b ordering.
            reason = (
                reason.replace("Output A", "Output __X__")
                .replace("Output B", "Output A")
                .replace("Output __X__", "Output B")
            )

        board.comparison_log.append(
            {
                "sample_idx": r.sample_idx,
                "model_a": r.model_a,
                "model_b": r.model_b,
                "winner": winner,
                "reason": reason,
                "agreement": r.agreement,
                "shown_first": r.shown_first,
                "text_a": r.text_a,
                "text_b": r.text_b,
                "col_a": r.col_a,
                "col_b": r.col_b,
                # Keyed to model_a/model_b (like text_a/text_b), so no swap
                # flip: truncated_a is always about model_a's output.
                "truncated_a": r.truncated_a,
                "truncated_b": r.truncated_b,
                "provenance_hash": r.provenance_hash,
            }
        )

    # Fit BT-MLE
    win_counts, _ = _build_win_matrix(results)
    theta = _bt_mle(win_counts, model_names)
    board.elo = _theta_to_elo(theta)

    # Bootstrap CIs
    if n_bootstrap > 0 and results:
        replicates = _bootstrap_replicates(results, model_names, n_bootstrap=n_bootstrap)
        board.elo_ci = _intervals(replicates, model_names)
        pages_seen: dict[str, set[int]] = {name: set() for name in model_names}
        for result in results:
            pages_seen.setdefault(result.model_a, set()).add(result.sample_idx)
            pages_seen.setdefault(result.model_b, set()).add(result.sample_idx)
        board.pages_by_model = {name: len(pages) for name, pages in pages_seen.items()}
        board.rank_range = _rank_ranges(board.elo, replicates, board.pages_by_model)

    return board
