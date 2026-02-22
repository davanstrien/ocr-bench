"""Bradley-Terry ELO rating computation for pairwise comparisons."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

INITIAL_ELO: float = 1500.0
K: float = 32.0

Winner = Literal["A", "B", "tie"]


def update_elo(elo_a: float, elo_b: float, winner: Winner) -> tuple[float, float]:
    """Update ELO ratings for a single pairwise comparison.

    Bradley-Terry model: expected score = 1 / (1 + 10^((elo_b - elo_a) / 400)).
    """
    expected_a = 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / 400.0))
    if winner == "A":
        score_a = 1.0
    elif winner == "B":
        score_a = 0.0
    else:
        score_a = 0.5
    new_a = elo_a + K * (score_a - expected_a)
    new_b = elo_b + K * ((1.0 - score_a) - (1.0 - expected_a))
    return new_a, new_b


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


def compute_elo(
    results: list[ComparisonResult],
    model_names: list[str],
    initial_elo: float = INITIAL_ELO,
) -> Leaderboard:
    """Compute ELO ratings from pairwise comparison results.

    Handles position-bias unswapping: if a result has swapped=True,
    the winner is flipped before updating ratings.
    """
    board = Leaderboard(
        elo={m: initial_elo for m in model_names},
        wins={m: 0 for m in model_names},
        losses={m: 0 for m in model_names},
        ties={m: 0 for m in model_names},
    )

    for r in results:
        # Unswap if positions were randomized
        winner = r.winner
        if r.swapped:
            if winner == "A":
                winner = "B"
            elif winner == "B":
                winner = "A"

        if winner == "A":
            board.elo[r.model_a], board.elo[r.model_b] = update_elo(
                board.elo[r.model_a], board.elo[r.model_b], "A"
            )
            board.wins[r.model_a] += 1
            board.losses[r.model_b] += 1
        elif winner == "B":
            board.elo[r.model_a], board.elo[r.model_b] = update_elo(
                board.elo[r.model_a], board.elo[r.model_b], "B"
            )
            board.losses[r.model_a] += 1
            board.wins[r.model_b] += 1
        else:
            board.elo[r.model_a], board.elo[r.model_b] = update_elo(
                board.elo[r.model_a], board.elo[r.model_b], "tie"
            )
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

    return board
