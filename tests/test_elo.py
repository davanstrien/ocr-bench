"""Tests for Bradley-Terry MLE rating computation."""

import random

from ocr_bench.elo import (
    INITIAL_ELO,
    ComparisonResult,
    Leaderboard,
    compute_elo,
)


class TestComputeElo:
    def test_single_comparison(self):
        results = [
            ComparisonResult(sample_idx=0, model_a="alpha", model_b="beta", winner="A"),
        ]
        board = compute_elo(results, ["alpha", "beta"], n_bootstrap=0)
        assert board.elo["alpha"] > INITIAL_ELO
        assert board.elo["beta"] < INITIAL_ELO
        assert board.wins["alpha"] == 1
        assert board.losses["beta"] == 1

    def test_swapped_comparison_unswaps(self):
        """If swapped=True, winner="A" means model_b actually won."""
        results = [
            ComparisonResult(
                sample_idx=0, model_a="alpha", model_b="beta", winner="A", swapped=True
            ),
        ]
        board = compute_elo(results, ["alpha", "beta"], n_bootstrap=0)
        # After unswap, B won, so beta should gain
        assert board.elo["beta"] > INITIAL_ELO
        assert board.elo["alpha"] < INITIAL_ELO
        assert board.wins["beta"] == 1
        assert board.losses["alpha"] == 1

    def test_tie_tracking(self):
        results = [
            ComparisonResult(sample_idx=0, model_a="alpha", model_b="beta", winner="tie"),
        ]
        board = compute_elo(results, ["alpha", "beta"], n_bootstrap=0)
        assert board.ties["alpha"] == 1
        assert board.ties["beta"] == 1

    def test_comparison_log_records_unswapped(self):
        results = [
            ComparisonResult(
                sample_idx=0, model_a="alpha", model_b="beta", winner="B", swapped=True
            ),
        ]
        board = compute_elo(results, ["alpha", "beta"], n_bootstrap=0)
        # After unswap, B→A
        assert board.comparison_log[0]["winner"] == "A"

    def test_comparison_log_includes_text_and_col_fields(self):
        results = [
            ComparisonResult(
                sample_idx=0,
                model_a="alpha",
                model_b="beta",
                winner="A",
                text_a="ocr output A",
                text_b="ocr output B",
                col_a="col_alpha",
                col_b="col_beta",
            ),
        ]
        board = compute_elo(results, ["alpha", "beta"], n_bootstrap=0)
        log = board.comparison_log[0]
        assert log["text_a"] == "ocr output A"
        assert log["text_b"] == "ocr output B"
        assert log["col_a"] == "col_alpha"
        assert log["col_b"] == "col_beta"

    def test_comparison_log_text_fields_default_empty(self):
        results = [
            ComparisonResult(sample_idx=0, model_a="alpha", model_b="beta", winner="tie"),
        ]
        board = compute_elo(results, ["alpha", "beta"], n_bootstrap=0)
        log = board.comparison_log[0]
        assert log["text_a"] == ""
        assert log["text_b"] == ""
        assert log["col_a"] == ""
        assert log["col_b"] == ""

    def test_multiple_models(self):
        results = [
            ComparisonResult(sample_idx=0, model_a="a", model_b="b", winner="A"),
            ComparisonResult(sample_idx=0, model_a="a", model_b="c", winner="A"),
            ComparisonResult(sample_idx=0, model_a="b", model_b="c", winner="B"),
        ]
        board = compute_elo(results, ["a", "b", "c"], n_bootstrap=0)
        ranked = board.ranked
        # "a" beat both, should be #1
        assert ranked[0][0] == "a"
        # "b" lost to both (winner="B" in match 3 means model_b "c" won), should be last
        assert ranked[-1][0] == "b"


class TestBradleyTerryProperties:
    def test_order_independent(self):
        """BT-MLE produces the same ratings regardless of comparison order."""
        results = [
            ComparisonResult(sample_idx=i, model_a="a", model_b="b", winner="A")
            for i in range(5)
        ] + [
            ComparisonResult(sample_idx=i, model_a="a", model_b="c", winner="A")
            for i in range(5, 10)
        ] + [
            ComparisonResult(sample_idx=i, model_a="b", model_b="c", winner="A")
            for i in range(10, 15)
        ]

        board1 = compute_elo(results, ["a", "b", "c"], n_bootstrap=0)

        shuffled = results.copy()
        random.Random(99).shuffle(shuffled)
        board2 = compute_elo(shuffled, ["a", "b", "c"], n_bootstrap=0)

        for model in ["a", "b", "c"]:
            assert abs(board1.elo[model] - board2.elo[model]) < 0.01, (
                f"ELO for {model} differs: {board1.elo[model]} vs {board2.elo[model]}"
            )

    def test_clear_winner_highest_rating(self):
        """Model that wins all matches gets highest rating."""
        results = [
            ComparisonResult(sample_idx=0, model_a="champ", model_b="mid", winner="A"),
            ComparisonResult(sample_idx=1, model_a="champ", model_b="weak", winner="A"),
            ComparisonResult(sample_idx=2, model_a="mid", model_b="weak", winner="A"),
        ]
        board = compute_elo(results, ["champ", "mid", "weak"], n_bootstrap=0)
        ranked = board.ranked
        assert ranked[0][0] == "champ"
        assert ranked[1][0] == "mid"
        assert ranked[2][0] == "weak"

    def test_ties_handled(self):
        """All ties should produce equal ratings centered at 1500."""
        results = [
            ComparisonResult(sample_idx=0, model_a="x", model_b="y", winner="tie"),
            ComparisonResult(sample_idx=1, model_a="x", model_b="z", winner="tie"),
            ComparisonResult(sample_idx=2, model_a="y", model_b="z", winner="tie"),
        ]
        board = compute_elo(results, ["x", "y", "z"], n_bootstrap=0)
        for model in ["x", "y", "z"]:
            assert abs(board.elo[model] - 1500.0) < 1.0, (
                f"{model} ELO {board.elo[model]} not near 1500"
            )

    def test_elo_centered(self):
        """Average ELO across all models should be close to 1500."""
        results = [
            ComparisonResult(sample_idx=0, model_a="a", model_b="b", winner="A"),
            ComparisonResult(sample_idx=1, model_a="b", model_b="c", winner="A"),
            ComparisonResult(sample_idx=2, model_a="a", model_b="c", winner="tie"),
        ]
        board = compute_elo(results, ["a", "b", "c"], n_bootstrap=0)
        avg = sum(board.elo.values()) / len(board.elo)
        assert abs(avg - 1500.0) < 1.0


class TestBootstrapCI:
    def test_ci_returns_intervals(self):
        """Bootstrap should return CI for every model with lower < elo < upper."""
        results = [
            ComparisonResult(sample_idx=i, model_a="a", model_b="b", winner="A")
            for i in range(10)
        ] + [
            ComparisonResult(sample_idx=i, model_a="b", model_b="c", winner="A")
            for i in range(10, 20)
        ]
        board = compute_elo(results, ["a", "b", "c"], n_bootstrap=200)

        assert board.elo_ci, "elo_ci should not be empty"
        for model in ["a", "b", "c"]:
            assert model in board.elo_ci
            lo, hi = board.elo_ci[model]
            assert lo <= board.elo[model] + 5  # small tolerance
            assert hi >= board.elo[model] - 5

    def test_ci_disabled_when_zero(self):
        """n_bootstrap=0 should produce empty elo_ci."""
        results = [
            ComparisonResult(sample_idx=0, model_a="a", model_b="b", winner="A"),
        ]
        board = compute_elo(results, ["a", "b"], n_bootstrap=0)
        assert board.elo_ci == {}


class TestLeaderboard:
    def test_win_pct(self):
        board = Leaderboard(
            elo={"a": 1600.0},
            wins={"a": 3},
            losses={"a": 1},
            ties={"a": 1},
        )
        assert board.win_pct("a") == 60.0

    def test_win_pct_no_games(self):
        board = Leaderboard(
            elo={"a": 1500.0},
            wins={"a": 0},
            losses={"a": 0},
            ties={"a": 0},
        )
        assert board.win_pct("a") is None

    def test_ranked_order(self):
        board = Leaderboard(elo={"a": 1400.0, "b": 1600.0, "c": 1500.0})
        assert [m for m, _ in board.ranked] == ["b", "c", "a"]

    def test_elo_ci_default_empty(self):
        board = Leaderboard()
        assert board.elo_ci == {}
