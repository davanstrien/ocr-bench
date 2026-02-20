"""Tests for ELO rating computation."""

from ocr_bench.elo import (
    INITIAL_ELO,
    ComparisonResult,
    Leaderboard,
    compute_elo,
    update_elo,
)


class TestUpdateElo:
    def test_winner_a_gains_rating(self):
        new_a, new_b = update_elo(1500.0, 1500.0, "A")
        assert new_a > 1500.0
        assert new_b < 1500.0

    def test_winner_b_gains_rating(self):
        new_a, new_b = update_elo(1500.0, 1500.0, "B")
        assert new_a < 1500.0
        assert new_b > 1500.0

    def test_tie_no_change_when_equal(self):
        new_a, new_b = update_elo(1500.0, 1500.0, "tie")
        assert new_a == 1500.0
        assert new_b == 1500.0

    def test_upset_larger_shift(self):
        """Beating a higher-rated opponent should yield a bigger gain."""
        # Expected win (strong beats weak)
        gain_expected_a, _ = update_elo(1600.0, 1400.0, "A")
        # Upset (weak beats strong)
        gain_upset_a, _ = update_elo(1400.0, 1600.0, "A")
        assert (gain_upset_a - 1400.0) > (gain_expected_a - 1600.0)

    def test_zero_sum(self):
        """Total ELO should be conserved."""
        new_a, new_b = update_elo(1500.0, 1500.0, "A")
        assert abs((new_a + new_b) - 3000.0) < 1e-10

    def test_zero_sum_unequal(self):
        new_a, new_b = update_elo(1600.0, 1400.0, "B")
        assert abs((new_a + new_b) - 3000.0) < 1e-10


class TestComputeElo:
    def test_single_comparison(self):
        results = [
            ComparisonResult(sample_idx=0, model_a="alpha", model_b="beta", winner="A"),
        ]
        board = compute_elo(results, ["alpha", "beta"])
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
        board = compute_elo(results, ["alpha", "beta"])
        # After unswap, B won, so beta should gain
        assert board.elo["beta"] > INITIAL_ELO
        assert board.elo["alpha"] < INITIAL_ELO
        assert board.wins["beta"] == 1
        assert board.losses["alpha"] == 1

    def test_tie_tracking(self):
        results = [
            ComparisonResult(sample_idx=0, model_a="alpha", model_b="beta", winner="tie"),
        ]
        board = compute_elo(results, ["alpha", "beta"])
        assert board.ties["alpha"] == 1
        assert board.ties["beta"] == 1

    def test_comparison_log_records_unswapped(self):
        results = [
            ComparisonResult(
                sample_idx=0, model_a="alpha", model_b="beta", winner="B", swapped=True
            ),
        ]
        board = compute_elo(results, ["alpha", "beta"])
        # After unswap, B→A
        assert board.comparison_log[0]["winner"] == "A"

    def test_multiple_models(self):
        results = [
            ComparisonResult(sample_idx=0, model_a="a", model_b="b", winner="A"),
            ComparisonResult(sample_idx=0, model_a="a", model_b="c", winner="A"),
            ComparisonResult(sample_idx=0, model_a="b", model_b="c", winner="B"),
        ]
        board = compute_elo(results, ["a", "b", "c"])
        ranked = board.ranked
        # "a" beat both, should be #1
        assert ranked[0][0] == "a"
        # "b" lost to both (winner="B" in match 3 means model_b "c" won), should be last
        assert ranked[-1][0] == "b"

    def test_elo_conserved(self):
        """Total ELO across all models stays at n * initial."""
        results = [
            ComparisonResult(sample_idx=0, model_a="a", model_b="b", winner="A"),
            ComparisonResult(sample_idx=1, model_a="b", model_b="c", winner="A"),
            ComparisonResult(sample_idx=2, model_a="a", model_b="c", winner="tie"),
        ]
        board = compute_elo(results, ["a", "b", "c"])
        total = sum(board.elo.values())
        assert abs(total - 3 * INITIAL_ELO) < 1e-10


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
