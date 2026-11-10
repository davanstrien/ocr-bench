"""Tests for Bradley-Terry MLE rating computation."""

import random

from ocr_bench.elo import (
    INITIAL_ELO,
    MIN_PAGES_FOR_RANK_RANGE,
    MIN_REPLICATES_FOR_RANK_RANGE,
    ComparisonResult,
    Leaderboard,
    _bootstrap_ci,
    _bootstrap_replicates,
    _clearly_above,
    _percentile_bounds,
    _rank_ranges,
    compute_elo,
    rankings_resolved,
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

    def test_comparison_log_records_display_order(self):
        results = [
            ComparisonResult(
                sample_idx=0,
                model_a="alpha",
                model_b="beta",
                winner="B",
                swapped=True,
                shown_first="B",
            ),
            ComparisonResult(sample_idx=1, model_a="alpha", model_b="beta", winner="tie"),
        ]
        board = compute_elo(results, ["alpha", "beta"], n_bootstrap=0)
        assert [row["shown_first"] for row in board.comparison_log] == ["B", ""]

    def test_stored_display_order_does_not_flip_a_reloaded_winner(self):
        # A row read back from the Hub holds the un-swapped winner and
        # swapped=False. Its display order is a record, not an instruction.
        reloaded = [
            ComparisonResult(
                sample_idx=0, model_a="alpha", model_b="beta", winner="A", shown_first="B"
            ),
        ]
        board = compute_elo(reloaded, ["alpha", "beta"], n_bootstrap=0)
        assert board.comparison_log[0]["winner"] == "A"
        assert board.comparison_log[0]["shown_first"] == "B"
        assert (board.wins["alpha"], board.wins["beta"]) == (1, 0)

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

    def test_comparison_log_records_truncation_flags(self):
        results = [
            ComparisonResult(
                sample_idx=0,
                model_a="alpha",
                model_b="beta",
                winner="A",
                truncated_a=True,
                truncated_b=False,
            ),
        ]
        board = compute_elo(results, ["alpha", "beta"], n_bootstrap=0)
        log = board.comparison_log[0]
        assert log["truncated_a"] is True
        assert log["truncated_b"] is False

    def test_comparison_log_truncation_flags_default_false(self):
        results = [
            ComparisonResult(sample_idx=0, model_a="alpha", model_b="beta", winner="tie"),
        ]
        board = compute_elo(results, ["alpha", "beta"], n_bootstrap=0)
        log = board.comparison_log[0]
        assert log["truncated_a"] is False
        assert log["truncated_b"] is False

    def test_comparison_log_truncation_flags_keyed_to_model_not_swap(self):
        # truncated_a is about model_a's output regardless of display swap —
        # it must NOT be flipped the way winner is.
        results = [
            ComparisonResult(
                sample_idx=0,
                model_a="alpha",
                model_b="beta",
                winner="B",
                swapped=True,
                truncated_a=True,
                truncated_b=False,
            ),
        ]
        board = compute_elo(results, ["alpha", "beta"], n_bootstrap=0)
        log = board.comparison_log[0]
        assert log["winner"] == "A"  # unswapped
        assert log["truncated_a"] is True  # still model_a, not flipped
        assert log["truncated_b"] is False

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
        results = (
            [ComparisonResult(sample_idx=i, model_a="a", model_b="b", winner="A") for i in range(5)]
            + [
                ComparisonResult(sample_idx=i, model_a="a", model_b="c", winner="A")
                for i in range(5, 10)
            ]
            + [
                ComparisonResult(sample_idx=i, model_a="b", model_b="c", winner="A")
                for i in range(10, 15)
            ]
        )

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
            ComparisonResult(sample_idx=i, model_a="a", model_b="b", winner="A") for i in range(10)
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

    def test_ci_resamples_whole_pages(self):
        """Comparisons from one page stay together, so page conflict widens the CI."""
        page_where_alpha_wins = [
            ComparisonResult(sample_idx=0, model_a="alpha", model_b="beta", winner="A")
            for _ in range(20)
        ]
        page_where_beta_wins = [
            ComparisonResult(sample_idx=1, model_a="alpha", model_b="beta", winner="B")
            for _ in range(20)
        ]
        board = compute_elo(
            page_where_alpha_wins + page_where_beta_wins,
            ["alpha", "beta"],
            n_bootstrap=200,
        )
        low, high = board.elo_ci["alpha"]
        assert low < 1400
        assert high > 1600

    def test_too_few_pages_never_resolve(self):
        identical_pages = [
            ComparisonResult(sample_idx=page, model_a="alpha", model_b="beta", winner="A")
            for page in range(5)
        ]
        board = compute_elo(identical_pages, ["alpha", "beta"], n_bootstrap=50)
        assert board.pages_by_model == {"alpha": 5, "beta": 5}
        assert not rankings_resolved(board)

    def test_enough_consistent_pages_can_resolve(self):
        pages = [
            ComparisonResult(sample_idx=page, model_a="alpha", model_b="beta", winner="A")
            for page in range(9)
        ] + [ComparisonResult(sample_idx=9, model_a="alpha", model_b="beta", winner="A")]
        board = compute_elo(pages, ["alpha", "beta"], n_bootstrap=50)
        board.elo_ci = {"alpha": (1600.0, 1700.0), "beta": (1300.0, 1400.0)}
        assert rankings_resolved(board)

    def test_model_on_few_pages_never_resolves(self):
        one_page_for_rare = [
            ComparisonResult(sample_idx=0, model_a="rare", model_b="alpha", winner="A"),
            ComparisonResult(sample_idx=0, model_a="rare", model_b="beta", winner="A"),
        ]
        many_pages = [
            ComparisonResult(sample_idx=page, model_a="alpha", model_b="beta", winner="A")
            for page in range(1, 31)
        ]
        board = compute_elo(
            one_page_for_rare + many_pages, ["alpha", "beta", "rare"], n_bootstrap=50
        )
        assert board.pages_by_model["rare"] == 1
        assert not rankings_resolved(board)

    def test_rare_model_does_not_inflate_other_intervals(self):
        def close_pair(page_offset: int = 0) -> list[ComparisonResult]:
            winners = ["A"] * 7 + ["B"] * 5
            return [
                ComparisonResult(
                    sample_idx=page + page_offset, model_a="alpha", model_b="beta", winner=winner
                )
                for page, winner in enumerate(winners)
            ]

        rare_pages = [
            ComparisonResult(sample_idx=20 + page, model_a="rare", model_b="alpha", winner="A")
            for page in range(2)
        ]
        with_rare = _bootstrap_ci(
            close_pair() + rare_pages, ["alpha", "beta", "rare"], n_bootstrap=200
        )
        board = compute_elo(close_pair() + rare_pages, ["alpha", "beta", "rare"], n_bootstrap=0)
        low, high = with_rare["alpha"]
        assert low <= board.elo["alpha"] <= high
        assert high < board.elo["alpha"] + 400

    def test_model_absent_from_replicate_gets_no_arbitrary_rating(self):
        warm_up = [
            ComparisonResult(sample_idx=0, model_a="rare", model_b="alpha", winner="A"),
        ]
        targeted = [
            ComparisonResult(sample_idx=page, model_a="alpha", model_b="beta", winner="A")
            for page in range(1, 12)
        ]
        cis = _bootstrap_ci(warm_up + targeted, ["alpha", "beta", "rare"], n_bootstrap=200)
        assert set(cis) == {"alpha", "beta", "rare"}
        assert cis["rare"][0] > 1500

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

    def test_equal_elo_uses_stable_model_name_tie_break(self):
        board = Leaderboard(elo={"z-model": 1500.0, "a-model": 1500.0})
        assert [model for model, _ in board.ranked] == ["a-model", "z-model"]

    def test_elo_ci_default_empty(self):
        board = Leaderboard()
        assert board.elo_ci == {}


class TestRankingsResolved:
    def test_non_overlapping_cis(self):
        """Non-overlapping CIs → resolved."""
        board = Leaderboard(
            elo={"a": 1600.0, "b": 1500.0, "c": 1400.0},
            elo_ci={
                "a": (1580.0, 1620.0),
                "b": (1480.0, 1520.0),
                "c": (1380.0, 1420.0),
            },
        )
        assert rankings_resolved(board) is True

    def test_overlapping_cis(self):
        """Overlapping CIs between adjacent ranks → not resolved."""
        board = Leaderboard(
            elo={"a": 1550.0, "b": 1500.0, "c": 1400.0},
            elo_ci={
                "a": (1490.0, 1610.0),  # overlaps with b's upper
                "b": (1480.0, 1520.0),
                "c": (1380.0, 1420.0),
            },
        )
        assert rankings_resolved(board) is False

    def test_no_cis(self):
        """No CIs → not resolved."""
        board = Leaderboard(elo={"a": 1600.0, "b": 1500.0})
        assert rankings_resolved(board) is False

    def test_single_model(self):
        """Single model → not resolved (fewer than 2 models)."""
        board = Leaderboard(
            elo={"a": 1500.0},
            elo_ci={"a": (1480.0, 1520.0)},
        )
        assert rankings_resolved(board) is False

    def test_missing_ci_for_model(self):
        """CI missing for one model → not resolved."""
        board = Leaderboard(
            elo={"a": 1600.0, "b": 1500.0},
            elo_ci={"a": (1580.0, 1620.0)},  # b missing
        )
        assert rankings_resolved(board) is False

    def test_exact_boundary(self):
        """CIs touching exactly (hi_of_lower == lo_of_higher) → not resolved."""
        board = Leaderboard(
            elo={"a": 1600.0, "b": 1500.0},
            elo_ci={
                "a": (1550.0, 1650.0),
                "b": (1450.0, 1550.0),  # upper == lower of a
            },
        )
        assert rankings_resolved(board) is False


class TestRankRanges:
    """The ranks a model could hold, from the paired bootstrap of rating differences."""

    @staticmethod
    def _results(pages, win_rule):
        """One comparison per pair per page; ``win_rule(page, a, b)`` names the winner."""
        models = ["strong", "middle", "weak"]
        results = []
        for page in range(pages):
            for i, a in enumerate(models):
                for b in models[i + 1 :]:
                    results.append(ComparisonResult(page, a, b, win_rule(page, a, b)))
        return models, results

    def test_clearly_different_models_hold_one_rank_each(self):
        # The model listed first wins 9 pages in 10.
        models, results = self._results(60, lambda page, a, b: "B" if page % 10 == 0 else "A")
        board = compute_elo(results, models, n_bootstrap=200)
        assert board.rank_range == {"strong": (1, 1), "middle": (2, 2), "weak": (3, 3)}

    def test_models_the_data_does_not_separate_share_a_range(self):
        def win_rule(page, a, b):
            if (a, b) == ("strong", "middle"):
                return "A" if page % 2 == 0 else "B"  # an even split
            return "B" if page % 10 == 0 else "A"

        models, results = self._results(60, win_rule)
        board = compute_elo(results, models, n_bootstrap=200)
        assert board.rank_range["strong"] == (1, 2)
        assert board.rank_range["middle"] == (1, 2)
        assert board.rank_range["weak"] == (3, 3)

    def test_the_range_holds_the_rank_from_the_fit(self):
        rng = random.Random(3)
        models = [f"m{n}" for n in range(6)]
        strength = {m: 1.0 + 0.15 * n for n, m in enumerate(models)}
        results = []
        for page in range(40):
            for i, a in enumerate(models):
                for b in models[i + 1 :]:
                    p_a = strength[a] / (strength[a] + strength[b])
                    results.append(ComparisonResult(page, a, b, "A" if rng.random() < p_a else "B"))
        board = compute_elo(results, models, n_bootstrap=200)
        for rank, (model, _) in enumerate(board.ranked, start=1):
            best, worst = board.rank_range[model]
            assert 1 <= best <= rank <= worst <= len(models)

    @staticmethod
    def _co_moving_results(pages=80):
        """alpha and beta win or lose against six anchors together, page by page.

        Both ratings move the same way when the pages are resampled. alpha
        beats beta on 7 pages in 10.
        """
        anchors = [f"anchor-{n}" for n in range(6)]
        results = []
        for page in range(pages):
            both_win = (page * 7) % 10 < 5
            for top in ("alpha", "beta"):
                for anchor in anchors:
                    results.append(ComparisonResult(page, top, anchor, "A" if both_win else "B"))
            results.append(ComparisonResult(page, "alpha", "beta", "A" if page % 10 < 7 else "B"))
            for i, a in enumerate(anchors):
                for b in anchors[i + 1 :]:
                    results.append(ComparisonResult(page, a, b, "tie"))
        return results, ["alpha", "beta", *anchors]

    def test_overlapping_intervals_can_still_be_clearly_apart(self):
        results, models = self._co_moving_results()
        board = compute_elo(results, models, n_bootstrap=300)

        alpha_low, _ = board.elo_ci["alpha"]
        _, beta_high = board.elo_ci["beta"]
        assert beta_high > alpha_low  # the two intervals overlap
        # alpha could be first and beta could not: alpha is clearly above beta.
        assert board.rank_range["alpha"][0] == 1
        assert board.rank_range["beta"][0] == 2

    def test_the_difference_is_taken_within_each_resample(self):
        # The same two models, with alpha's rating from one resample set against
        # beta's rating from another: the common movement no longer cancels and
        # the difference is not clear. Only the paired difference separates them.
        results, models = self._co_moving_results()
        replicates = _bootstrap_replicates(results, models, n_bootstrap=300)
        assert _clearly_above("alpha", "beta", replicates)
        for shift in (7, 31, 101, 150):
            unpaired = [
                {
                    "alpha": replicates[i]["alpha"],
                    "beta": replicates[(i + shift) % len(replicates)]["beta"],
                }
                for i in range(len(replicates))
            ]
            assert not _clearly_above("alpha", "beta", unpaired)

    def test_models_with_the_same_results_are_never_apart(self):
        # twin-a and twin-b have the same result against the third model on
        # every page and tie with each other: their difference is exactly zero
        # in every resample.
        results = []
        for page in range(40):
            results.append(ComparisonResult(page, "twin-a", "twin-b", "tie"))
            winner = "A" if page % 3 else "B"
            results.append(ComparisonResult(page, "twin-a", "other", winner))
            results.append(ComparisonResult(page, "twin-b", "other", winner))
        models = ["twin-a", "twin-b", "other"]
        replicates = _bootstrap_replicates(results, models, n_bootstrap=200)
        assert not _clearly_above("twin-a", "twin-b", replicates)
        assert not _clearly_above("twin-b", "twin-a", replicates)
        board = compute_elo(results, models, n_bootstrap=200)
        assert board.rank_range["twin-a"] == board.rank_range["twin-b"]

    def test_clearly_above_follows_the_level(self):
        # 100 differences from -5 to 94: five are below zero. The lower bound
        # of the 95% interval is the third smallest (-3); that of the 60%
        # interval is the 21st (15).
        replicates = [{"a": float(i - 5), "b": 0.0} for i in range(100)]
        assert not _clearly_above("a", "b", replicates, ci=0.95)
        assert _clearly_above("a", "b", replicates, ci=0.60)
        # A lower bound of exactly zero is not above zero.
        at_zero = [{"a": float(max(i - 2, 0)), "b": 0.0} for i in range(100)]
        assert not _clearly_above("a", "b", at_zero, ci=0.95)

    def test_a_model_on_few_pages_widens_the_other_ranges_by_one(self):
        # Three models on 60 pages, clearly apart, and a fourth that is
        # compared on 12 pages only.
        models, results = self._results(60, lambda page, a, b: "B" if page % 10 == 0 else "A")
        for page in range(12):
            for other in models:
                results.append(ComparisonResult(page, "thin", other, "A" if page % 2 else "B"))
        board = compute_elo(results, [*models, "thin"], n_bootstrap=200)
        assert board.pages_by_model["thin"] == 12 < MIN_PAGES_FOR_RANK_RANGE
        assert board.rank_range["thin"] == (1, 4)
        widths = {name: worst - best for name, (best, worst) in board.rank_range.items()}
        assert widths == {"strong": 1, "middle": 1, "weak": 1, "thin": 3}

    def test_ranges_that_do_not_overlap_do_not_decide_a_pair(self):
        # A is clearly above C and D, and C and D are clearly above B, but the
        # difference between A and B is below zero in 40 of 1000 resamples
        # (20 where A is low, 20 where B is high): that pair fails its own
        # test. The ranges of A and B still do not overlap, so a range must
        # not be read as a statement about one pair.
        common = {"C": 1.0, "D": 1.0}
        replicates = (
            [{"A": 2.0, "B": 0.0, **common}] * 960
            + [{"A": -1.0, "B": 0.0, **common}] * 20
            + [{"A": 2.0, "B": 3.0, **common}] * 20
        )
        elo = {"A": 2.0, "C": 1.0, "D": 1.0, "B": 0.0}
        pages = dict.fromkeys(elo, MIN_PAGES_FOR_RANK_RANGE)
        assert _clearly_above("A", "C", replicates) and _clearly_above("C", "B", replicates)
        assert not _clearly_above("A", "B", replicates)
        assert not _clearly_above("B", "A", replicates)
        ranges = _rank_ranges(elo, replicates, pages)
        assert ranges == {"A": (1, 2), "C": (2, 3), "D": (2, 3), "B": (3, 4)}

    def test_a_range_of_one_rank_means_apart_from_every_other_model(self):
        # The statement the card makes. Checked on boards of random results.
        rng = random.Random(11)
        single_rank_models = 0
        for _ in range(12):
            models = [f"m{n}" for n in range(5)]
            strength = {m: 1.0 + rng.random() * 3 for m in models}
            results = []
            for page in range(40):
                for i, a in enumerate(models):
                    for b in models[i + 1 :]:
                        p_a = strength[a] / (strength[a] + strength[b])
                        results.append(
                            ComparisonResult(page, a, b, "A" if rng.random() < p_a else "B")
                        )
            replicates = _bootstrap_replicates(results, models, n_bootstrap=200)
            board = compute_elo(results, models, n_bootstrap=200)
            for model, (best, worst) in board.rank_range.items():
                if best != worst:
                    continue
                single_rank_models += 1
                for other in models:
                    if other != model:
                        assert _clearly_above(model, other, replicates) or _clearly_above(
                            other, model, replicates
                        )
        assert single_rank_models > 5  # the loop above checked something

    def test_the_page_gate_is_twenty_pages_for_both_models_of_a_pair(self):
        replicates = [{"high": 1600.0, "low": 1400.0}] * 200
        elo = {"high": 1600.0, "low": 1400.0}
        assert MIN_PAGES_FOR_RANK_RANGE == 20
        assert _rank_ranges(elo, replicates, {"high": 20, "low": 20}) == {
            "high": (1, 1),
            "low": (2, 2),
        }
        # One page short, on either side: that pair is not tested.
        third = {"other": 1500.0}
        replicates3 = [{"high": 1600.0, "low": 1400.0, "other": 1500.0}] * 200
        elo3 = {**elo, **third}
        assert _rank_ranges(elo3, replicates3, {"high": 19, "low": 20, "other": 20})["high"] == (
            1,
            3,
        )
        assert _rank_ranges(elo3, replicates3, {"high": 20, "low": 19, "other": 20})["high"] == (
            1,
            2,
        )

    def test_no_ranges_when_fewer_than_two_models_have_enough_pages(self):
        replicates = [{"high": 1600.0, "low": 1400.0}] * 200
        elo = {"high": 1600.0, "low": 1400.0}
        assert _rank_ranges(elo, replicates, {"high": 19, "low": 19}) == {}
        assert _rank_ranges(elo, replicates, {"high": 50, "low": 19}) == {}

    def test_two_models_on_few_pages_add_two_ranks_to_the_other_ranges(self):
        names = ["a", "b", "thin-1", "thin-2"]
        elo = {"a": 1700.0, "b": 1300.0, "thin-1": 1500.0, "thin-2": 1490.0}
        replicates = [dict(elo)] * 200
        pages = {"a": 60, "b": 60, "thin-1": 5, "thin-2": 5}
        ranges = _rank_ranges(elo, replicates, pages)
        assert ranges == {"a": (1, 3), "thin-1": (1, 4), "thin-2": (1, 4), "b": (2, 4)}
        assert set(ranges) == set(names)

    def test_the_level_of_the_ranges_is_95_percent_and_can_be_set(self):
        # "a" is below "b" in 5 of 100 resamples: not apart at 95%, apart at 60%.
        replicates = [{"a": float(i - 5), "b": 0.0} for i in range(100)]
        elo = {"a": 44.5, "b": 0.0}
        pages = {"a": 50, "b": 50}
        assert _rank_ranges(elo, replicates, pages) == {"a": (1, 2), "b": (1, 2)}
        assert _rank_ranges(elo, replicates, pages, ci=0.60) == {"a": (1, 1), "b": (2, 2)}

    def test_the_replicate_floor_is_one_hundred(self):
        elo = {"high": 1600.0, "low": 1400.0}
        pages = {"high": 50, "low": 50}
        assert _rank_ranges(elo, [dict(elo)] * 99, pages) == {}
        assert _rank_ranges(elo, [dict(elo)] * 100, pages) != {}

    def test_the_range_always_holds_the_fitted_rank(self):
        # Replicates that disagree with the fit (they put "low" above "high" in
        # every resample) cannot push a range past the fitted rank.
        elo = {"high": 1600.0, "low": 1400.0}
        replicates = [{"high": 1400.0 - i, "low": 1600.0 + i} for i in range(200)]
        ranges = _rank_ranges(elo, replicates, {"high": 50, "low": 50})
        assert ranges == {"high": (1, 2), "low": (1, 2)}

    def test_no_rank_range_from_a_few_replicates(self):
        models, results = self._results(60, lambda page, a, b: "B" if page % 10 == 0 else "A")
        few = compute_elo(results, models, n_bootstrap=MIN_REPLICATES_FOR_RANK_RANGE - 1)
        assert few.rank_range == {}
        assert few.elo_ci  # the intervals are still given
        enough = compute_elo(results, models, n_bootstrap=MIN_REPLICATES_FOR_RANK_RANGE)
        assert set(enough.rank_range) == set(models)

    def test_no_rank_range_without_a_bootstrap(self):
        models, results = self._results(20, lambda page, a, b: "A")
        assert compute_elo(results, models, n_bootstrap=0).rank_range == {}

    def test_the_bootstrap_intervals_do_not_depend_on_the_rank_ranges(self):
        # One set of resamples gives both: the published intervals stay the
        # same as the intervals of the separate interval function.
        models, results = self._results(30, lambda page, a, b: "B" if page % 4 == 0 else "A")
        board = compute_elo(results, models, n_bootstrap=100)
        assert board.elo_ci == _bootstrap_ci(results, models, n_bootstrap=100)

    def test_interval_bounds_use_the_same_order_statistics_at_every_level(self):
        # The published intervals must not move. The index arithmetic is the one
        # the intervals have always used; a different order of the same
        # operations picks another sample at some levels (0.84 with 125 samples).
        for n_samples in (20, 125, 200, 1000):
            samples = [float(value) for value in range(n_samples)]
            for ci in (0.5, 0.8, 0.84, 0.9, 0.95, 0.99):
                alpha = (1.0 - ci) / 2.0
                low_index = int(n_samples * (alpha * 100) / 100)
                high_index = min(int(n_samples * ((1.0 - alpha) * 100) / 100), n_samples - 1)
                assert _percentile_bounds(samples, ci) == (
                    samples[low_index],
                    samples[high_index],
                ), (n_samples, ci)
        assert _percentile_bounds([float(v) for v in range(125)], 0.84) == (10.0, 115.0)
