"""Property-based tests (Hypothesis) for the pure judge/scoring functions.

These target the boundary where LLM judge output enters the pipeline —
arbitrary text and JSON must never crash the parser or the vote
aggregator — plus algebraic invariants of the swap and ELO logic.
"""

from __future__ import annotations

import json

from hypothesis import example, given
from hypothesis import strategies as st

from ocr_bench.backends import aggregate_jury_votes
from ocr_bench.elo import ComparisonResult, _unswap_winner, compute_elo
from ocr_bench.judge import parse_judge_output

WINNERS = ("A", "B", "tie")


# ---------------------------------------------------------------------------
# parse_judge_output — judge output is adversarial by nature
# ---------------------------------------------------------------------------

json_values = st.recursive(
    st.none()
    | st.booleans()
    | st.integers()
    | st.floats(allow_nan=False)
    | st.text(max_size=20),
    lambda children: st.lists(children, max_size=4)
    | st.dictionaries(st.text(max_size=10), children, max_size=4),
    max_leaves=10,
)


class TestParseJudgeOutputProperties:
    @given(st.text(max_size=200))
    @example("```")  # truncated fence: opening backticks, no body
    @example("```json")
    @example("```\n")
    def test_never_raises_on_arbitrary_text(self, text):
        result = parse_judge_output(text)
        assert result == {} or result["winner"] in WINNERS

    @given(json_values)
    def test_never_raises_on_arbitrary_json(self, value):
        result = parse_judge_output(json.dumps(value))
        if result:
            assert result["winner"] in WINNERS
            assert isinstance(result["reason"], str)

    @given(st.sampled_from(WINNERS), st.text(max_size=50))
    def test_well_formed_output_round_trips(self, winner, reason):
        result = parse_judge_output(json.dumps({"winner": winner, "reason": reason}))
        assert result["winner"] == winner
        assert result["reason"] == reason


# ---------------------------------------------------------------------------
# Position-bias swap
# ---------------------------------------------------------------------------


class TestUnswapProperties:
    @given(st.sampled_from(WINNERS), st.booleans())
    def test_unswap_is_involution(self, winner, swapped):
        assert _unswap_winner(_unswap_winner(winner, swapped), swapped) == winner

    @given(st.booleans())
    def test_tie_is_swap_invariant(self, swapped):
        assert _unswap_winner("tie", swapped) == "tie"


# ---------------------------------------------------------------------------
# aggregate_jury_votes
# ---------------------------------------------------------------------------

vote_result = st.one_of(
    st.just({}),  # failed judge call
    st.fixed_dictionaries(
        {"winner": st.sampled_from(WINNERS), "reason": st.text(max_size=20)}
    ),
)


@st.composite
def jury_results(draw):
    n_judges = draw(st.integers(min_value=1, max_value=5))
    n_comps = draw(st.integers(min_value=0, max_value=6))
    results = [[draw(vote_result) for _ in range(n_comps)] for _ in range(n_judges)]
    names = [f"judge-{i}" for i in range(n_judges)]
    return results, names


class TestAggregateJuryProperties:
    @given(jury_results())
    def test_output_shape_and_verdict_validity(self, args):
        results, names = args
        agg = aggregate_jury_votes(results, names)
        assert len(agg) == len(results[0])
        for out in agg:
            assert out["winner"] in WINNERS
            num, den = (int(p) for p in out["agreement"].split("/"))
            assert 0 <= num <= den <= len(names)

    @given(jury_results())
    def test_judge_order_does_not_matter(self, args):
        """The verdict must be independent of jury listing order."""
        results, names = args
        forward = aggregate_jury_votes(results, names)
        backward = aggregate_jury_votes(list(reversed(results)), list(reversed(names)))
        assert [o["winner"] for o in forward] == [o["winner"] for o in backward]
        assert [o["agreement"] for o in forward] == [o["agreement"] for o in backward]

    @given(jury_results())
    def test_unanimous_votes_win(self, args):
        """If every valid vote agrees, that vote is the verdict."""
        results, names = args
        agg = aggregate_jury_votes(results, names)
        for i, out in enumerate(agg):
            votes = {r[i]["winner"] for r in results if i < len(r) and r[i]}
            if len(votes) == 1:
                assert out["winner"] == votes.pop()


# ---------------------------------------------------------------------------
# compute_elo
# ---------------------------------------------------------------------------

MODELS = ["model-x", "model-y", "model-z"]


@st.composite
def comparison_results(draw):
    n = draw(st.integers(min_value=0, max_value=20))
    results = []
    for i in range(n):
        a, b = draw(st.permutations(MODELS))[:2]
        results.append(
            ComparisonResult(
                sample_idx=i,
                model_a=a,
                model_b=b,
                winner=draw(st.sampled_from(WINNERS)),
                swapped=draw(st.booleans()),
            )
        )
    return results


class TestComputeEloProperties:
    @given(comparison_results())
    def test_never_crashes_and_covers_all_models(self, results):
        board = compute_elo(results, MODELS, n_bootstrap=0)
        assert set(board.elo) == set(MODELS)
        for model in MODELS:
            assert board.wins[model] >= 0

    @given(comparison_results())
    def test_wins_and_losses_balance(self, results):
        """Every win is someone else's loss; ties come in pairs."""
        board = compute_elo(results, MODELS, n_bootstrap=0)
        assert sum(board.wins.values()) == sum(board.losses.values())
        assert sum(board.ties.values()) % 2 == 0

    @given(comparison_results())
    def test_swap_is_neutralised(self, results):
        """Flipping swapped + winner on every result must not change tallies."""
        flipped = [
            ComparisonResult(
                sample_idx=r.sample_idx,
                model_a=r.model_a,
                model_b=r.model_b,
                winner=_unswap_winner(r.winner, True),
                swapped=not r.swapped,
            )
            for r in results
        ]
        a = compute_elo(results, MODELS, n_bootstrap=0)
        b = compute_elo(flipped, MODELS, n_bootstrap=0)
        assert a.wins == b.wins
        assert a.losses == b.losses
        assert a.ties == b.ties
