"""Regression tests for the read-only adaptive replay experiment."""

from __future__ import annotations

import importlib.util
import sys
from collections import Counter
from pathlib import Path

from ocr_bench.elo import ComparisonResult, Leaderboard

_REPLAY_PATH = (
    Path(__file__).resolve().parents[1] / "experiments" / "adaptive-stopping" / "replay.py"
)
_SPEC = importlib.util.spec_from_file_location("adaptive_stopping_replay", _REPLAY_PATH)
assert _SPEC is not None and _SPEC.loader is not None
replay = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = replay
_SPEC.loader.exec_module(replay)


def _stored_grid(n_samples: int = 8) -> list[ComparisonResult]:
    return [
        ComparisonResult(sample_idx, model_a, model_b, "A")
        for sample_idx in range(n_samples)
        for model_a, model_b in (("a", "b"), ("a", "c"), ("b", "c"))
    ]


def _fixed_board() -> Leaderboard:
    return Leaderboard(
        elo={"a": 1600.0, "b": 1500.0, "c": 1400.0},
        elo_ci={
            "a": (1580.0, 1620.0),
            "b": (1450.0, 1550.0),
            "c": (1380.0, 1520.0),
        },
    )


def test_targeted_replay_warms_up_balanced_then_filters_pairs(monkeypatch):
    stored = _stored_grid()
    monkeypatch.setattr(replay, "compute_elo", lambda *args, **kwargs: _fixed_board())

    targeted = replay.replay_strategy(
        stored,
        ["a", "b", "c"],
        replay.StrategyConfig("targeted", "targeted"),
        n_bootstrap=1,
        batch_samples=1,
    )
    balanced = replay.replay_strategy(
        stored,
        ["a", "b", "c"],
        replay.StrategyConfig("balanced", "balanced"),
        n_bootstrap=1,
        batch_samples=1,
    )

    # Production waits for max(3 * 3 pairs, 20) outcomes: seven complete
    # samples (21 rows), then only the unresolved b/c pair is selected.
    assert len(targeted.comparisons) == 22
    assert len(balanced.comparisons) == 24
    assert targeted.round_history[-1]["active_pairs"] == 1
    assert targeted.comparisons[-1].model_a == "b"
    assert targeted.comparisons[-1].model_b == "c"
    assert targeted.stopping_reason == "sample_batches_exhausted"


def test_per_pair_warmup_requires_every_pair_to_reach_floor():
    counts = Counter({("a", "b"): 5, ("a", "c"): 5, ("b", "c"): 4})

    assert replay._warmup_ready(
        counts,
        n_pairs=3,
        total_comparisons=20,
        min_before_check=20,
        min_per_pair=None,
    )
    assert not replay._warmup_ready(
        counts,
        n_pairs=3,
        total_comparisons=20,
        min_before_check=20,
        min_per_pair=5,
    )
    counts[("b", "c")] = 5
    assert replay._warmup_ready(
        counts,
        n_pairs=3,
        total_comparisons=20,
        min_before_check=20,
        min_per_pair=5,
    )


def test_periodic_exploration_restores_a_balanced_batch(monkeypatch):
    stored = _stored_grid(10)
    monkeypatch.setattr(replay, "compute_elo", lambda *args, **kwargs: _fixed_board())

    result = replay.replay_strategy(
        stored,
        ["a", "b", "c"],
        replay.StrategyConfig(
            "explore",
            "targeted",
            balanced_every_n_post_warmup_batches=3,
        ),
        n_bootstrap=1,
        batch_samples=1,
    )

    assert len(result.comparisons) == 26
    assert result.round_history[-1]["allocation_mode"] == "balanced-exploration"
    assert result.round_history[-1]["batch_comparisons"] == 3


def test_size_rule_can_annotate_without_controlling_sampling():
    small = "zai-org/GLM-OCR"
    large = "rednote-hilab/dots.mocr"
    board = Leaderboard(
        elo={large: 1510.0, small: 1500.0},
        elo_ci={large: (1450.0, 1570.0), small: (1440.0, 1560.0)},
    )
    comparisons = [ComparisonResult(i, large, small, "A") for i in range(10)]
    config = replay.StrategyConfig(
        "annotate-only",
        "targeted",
        size_tie_ratio=3.0,
        size_rule_controls_sampling=False,
    )

    assert replay._classify(board, comparisons, config)[0].status == "prefer-smaller"
    assert replay._classify(board, comparisons, config, for_sampling=True)[0].status == "unresolved"


def test_fixed_pair_balanced_sample_is_exact_balanced_and_seeded():
    stored = _stored_grid(8)

    first = replay._fixed_pair_balanced_sample(stored, budget=14, seed=42)
    repeated = replay._fixed_pair_balanced_sample(stored, budget=14, seed=42)
    other_seed = replay._fixed_pair_balanced_sample(stored, budget=14, seed=43)
    counts = replay.comparison_pair_counts(first)

    assert len(first) == 14
    assert sorted(counts.values()) == [4, 5, 5]
    assert [replay._comparison_key(row) for row in first] == [
        replay._comparison_key(row) for row in repeated
    ]
    assert [replay._comparison_key(row) for row in first] != [
        replay._comparison_key(row) for row in other_seed
    ]


def test_mixed_sample_preserves_required_warmup_and_rebalances():
    stored = _stored_grid(8)
    required = stored[:3]

    selected = replay._fixed_pair_balanced_sample(
        stored,
        budget=14,
        seed=42,
        required=required,
    )
    keys = [replay._comparison_key(row) for row in selected]
    counts = replay.comparison_pair_counts(selected)

    assert selected[:3] == required
    assert len(keys) == len(set(keys)) == 14
    assert max(counts.values()) - min(counts.values()) == 1


def test_graph_metrics_report_coverage_and_connectivity():
    complete = replay.graph_metrics(_stored_grid(1), ["a", "b", "c"])
    disconnected = replay.graph_metrics(
        [ComparisonResult(0, "a", "b", "A")],
        ["a", "b", "c"],
    )

    assert complete["observed_pairs"] == 3
    assert complete["pair_coverage_pct"] == 100.0
    assert complete["connected"] is True
    assert disconnected["observed_pairs"] == 1
    assert disconnected["connected"] is False
    assert disconnected["connected_components"] == [["a", "b"], ["c"]]
