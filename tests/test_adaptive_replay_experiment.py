"""Regression tests for the read-only adaptive replay experiment."""

from __future__ import annotations

import importlib.util
import sys
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
