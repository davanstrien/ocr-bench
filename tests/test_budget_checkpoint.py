"""Tests for --max-comparisons budget and --checkpoint-every checkpointing.

These drive ``cli.cmd_judge`` end-to-end with a fake dataset + fake judge and
mocked Hub publish/checkpoint, so they exercise the real budget-trim, chunked
checkpoint, and resume-skip control flow without any network I/O.
"""

from __future__ import annotations

from unittest.mock import patch

from PIL import Image

from ocr_bench import cli
from ocr_bench.cli import build_parser
from ocr_bench.elo import ComparisonResult
from ocr_bench.judge import _normalize_pair


class FakeDataset:
    """Minimal stand-in for an HF Dataset: image column + OCR text columns."""

    def __init__(self, n: int, columns: dict[str, list[str]]):
        self._n = n
        self._columns = columns
        self._img = Image.new("RGB", (8, 8), "white")

    def __len__(self) -> int:
        return self._n

    @property
    def column_names(self) -> list[str]:
        return list(self._columns)

    def __getitem__(self, key):
        if isinstance(key, str):
            return self._columns[key]
        row = {c: self._columns[c][key] for c in self._columns}
        row["image"] = self._img
        return row


class FakeJudge:
    """Judge backend recording how many comparisons (and which pairs) it saw."""

    def __init__(self, name: str = "fake-judge", winner: str = "A"):
        self.name = name
        self.winner = winner
        self.judged = 0
        self.pairs_seen: list[tuple[str, str]] = []

    def judge(self, comparisons):
        self.judged += len(comparisons)
        for comp in comparisons:
            self.pairs_seen.append(_normalize_pair(comp.model_a, comp.model_b))
        return [{"winner": self.winner, "reason": "r"} for _ in comparisons]


def make_ds(n: int = 10, models: tuple[str, ...] = ("a", "b", "c")):
    """Build a FakeDataset with `n` rows and one text column per model."""
    cols = {f"col_{m}": [f"text {m} {i}" for i in range(n)] for m in models}
    ocr_columns = {f"col_{m}": f"model-{m}" for m in models}
    return FakeDataset(n, cols), ocr_columns


def _run_judge(
    argv_extra: list[str],
    ds: FakeDataset,
    ocr_columns: dict[str, str],
    *,
    existing: list[ComparisonResult] | None = None,
    judge: FakeJudge | None = None,
    checkpoint_side_effect=None,
):
    """Run cmd_judge with dataset load, judge, and Hub calls patched out."""
    judge = judge or FakeJudge()
    argv = [
        "judge",
        "user/ds",
        "--columns",
        *ocr_columns.keys(),
        "--save-results",
        "user/results",
        *argv_extra,
    ]
    args = build_parser().parse_args(argv)

    with (
        patch.object(cli, "load_flat_dataset", return_value=(ds, ocr_columns)),
        patch.object(cli, "parse_judge_spec", return_value=judge),
        patch.object(cli, "load_existing_comparisons", return_value=existing or []),
        patch.object(cli, "load_existing_metadata", return_value=[]),
        patch.object(cli, "publish_results") as m_publish,
        patch.object(cli, "publish_checkpoint") as m_checkpoint,
    ):
        if checkpoint_side_effect is not None:
            m_checkpoint.side_effect = checkpoint_side_effect
        cli.cmd_judge(args)

    return judge, m_publish, m_checkpoint


def _published_metadata(m_publish):
    """Extract the EvalMetadata passed to publish_results (3rd positional arg)."""
    return m_publish.call_args.args[2]


class TestParserFlags:
    def test_max_comparisons_default_none(self):
        args = build_parser().parse_args(["judge", "user/ds"])
        assert args.max_comparisons is None

    def test_checkpoint_every_default_500(self):
        args = build_parser().parse_args(["judge", "user/ds"])
        assert args.checkpoint_every == 500

    def test_flags_parse_explicitly(self):
        args = build_parser().parse_args(
            ["judge", "user/ds", "--max-comparisons", "100", "--checkpoint-every", "0"]
        )
        assert args.max_comparisons == 100
        assert args.checkpoint_every == 0


class TestBudget:
    def test_budget_stops_at_n_and_publishes(self):
        # 10 samples x 3 pairs = 30 possible comparisons; cap at 12.
        # Ties never converge, so only the budget can stop the adaptive run.
        ds, ocr = make_ds(n=10)
        judge, m_publish, _ = _run_judge(
            ["--max-comparisons", "12", "--checkpoint-every", "0"],
            ds,
            ocr,
            judge=FakeJudge(winner="tie"),
        )
        assert judge.judged == 12
        m_publish.assert_called_once()
        meta = _published_metadata(m_publish)
        assert meta.budget_exhausted is True
        assert meta.max_comparisons == 12
        assert meta.total_comparisons == 12

    def test_budget_non_adaptive_trims(self):
        ds, ocr = make_ds(n=10)  # 30 possible
        judge, m_publish, _ = _run_judge(
            ["--no-adaptive", "--max-comparisons", "7", "--checkpoint-every", "0"],
            ds,
            ocr,
        )
        assert judge.judged == 7
        assert _published_metadata(m_publish).budget_exhausted is True

    def test_no_budget_judges_everything(self):
        ds, ocr = make_ds(n=10)
        judge, m_publish, _ = _run_judge(
            ["--no-adaptive", "--checkpoint-every", "0"], ds, ocr
        )
        assert judge.judged == 30
        assert _published_metadata(m_publish).budget_exhausted is False


class TestCheckpointing:
    def test_checkpoints_fire_every_k(self):
        # 30 comparisons, checkpoint every 10 -> pushes at 10, 20, 30.
        ds, ocr = make_ds(n=10)
        judge, m_publish, m_checkpoint = _run_judge(
            ["--no-adaptive", "--checkpoint-every", "10"], ds, ocr
        )
        assert judge.judged == 30
        assert m_checkpoint.call_count == 3
        m_publish.assert_called_once()

    def test_checkpoint_off_never_pushes(self):
        ds, ocr = make_ds(n=10)
        _, _, m_checkpoint = _run_judge(
            ["--no-adaptive", "--checkpoint-every", "0"], ds, ocr
        )
        m_checkpoint.assert_not_called()

    def test_checkpoint_failure_does_not_abort(self):
        # Every checkpoint push raises; the run must still judge all pairs and
        # reach the final publish.
        ds, ocr = make_ds(n=10)
        judge, m_publish, m_checkpoint = _run_judge(
            ["--no-adaptive", "--checkpoint-every", "10"],
            ds,
            ocr,
            checkpoint_side_effect=RuntimeError("hub down"),
        )
        assert judge.judged == 30
        assert m_checkpoint.call_count == 3  # attempted despite failing
        m_publish.assert_called_once()

    def test_checkpoints_fire_in_adaptive_mode(self):
        # Adaptive checkpoints at batch boundaries: batch of 5 samples x 3 pairs
        # = 15 comparisons; with K=5 each batch crosses the threshold once.
        ds, ocr = make_ds(n=10)
        _, _, m_checkpoint = _run_judge(
            ["--checkpoint-every", "5"], ds, ocr, judge=FakeJudge(winner="tie")
        )
        assert m_checkpoint.call_count >= 1


class TestResume:
    def test_resume_skips_checkpointed_pairs(self):
        # A prior (checkpointed) run already judged the (model-a, model-b) pair.
        # Relaunch WITHOUT --full-rejudge: that pair is skipped for ALL samples
        # (pair-level skip), so only (a,c) and (b,c) get judged this run.
        ds, ocr = make_ds(n=10)
        existing = [
            ComparisonResult(
                sample_idx=i, model_a="model-a", model_b="model-b", winner="A"
            )
            for i in range(4)
        ]
        judge, m_publish, _ = _run_judge(
            ["--no-adaptive", "--checkpoint-every", "0"],
            ds,
            ocr,
            existing=existing,
        )
        # 10 samples x 2 remaining pairs = 20 (the (a,b) pair is fully skipped).
        assert judge.judged == 20
        assert ("model-a", "model-b") not in judge.pairs_seen
        assert ("model-a", "model-c") in judge.pairs_seen
        assert ("model-b", "model-c") in judge.pairs_seen
        m_publish.assert_called_once()

    def test_full_rejudge_ignores_existing(self):
        ds, ocr = make_ds(n=10)
        existing = [
            ComparisonResult(
                sample_idx=i, model_a="model-a", model_b="model-b", winner="A"
            )
            for i in range(4)
        ]
        judge, _, _ = _run_judge(
            ["--no-adaptive", "--full-rejudge", "--checkpoint-every", "0"],
            ds,
            ocr,
            existing=existing,
        )
        # --full-rejudge drops skip_pairs, so all 3 pairs x 10 samples = 30.
        assert judge.judged == 30
        assert ("model-a", "model-b") in judge.pairs_seen
