"""Tests for --max-comparisons budget and --checkpoint-every checkpointing.

These drive ``cli.cmd_judge`` end-to-end with a fake dataset + fake judge and
mocked Hub publish/checkpoint, so they exercise the real budget-trim, chunked
checkpoint, and resume-skip control flow without any network I/O.
"""

from __future__ import annotations

import json
from contextlib import nullcontext
from unittest.mock import patch

import pytest
from datasets import Dataset
from PIL import Image

from ocr_bench import cli
from ocr_bench.cli import build_parser
from ocr_bench.dataset import DatasetError
from ocr_bench.elo import ComparisonResult
from ocr_bench.judge import CRITERIA_PROFILES, _normalize_pair, prompt_hash


class FakeDataset:
    """Minimal stand-in for an HF Dataset: image column + OCR text columns."""

    def __init__(self, n: int, columns: dict[str, list[str]]):
        self._n = n
        self._columns = columns
        self._img = Image.new("RGB", (8, 8), "white")
        # cmd_judge requires the same stable identity real HF Datasets expose.
        self._fingerprint = f"fake-{n}-{'-'.join(columns)}"

    def __len__(self) -> int:
        return self._n

    @property
    def column_names(self) -> list[str]:
        return [*self._columns, "image"]

    features: dict = {}

    def select_columns(self, _columns: list[str]):
        return self

    def select(self, indices: list[int]):
        return FakeDataset(
            len(indices),
            {column: [values[i] for i in indices] for column, values in self._columns.items()},
        )

    def __getitem__(self, key):
        if isinstance(key, str):
            if key == "image":
                return [self._img] * self._n
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
        self.pair_samples: set[tuple[tuple[str, str], int]] = set()

    def judge(self, comparisons):
        self.judged += len(comparisons)
        for comp in comparisons:
            pair = _normalize_pair(comp.model_a, comp.model_b)
            self.pairs_seen.append(pair)
            self.pair_samples.add((pair, comp.sample_idx))
        return [{"winner": self.winner, "reason": "r"} for _ in comparisons]


def make_ds(n: int = 10, models: tuple[str, ...] = ("a", "b", "c")):
    """Build a FakeDataset with `n` rows and one text column per model.

    Text is kept well above the default ``--min-chars`` threshold and distinct
    per (model, sample) so every pair is judged — these tests exercise budget /
    checkpoint / resume mechanics, not the blank-pair or auto-tie filters.
    """
    cols = {
        f"col_{m}": [f"OCR transcription output for model {m}, sample {i}" for i in range(n)]
        for m in models
    }
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
    stamp_existing_provenance: bool = True,
    checkpoint_resume: bool = False,
    existing_meta_override: list[dict] | None = None,
    config_datasets: dict[str, Dataset] | None = None,
):
    """Run cmd_judge with dataset load, judge, and Hub calls patched out."""
    judge = judge or FakeJudge()
    argv = [
        "judge",
        "user/ds",
        "--configs" if config_datasets is not None else "--columns",
        *(config_datasets if config_datasets is not None else ocr_columns),
        "--save-results",
        "user/results",
        *argv_extra,
    ]
    args = build_parser().parse_args(argv)
    if existing and stamp_existing_provenance:
        model_to_column = {model: column for column, model in ocr_columns.items()}
        for result in existing:
            if result.model_a in model_to_column and result.model_b in model_to_column:
                result.col_a = result.col_a or model_to_column[result.model_a]
                result.col_b = result.col_b or model_to_column[result.model_b]
                result.provenance_hash = "matching-test-provenance"
    existing_meta = []
    if existing and not checkpoint_resume:
        existing_meta = [
            {
                "criteria": "default",
                "prompt_hash": prompt_hash(CRITERIA_PROFILES["default"]),
                "judge_text_mode": "normalized",
                "max_ocr_text_len": 2500,
                "judge_image_dim": 1024,
            }
        ]
    if existing_meta_override is not None:
        existing_meta = existing_meta_override
    provenance_patch = (
        patch.object(
            cli,
            "evaluation_provenance_hash",
            return_value="matching-test-provenance",
        )
        if stamp_existing_provenance
        else nullcontext()
    )

    with (
        patch.object(cli, "load_flat_dataset", return_value=(ds, ocr_columns)),
        patch(
            "ocr_bench.dataset.load_dataset",
            side_effect=lambda path, name, **kwargs: config_datasets[name],
        )
        if config_datasets is not None
        else nullcontext(),
        patch.object(cli, "parse_judge_spec", return_value=judge),
        patch.object(cli, "load_existing_comparisons", return_value=existing or []),
        patch.object(cli, "load_existing_metadata", return_value=existing_meta),
        provenance_patch,
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

    def test_checkpoint_every_default_none_sentinel(self):
        # Parser leaves it None (unspecified); cmd_judge resolves to 500, or 0
        # under --full-rejudge.
        args = build_parser().parse_args(["judge", "user/ds"])
        assert args.checkpoint_every is None

    def test_flags_parse_explicitly(self):
        args = build_parser().parse_args(
            ["judge", "user/ds", "--max-comparisons", "100", "--checkpoint-every", "0"]
        )
        assert args.max_comparisons == 100
        assert args.checkpoint_every == 0


class TestArgValidators:
    @pytest.mark.parametrize("bad", ["0", "-1", "-100"])
    def test_max_comparisons_rejects_non_positive(self, bad):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["judge", "user/ds", "--max-comparisons", bad])

    @pytest.mark.parametrize("bad", ["-1", "-100"])
    def test_checkpoint_every_rejects_negative(self, bad):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["judge", "user/ds", "--checkpoint-every", bad])

    def test_checkpoint_every_zero_allowed(self):
        args = build_parser().parse_args(["judge", "user/ds", "--checkpoint-every", "0"])
        assert args.checkpoint_every == 0

    def test_max_comparisons_one_allowed(self):
        args = build_parser().parse_args(["judge", "user/ds", "--max-comparisons", "1"])
        assert args.max_comparisons == 1


class TestTargetedAdaptive:
    def test_opt_in_targets_only_adjacent_pairs_after_balanced_round(self):
        # 4 models -> 6 pairs. Round 1 judges all 6 pairs over 5 samples (30).
        # All-tie CIs overlap, so round 2 targets only 3 adjacent pairs (15),
        # rather than another full 30-comparison grid.
        ds, ocr = make_ds(n=10, models=("a", "b", "c", "d"))
        judge, m_publish, _ = _run_judge(
            ["--adaptive-strategy", "targeted", "--checkpoint-every", "0"],
            ds,
            ocr,
            judge=FakeJudge(winner="tie"),
        )
        assert judge.judged == 45
        pair_counts = {}
        for pair in judge.pairs_seen:
            pair_counts[pair] = pair_counts.get(pair, 0) + 1
        assert sorted(pair_counts.values()) == [5, 5, 5, 10, 10, 10]
        meta = _published_metadata(m_publish)
        assert meta.adaptive_strategy == "targeted"
        assert meta.size_tie_ratio is None

    def test_resume_reconstructs_target_pairs_without_filling_sparse_grid(self):
        ds, ocr = make_ds(n=10, models=("a", "b", "c", "d"))
        _, first_publish, _ = _run_judge(
            ["--adaptive-strategy", "targeted", "--checkpoint-every", "0"],
            ds,
            ocr,
            judge=FakeJudge(winner="tie"),
        )
        first_board = first_publish.call_args.args[1]
        existing = [
            ComparisonResult(
                sample_idx=row["sample_idx"],
                model_a=row["model_a"],
                model_b=row["model_b"],
                winner=row["winner"],
                reason=row.get("reason", ""),
                agreement=row.get("agreement", "1/1"),
                col_a=row.get("col_a", ""),
                col_b=row.get("col_b", ""),
                provenance_hash=row.get("provenance_hash", ""),
            )
            for row in first_board.comparison_log
        ]

        resumed_judge, resumed_publish, _ = _run_judge(
            ["--adaptive-strategy", "targeted", "--checkpoint-every", "0"],
            ds,
            ocr,
            existing=existing,
            judge=FakeJudge(winner="tie"),
        )
        assert len(existing) == 45
        assert resumed_judge.judged == 0
        resumed_board = resumed_publish.call_args.args[1]
        assert len(resumed_board.comparison_log) == 45

    def test_added_model_gets_balanced_warmup_before_targeting(self):
        ds, ocr = make_ds(n=10, models=("a", "b", "c", "d"))
        existing = [
            ComparisonResult(
                sample_idx=sample_idx,
                model_a=f"model-{left}",
                model_b=f"model-{right}",
                winner="tie",
            )
            for sample_idx in range(10)
            for left, right in (("a", "b"), ("a", "c"), ("b", "c"))
        ]
        judge, _, _ = _run_judge(
            ["--adaptive-strategy", "targeted", "--checkpoint-every", "0"],
            ds,
            ocr,
            existing=existing,
            judge=FakeJudge(winner="tie"),
        )
        pairs_with_new_model = {pair for pair in judge.pairs_seen if "model-d" in pair}
        assert pairs_with_new_model == {
            ("model-a", "model-d"),
            ("model-b", "model-d"),
            ("model-c", "model-d"),
        }

    def test_interrupted_new_model_warmup_stays_balanced_on_resume(self):
        ds, ocr = make_ds(n=10, models=("a", "b", "c", "d"))
        existing = [
            ComparisonResult(
                sample_idx=sample_idx,
                model_a=f"model-{left}",
                model_b=f"model-{right}",
                winner="tie",
            )
            for sample_idx in range(10)
            for left, right in (("a", "b"), ("a", "c"), ("b", "c"))
        ]
        # Simulate a budget/kill after the first new-model comparison: model-d
        # appears in the log, but two of its three pairs have no evidence yet.
        existing.append(ComparisonResult(0, "model-a", "model-d", "tie"))

        judge, _, _ = _run_judge(
            ["--adaptive-strategy", "targeted", "--checkpoint-every", "0"],
            ds,
            ocr,
            existing=existing,
            judge=FakeJudge(winner="tie"),
        )
        assert ("model-b", "model-d") in judge.pairs_seen
        assert ("model-c", "model-d") in judge.pairs_seen

    def test_balanced_default_still_judges_all_pairs_each_round(self):
        ds, ocr = make_ds(n=10, models=("a", "b", "c", "d"))
        judge, m_publish, _ = _run_judge(
            ["--checkpoint-every", "0"],
            ds,
            ocr,
            judge=FakeJudge(winner="tie"),
        )
        assert judge.judged == 60
        assert _published_metadata(m_publish).adaptive_strategy == "balanced"

    def test_size_rule_can_stop_after_minimum_direct_evidence(self):
        model_ids = (
            "PaddlePaddle/PP-OCRv6_medium",  # 34.5M
            "tiiuae/Falcon-OCR",  # 0.3B
            "lightonai/LightOnOCR-2-1B",  # 1B
            "deepseek-ai/DeepSeek-OCR",  # 4B
        )
        ds, _ = make_ds(n=10, models=("a", "b", "c", "d"))
        ocr = {
            f"col_{name}": model_id
            for name, model_id in zip(("a", "b", "c", "d"), model_ids, strict=True)
        }
        judge, m_publish, _ = _run_judge(
            [
                "--adaptive-strategy",
                "targeted",
                "--size-tie-ratio",
                "3",
                "--size-tie-min-samples",
                "5",
                "--checkpoint-every",
                "0",
            ],
            ds,
            ocr,
            judge=FakeJudge(winner="tie"),
        )
        assert judge.judged == 30  # one balanced 5-sample round, then stop
        meta = _published_metadata(m_publish)
        assert meta.size_tie_ratio == 3.0
        assert meta.size_tie_min_samples == 5

    def test_targeted_strategy_rejects_no_adaptive(self):
        ds, ocr = make_ds()
        with pytest.raises(DatasetError, match="requires adaptive mode"):
            _run_judge(
                ["--no-adaptive", "--adaptive-strategy", "targeted"],
                ds,
                ocr,
            )


class NameOrderJudge(FakeJudge):
    """A judge that always prefers the model whose name sorts first."""

    def judge(self, comparisons):
        outputs = super().judge(comparisons)
        for comp, output in zip(comparisons, outputs, strict=True):
            model_a_is_better = comp.model_a < comp.model_b
            # The verdict names the displayed position. With ``swapped``, the
            # judge saw model_b's output as "Output A".
            model_a_shown_as_a = not comp.swapped
            output["winner"] = "A" if model_a_is_better == model_a_shown_as_a else "B"
        return outputs


class TestBalancedRerun:
    """Balanced mode checks the stopping rule on the stored comparisons first."""

    FLAGS = ("--checkpoint-every", "0")

    @staticmethod
    def _stored_results(m_publish) -> list[ComparisonResult]:
        """The published comparisons, as a later run loads them."""
        return [
            ComparisonResult(
                sample_idx=row["sample_idx"],
                model_a=row["model_a"],
                model_b=row["model_b"],
                winner=row["winner"],
            )
            for row in m_publish.call_args.args[1].comparison_log
        ]

    def _converged_board(self) -> list[ComparisonResult]:
        """Run three models to convergence: 2 of the 8 batches are enough."""
        ds, ocr = make_ds(n=40, models=("a", "b", "c"))
        judge, m_publish, _ = _run_judge([*self.FLAGS], ds, ocr, judge=NameOrderJudge())
        assert judge.judged == 30  # stopped by the rule, not at the end of the sample
        return self._stored_results(m_publish)

    def test_rerun_of_a_converged_board_makes_no_judge_call(self, capsys):
        existing = self._converged_board()
        ds, ocr = make_ds(n=40, models=("a", "b", "c"))
        capsys.readouterr()

        judge, m_publish, _ = _run_judge(
            [*self.FLAGS], ds, ocr, existing=existing, judge=NameOrderJudge()
        )

        assert judge.judged == 0
        out = " ".join(capsys.readouterr().out.split())  # undo console line wrapping
        assert "Existing comparisons already meet the adaptive stopping criteria." in out
        assert len(m_publish.call_args.args[1].comparison_log) == 30
        assert _published_metadata(m_publish).total_comparisons == 0

    def test_the_message_says_how_to_judge_more(self, capsys):
        existing = self._converged_board()
        ds, ocr = make_ds(n=40, models=("a", "b", "c"))
        capsys.readouterr()
        _run_judge([*self.FLAGS], ds, ocr, existing=existing, judge=NameOrderJudge())
        out = " ".join(capsys.readouterr().out.split())
        assert "To judge more pages anyway, run with --no-adaptive" in out

    def test_no_check_before_the_first_batch_after_a_run_that_hit_its_budget(self):
        # A budget stops a run in the middle of a batch. Checking the rule at
        # that point, at every budget stop, would look at it far more often
        # than one run with no budget does. The run goes on to the end of a
        # batch and checks there.
        existing = self._converged_board()
        ds, ocr = make_ds(n=40, models=("a", "b", "c"))
        meta_row = {
            "criteria": "default",
            "prompt_hash": prompt_hash(CRITERIA_PROFILES["default"]),
            "judge_text_mode": "normalized",
            "max_ocr_text_len": 2500,
            "judge_image_dim": 1024,
        }
        after_budget_stop, _, _ = _run_judge(
            [*self.FLAGS],
            ds,
            ocr,
            existing=existing,
            judge=NameOrderJudge(),
            existing_meta_override=[{**meta_row, "budget_exhausted": True}],
        )
        assert after_budget_stop.judged == 15  # one batch of five pages, then the rule
        after_own_stop, _, _ = _run_judge(
            [*self.FLAGS],
            ds,
            ocr,
            existing=self._converged_board(),
            judge=NameOrderJudge(),
            existing_meta_override=[{**meta_row, "budget_exhausted": False}],
        )
        assert after_own_stop.judged == 0

    def test_too_few_stored_comparisons_for_the_rule_still_judges(self):
        # Two models: the rule is not checked before 20 comparisons. Twelve
        # stored comparisons, however one-sided, do not stop the run.
        ds, ocr = make_ds(n=40, models=("a", "b"))
        order = cli.judging_order(cli.sample_indices(40, None, 42), 42)
        existing = [
            ComparisonResult(sample_idx=page, model_a="model-a", model_b="model-b", winner="A")
            for page in order[:12]
        ]
        judge, _, _ = _run_judge([*self.FLAGS], ds, ocr, existing=existing, judge=NameOrderJudge())
        assert judge.judged > 0

    def test_rerun_with_an_added_model_still_judges(self, capsys):
        # The stored board is converged for a, b and c. The new model d has no
        # comparisons and so no interval: the stopping rule is not met.
        existing = self._converged_board()
        ds, ocr = make_ds(n=40, models=("a", "b", "c", "d"))
        capsys.readouterr()

        judge, _, _ = _run_judge([*self.FLAGS], ds, ocr, existing=existing, judge=NameOrderJudge())

        assert judge.judged > 0
        assert all("model-d" in pair for pair in judge.pairs_seen[:15])
        assert "already meet the adaptive stopping criteria" not in capsys.readouterr().out

    def test_rerun_of_an_unresolved_board_still_judges(self):
        # All ties: the intervals overlap, so the rule is not met on the stored
        # comparisons and the run judges the pages that are left.
        ds, ocr = make_ds(n=10, models=("a", "b", "c"))
        existing = [
            ComparisonResult(sample_idx=page, model_a=left, model_b=right, winner="tie")
            for page in range(8)
            for left, right in (
                ("model-a", "model-b"),
                ("model-a", "model-c"),
                ("model-b", "model-c"),
            )
        ]

        judge, _, _ = _run_judge(
            [*self.FLAGS], ds, ocr, existing=existing, judge=FakeJudge(winner="tie")
        )

        assert judge.judged == 6  # the 2 pages not yet judged, 3 pairs each


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

    @pytest.mark.parametrize("adaptive", [True, False])
    def test_budget_stop_reaches_both_halves_of_a_sorted_dataset(self, adaptive):
        # 40 pages x 3 pairs; a budget of 30 judges about 10 pages. Sorted
        # datasets (by collection or decade) are common, so the judged pages
        # must come from the whole sample, not only from its start.
        ds, ocr = make_ds(n=40)
        flags = ["--max-comparisons", "30", "--checkpoint-every", "0"]
        if not adaptive:
            flags.insert(0, "--no-adaptive")
        judge, _, _ = _run_judge(flags, ds, ocr, judge=FakeJudge(winner="tie"))
        assert judge.judged == 30
        pages = {sample for _, sample in judge.pair_samples}
        assert any(page < 20 for page in pages)
        assert any(page >= 20 for page in pages)

    @pytest.mark.parametrize("adaptive", [True, False])
    def test_budget_stop_keeps_no_auto_tie_from_later_pages(self, adaptive):
        # model-a and model-b give the same text on every second page of the
        # visiting order, so that pair is auto-tied there with no judge call.
        # A budget of 7 is used up on the third page visited:
        #   page 1: 3 judged; page 2: 1 auto-tie + 2 judged; page 3: 2 judged.
        ds, ocr = make_ds(n=40)
        order = cli.judging_order(cli.sample_indices(40, None, 42), 42)
        for page in order[1::2]:
            ds._columns["col_b"][page] = ds._columns["col_a"][page]
        flags = ["--max-comparisons", "7", "--checkpoint-every", "0"]
        if not adaptive:
            flags.insert(0, "--no-adaptive")

        judge, m_publish, _ = _run_judge(flags, ds, ocr, judge=FakeJudge(winner="tie"))

        assert judge.judged == 7
        position = {page: n for n, page in enumerate(order)}
        last_judged = max(position[page] for _, page in judge.pair_samples)
        assert last_judged == 2
        stored = m_publish.call_args.args[1].comparison_log
        assert len(stored) == 8
        assert all(position[row["sample_idx"]] <= last_judged for row in stored)
        auto_tied_pages = [row["sample_idx"] for row in stored if row["agreement"] == "auto"]
        assert auto_tied_pages == [order[1]]
        meta = _published_metadata(m_publish)
        assert meta.budget_exhausted is True
        assert meta.total_comparisons == 7
        assert meta.auto_tied == 1
        assert meta.valid_comparisons == 7

    @pytest.mark.parametrize("adaptive", [True, False])
    def test_runs_stopped_at_a_budget_end_with_the_comparisons_of_one_run(self, adaptive):
        # Auto-ties dropped at a budget stop are recorded by a later run: a
        # chain of budgeted runs ends with the rows of one run with no budget.
        def dataset():
            ds, ocr = make_ds(n=12)
            order = cli.judging_order(cli.sample_indices(12, None, 42), 42)
            for page in order[1::2]:
                ds._columns["col_b"][page] = ds._columns["col_a"][page]
            return ds, ocr

        def stored(m_publish):
            return [
                ComparisonResult(
                    sample_idx=row["sample_idx"],
                    model_a=row["model_a"],
                    model_b=row["model_b"],
                    winner=row["winner"],
                    agreement=row["agreement"],
                )
                for row in m_publish.call_args.args[1].comparison_log
            ]

        def keys(results):
            return sorted((r.sample_idx, r.model_a, r.model_b, r.agreement) for r in results)

        mode = [] if adaptive else ["--no-adaptive"]
        ds, ocr = dataset()
        _, one_run, _ = _run_judge(
            [*mode, "--checkpoint-every", "0"], ds, ocr, judge=FakeJudge(winner="tie")
        )
        expected = keys(stored(one_run))
        assert sum(1 for key in expected if key[3] == "auto") == 6

        existing: list[ComparisonResult] = []
        judge_calls = 0
        for _ in range(20):
            ds, ocr = dataset()
            judge, m_publish, _ = _run_judge(
                [*mode, "--max-comparisons", "4", "--checkpoint-every", "0"],
                ds,
                ocr,
                existing=existing,
                judge=FakeJudge(winner="tie"),
            )
            judge_calls += judge.judged
            existing = stored(m_publish)
            if judge.judged == 0:
                break
        assert keys(existing) == expected
        assert judge_calls == sum(1 for key in expected if key[3] != "auto")

    def test_no_budget_judges_everything(self):
        ds, ocr = make_ds(n=10)
        judge, m_publish, _ = _run_judge(["--no-adaptive", "--checkpoint-every", "0"], ds, ocr)
        assert judge.judged == 30
        assert _published_metadata(m_publish).budget_exhausted is False

    def test_budget_exactly_filled_by_last_batch_marks_exhausted(self):
        # 5 samples x 3 pairs = 15 comparisons in a single adaptive batch;
        # cap = 15 lands EXACTLY on the budget without tripping the trim/break.
        # The post-loop check must still mark it exhausted.
        ds, ocr = make_ds(n=5)
        judge, m_publish, _ = _run_judge(
            ["--max-comparisons", "15", "--checkpoint-every", "0"], ds, ocr
        )
        assert judge.judged == 15
        assert _published_metadata(m_publish).budget_exhausted is True


class TestFailedModelStatus:
    def test_fully_sentinel_model_is_marked_failed(self):
        ds, ocr = make_ds(n=10)
        ds._columns["col_a"] = ["[OCR ERROR]"] * 10

        _, m_publish, _ = _run_judge(
            ["--no-adaptive", "--checkpoint-every", "0"],
            ds,
            ocr,
        )

        board = m_publish.call_args.args[1]
        metadata = _published_metadata(m_publish)
        assert "model-a" not in board.elo
        assert metadata.failed_models == ["model-a"]
        assert metadata.failed_outputs == {"model-a": 10}

    @pytest.mark.parametrize("flags", [[], ["--max-samples", "10"], ["--max-samples", "50"]])
    def test_whole_dataset_run_counts_every_row(self, flags):
        # Without --max-samples below the dataset length, the judged sample is
        # every row, so the counts are those of the whole dataset.
        ds, ocr = make_ds(n=10)
        ds._columns["col_a"] = ["[OCR ERROR]"] * 10
        ds._columns["col_b"][3] = "[OCR ERROR]"
        ds._columns["col_b"][7] = "[OCR ERROR]"

        _, m_publish, _ = _run_judge(["--no-adaptive", "--checkpoint-every", "0", *flags], ds, ocr)

        metadata = _published_metadata(m_publish)
        assert metadata.failed_models == ["model-a"]
        assert metadata.failed_outputs == {"model-a": 10, "model-b": 2}

    def test_status_and_counts_follow_the_judged_sample(self, capsys):
        # 40 rows, 10 of them judged. model-x has an error sentinel on every
        # judged row and text on the others. model-y is the other way round.
        from ocr_bench.publish import build_leaderboard_rows

        ds, ocr = make_ds(n=40, models=("a", "b", "x", "y"))
        sample = set(cli.sample_indices(40, 10, 42))
        for row in range(40):
            failing_column = "col_x" if row in sample else "col_y"
            ds._columns[failing_column][row] = "[OCR ERROR]"

        _, m_publish, _ = _run_judge(
            ["--no-adaptive", "--checkpoint-every", "0", "--max-samples", "10"], ds, ocr
        )

        board = m_publish.call_args.args[1]
        metadata = _published_metadata(m_publish)
        assert metadata.failed_models == ["model-x"]
        assert metadata.failed_outputs == {"model-x": 10}
        assert metadata.failed_outputs_scope == "sample"
        assert "model-x" not in board.elo
        rows = {
            row["model"]: row
            for row in build_leaderboard_rows(
                board, metadata.failed_models, metadata.failed_outputs
            )
        }
        assert rows["model-x"]["status"] == "failed"
        assert rows["model-x"]["elo"] is None
        assert rows["model-y"]["status"] == "ranked"
        assert rows["model-y"]["failed_outputs"] == 0
        assert rows["model-y"]["wins"] + rows["model-y"]["losses"] + rows["model-y"]["ties"] == 20
        # The counts agree with the coverage report, which is per sample.
        for model, entry in metadata.model_coverage.items():
            assert entry["sentinel_pages"] == metadata.failed_outputs.get(model, 0)
        assert metadata.model_coverage["model-x"]["ranked_pages"] == 0
        assert metadata.model_coverage["model-y"]["ranked_pages"] == 10
        out = " ".join(capsys.readouterr().out.split())  # undo console line wrapping
        assert "model-x: 10/10 outputs in the judged sample are error sentinels (100%)" in out
        assert "this run is marked FAILED" in out
        assert "model-y:" not in out

    def test_model_failed_on_the_sample_does_not_block_adaptive_stopping(self):
        # model-x has text only on rows outside the judged sample. In the model
        # grid it would have no comparison and no interval, and a model with no
        # interval keeps the adaptive stopping rule from being met.
        ds, ocr = make_ds(n=80, models=("a", "b", "c", "x"))
        for row in cli.sample_indices(80, 40, 42):
            ds._columns["col_x"][row] = "[OCR ERROR]"

        judge, m_publish, _ = _run_judge(
            ["--max-samples", "40", "--checkpoint-every", "0"], ds, ocr, judge=NameOrderJudge()
        )

        assert _published_metadata(m_publish).failed_models == ["model-x"]
        # Two batches of 5 pages x 3 pairs, then the rule is met.
        assert judge.judged == 30


def _prose_ds(n: int, lengths: dict[str, int]):
    """A FakeDataset whose outputs are distinct prose of a set length per model."""
    import random
    import string

    cols = {}
    for m, length in lengths.items():
        rows = []
        for i in range(n):
            rng = random.Random(f"{m}-{i}")
            alphabet = string.ascii_lowercase + "     "
            rows.append("".join(rng.choice(alphabet) for _ in range(length)))
        cols[f"col_{m}"] = rows
    return FakeDataset(n, cols), {f"col_{m}": f"model-{m}" for m in lengths}


class TestCoverageReport:
    """judge reports on how many pages of the sample each model is ranked."""

    def test_metadata_records_ranked_pages_and_error_pages(self, capsys):
        ds, ocr = make_ds(n=10)
        ds._columns["col_b"][3] = "[OCR ERROR]"  # 10%: no warning, but a note

        _, m_publish, _ = _run_judge(["--no-adaptive", "--checkpoint-every", "0"], ds, ocr)

        metadata = _published_metadata(m_publish)
        assert metadata.model_coverage == {
            "model-a": {"sampled_pages": 10, "ranked_pages": 10, "sentinel_pages": 0},
            "model-b": {"sampled_pages": 10, "ranked_pages": 9, "sentinel_pages": 1},
            "model-c": {"sampled_pages": 10, "ranked_pages": 10, "sentinel_pages": 0},
        }
        out = " ".join(capsys.readouterr().out.split())  # undo console line wrapping
        assert "Note: model-b: 1/10 outputs in the judged sample are error sentinels (10.0%)" in out
        assert "its rank leaves out those pages" in out
        assert "9/10" in out  # the Pages column
        assert "model-b: ranked on 9 of 10 pages; 1 error sentinel(s) left out" in out

    def test_coverage_follows_max_samples(self):
        ds, ocr = make_ds(n=10)
        _, m_publish, _ = _run_judge(
            ["--no-adaptive", "--checkpoint-every", "0", "--max-samples", "4"], ds, ocr
        )
        coverage = _published_metadata(m_publish).model_coverage
        assert {entry["sampled_pages"] for entry in coverage.values()} == {4}
        assert {entry["ranked_pages"] for entry in coverage.values()} == {4}

    def test_failed_model_is_ranked_on_no_pages(self):
        ds, ocr = make_ds(n=10)
        ds._columns["col_a"] = ["[OCR ERROR]"] * 10
        _, m_publish, _ = _run_judge(["--no-adaptive", "--checkpoint-every", "0"], ds, ocr)
        coverage = _published_metadata(m_publish).model_coverage
        assert coverage["model-a"] == {
            "sampled_pages": 10,
            "ranked_pages": 0,
            "sentinel_pages": 10,
        }

    def test_high_sentinel_rate_keeps_the_warning_not_the_note(self, capsys):
        ds, ocr = make_ds(n=10)
        ds._columns["col_b"][0] = "[OCR ERROR]"
        ds._columns["col_b"][1] = "[OCR ERROR]"  # 20%
        _run_judge(["--no-adaptive", "--checkpoint-every", "0"], ds, ocr)
        out = " ".join(capsys.readouterr().out.split())
        assert "model-b: 2/10 outputs in the judged sample are error sentinels (20%)" in out
        assert "Note: model-b" not in out


class TestTextCapWarning:
    """judge warns, before judging, when the cap truncates over half the outputs."""

    def test_warns_with_the_screen_suggestion(self, capsys):
        ds, ocr = _prose_ds(10, {"a": 300, "b": 320, "c": 340})
        _, m_publish, _ = _run_judge(
            ["--no-adaptive", "--checkpoint-every", "0", "--max-ocr-text-len", "100"], ds, ocr
        )
        out = " ".join(capsys.readouterr().out.split())
        assert "--max-ocr-text-len 100 truncates 30 of 30 outputs (100%)" in out
        assert "Over half are truncated for: model-a (100%), model-b (100%), model-c (100%)" in out
        # The screen rounds the longest unflagged output (340) up to 500.
        assert "--max-ocr-text-len 500 fits every output" in out
        assert "This run keeps 100." in out
        # The default and this run's cap are unchanged.
        assert _published_metadata(m_publish).max_ocr_text_len == 100

    def test_one_model_over_half_is_enough(self, capsys):
        ds, ocr = _prose_ds(10, {"a": 60, "b": 60, "c": 60, "d": 300})
        _run_judge(
            ["--no-adaptive", "--checkpoint-every", "0", "--max-ocr-text-len", "100"], ds, ocr
        )
        out = " ".join(capsys.readouterr().out.split())
        assert "truncates 10 of 40 outputs (25%)" in out
        assert "Over half are truncated for: model-d (100%)" in out

    def test_no_warning_when_half_or_fewer_are_truncated(self, capsys):
        ds, ocr = _prose_ds(10, {"a": 60, "b": 60, "c": 60})
        long_ds, _ = _prose_ds(10, {"b": 300})
        # Exactly half of model-b's outputs are over the cap: not more than half.
        ds._columns["col_b"][:5] = long_ds._columns["col_b"][:5]
        _run_judge(
            ["--no-adaptive", "--checkpoint-every", "0", "--max-ocr-text-len", "100"], ds, ocr
        )
        assert "Text cap" not in capsys.readouterr().out

    def test_flagged_long_output_does_not_claim_a_higher_cap_cannot_help(self, capsys):
        # One complete 3,000-character transcription next to two 400-character
        # partial ones: the screen flags the complete one as >5x length, so it
        # suggests no cap. That flag is a lead, not proof that the text is wrong.
        ds, ocr = _prose_ds(5, {"a": 3000, "b": 400, "c": 400})
        _run_judge(["--no-adaptive", "--checkpoint-every", "0"], ds, ocr)
        out = " ".join(capsys.readouterr().out.split())
        assert "--max-ocr-text-len 2500 truncates 5 of 15 outputs (33%)" in out
        assert "Over half are truncated for: model-a (100%)" in out
        assert "--max-ocr-text-len 2500 fits every output that the `audit --screen` checks" in out
        assert "Every output over the cap is flagged" in out
        assert "`ocr-bench audit --screen --json` lists the flagged rows" in out
        assert "would not help" not in out

    def test_two_models_get_no_suggestion(self, capsys):
        ds, ocr = _prose_ds(10, {"a": 300, "b": 320})
        _run_judge(
            ["--no-adaptive", "--checkpoint-every", "0", "--max-ocr-text-len", "100"], ds, ocr
        )
        out = " ".join(capsys.readouterr().out.split())
        assert "truncates 20 of 20 outputs" in out
        assert "No cap is suggested: the screen checks need outputs from 3 or more models" in out


class TestCheckpointFullRejudge:
    def test_default_disabled_under_full_rejudge(self, capsys):
        # No explicit --checkpoint-every + --full-rejudge -> checkpointing off,
        # with an explanatory message (avoids clobbering complete published data).
        ds, ocr = make_ds(n=10)
        _, _, m_checkpoint = _run_judge(["--no-adaptive", "--full-rejudge"], ds, ocr)
        m_checkpoint.assert_not_called()
        assert "Checkpointing off under --full-rejudge" in capsys.readouterr().out

    def test_default_enabled_without_full_rejudge(self, capsys):
        # Sanity: the disabled message is specific to --full-rejudge.
        ds, ocr = make_ds(n=10)
        _run_judge(["--no-adaptive"], ds, ocr)
        assert "Checkpointing off under --full-rejudge" not in capsys.readouterr().out

    def test_explicit_override_honored_with_warning(self, capsys):
        # Explicit --checkpoint-every N>0 with --full-rejudge is honored but warns.
        ds, ocr = make_ds(n=10)
        _, _, m_checkpoint = _run_judge(
            ["--no-adaptive", "--full-rejudge", "--checkpoint-every", "10"], ds, ocr
        )
        assert m_checkpoint.call_count == 3  # honored despite full-rejudge
        assert "WARNING" in capsys.readouterr().out

    def test_explicit_zero_no_warning_under_full_rejudge(self, capsys):
        ds, ocr = make_ds(n=10)
        _, _, m_checkpoint = _run_judge(
            ["--no-adaptive", "--full-rejudge", "--checkpoint-every", "0"], ds, ocr
        )
        m_checkpoint.assert_not_called()
        assert "WARNING" not in capsys.readouterr().out


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
        _, _, m_checkpoint = _run_judge(["--no-adaptive", "--checkpoint-every", "0"], ds, ocr)
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

    def test_subset_run_preserves_out_of_grid_history_in_checkpoints_and_final_publish(self):
        ds, ocr = make_ds(n=3, models=("a", "b"))
        current = ComparisonResult(0, "model-a", "model-b", "A")
        stale = ComparisonResult(0, "model-a", "retired-model", "B")

        _, m_publish, m_checkpoint = _run_judge(
            ["--no-adaptive", "--checkpoint-every", "1"],
            ds,
            ocr,
            existing=[current, stale],
        )

        assert m_checkpoint.call_count == 2
        for call in m_checkpoint.call_args_list:
            checkpoint_results = call.args[1]
            assert stale in checkpoint_results
            assert "retired-model" in call.args[2]
        assert m_publish.call_args.kwargs["preserved_comparisons"] == [stale]
        board = m_publish.call_args.args[1]
        assert all(
            row["model_a"] != "retired-model" and row["model_b"] != "retired-model"
            for row in board.comparison_log
        )

    def test_failed_model_history_is_preserved_but_excluded_from_fit(self):
        ds, ocr = make_ds(n=3, models=("a", "b", "c"))
        ds._columns["col_c"] = ["[OCR ERROR]"] * 3
        current = ComparisonResult(0, "model-a", "model-b", "A")
        historical_failed = ComparisonResult(0, "model-a", "model-c", "B")

        _, m_publish, m_checkpoint = _run_judge(
            ["--no-adaptive", "--checkpoint-every", "1"],
            ds,
            ocr,
            existing=[current, historical_failed],
        )

        assert all(historical_failed in call.args[1] for call in m_checkpoint.call_args_list)
        assert m_publish.call_args.kwargs["preserved_comparisons"] == [historical_failed]
        board = m_publish.call_args.args[1]
        assert "model-c" not in board.elo
        assert all(
            row["model_a"] != "model-c" and row["model_b"] != "model-c"
            for row in board.comparison_log
        )

    def test_checkpoints_fire_in_adaptive_mode(self):
        # Adaptive checkpoints at batch boundaries: batch of 5 samples x 3 pairs
        # = 15 comparisons; with K=5 each batch crosses the threshold once.
        ds, ocr = make_ds(n=10)
        _, _, m_checkpoint = _run_judge(
            ["--checkpoint-every", "5"], ds, ocr, judge=FakeJudge(winner="tie")
        )
        assert m_checkpoint.call_count >= 1


class TestResume:
    @staticmethod
    def _saved_results(published):
        return [
            ComparisonResult(
                sample_idx=row["sample_idx"],
                model_a=row["model_a"],
                model_b=row["model_b"],
                winner=row["winner"],
                col_a=row["col_a"],
                col_b=row["col_b"],
                provenance_hash=row["provenance_hash"],
            )
            for row in published.call_args.args[1].comparison_log
        ]

    @staticmethod
    def _configs(ds, ocr, *, output_column="old_ocr", incomplete=False):
        configs = {}
        for column, model in ocr.items():
            entries = [{"column_name": output_column, "model_id": model}]
            if incomplete:
                entries.append({"model_id": "org/unfinished"})
            configs[column] = Dataset.from_dict(
                {
                    "id": list(range(len(ds))),
                    "image": ds["image"],
                    # Equal contents make the original column name the only change.
                    "old_ocr": ds[column],
                    "new_ocr": ds[column],
                    "inference_info": [json.dumps(entries)] * len(ds),
                }
            )
        return configs

    @pytest.mark.parametrize("change", ["unchanged", "output_column", "incomplete_entry"])
    def test_config_resolution_controls_resume_provenance(self, change, capsys):
        ds, ocr = make_ds(n=2, models=("a", "b"))
        flags = ["--no-adaptive", "--checkpoint-every", "0"]
        _, published, _ = _run_judge(
            flags,
            ds,
            ocr,
            stamp_existing_provenance=False,
            config_datasets=self._configs(ds, ocr),
        )
        assert _published_metadata(published).source_output_columns == {
            column: "old_ocr" for column in ocr
        }
        configs = self._configs(
            ds,
            ocr,
            output_column="new_ocr" if change == "output_column" else "old_ocr",
            incomplete=change == "incomplete_entry",
        )
        judge = FakeJudge()
        with pytest.raises(SystemExit) if change == "output_column" else nullcontext():
            _run_judge(
                flags,
                ds,
                ocr,
                existing=self._saved_results(published),
                judge=judge,
                stamp_existing_provenance=False,
                checkpoint_resume=True,
                config_datasets=configs,
            )
        assert judge.judged == 0
        if change == "output_column":
            assert "resume provenance mismatch" in capsys.readouterr().out

    def test_new_config_reuses_unchanged_pair_with_real_provenance(self):
        ds, ocr = make_ds(n=2, models=("a", "b"))
        flags = ["--no-adaptive", "--checkpoint-every", "0"]
        _, published, _ = _run_judge(
            flags,
            ds,
            ocr,
            stamp_existing_provenance=False,
            config_datasets=self._configs(ds, ocr),
        )
        ds, ocr = make_ds(n=2, models=("a", "b", "c"))
        judge, _, _ = _run_judge(
            flags,
            ds,
            ocr,
            existing=self._saved_results(published),
            stamp_existing_provenance=False,
            checkpoint_resume=True,
            config_datasets=self._configs(ds, ocr),
        )
        assert judge.judged == 4
        assert all("model-c" in pair for pair in judge.pairs_seen)

    @pytest.mark.parametrize(
        "change",
        [
            "ocr_text",
            "judge",
            "split",
            "criteria",
            "criteria_file",
            "text_cap",
            "text_mode",
            "image_dim",
        ],
    )
    def test_changed_inputs_reject_real_checkpoint_provenance(self, change, capsys, tmp_path):
        ds, ocr = make_ds(n=2, models=("a", "b"))
        flags = ["--no-adaptive", "--checkpoint-every", "0"]
        rubric = tmp_path / "rubric.txt"
        if change == "criteria_file":
            rubric.write_text("Custom rubric. A={ocr_text_a} B={ocr_text_b}")
            flags += ["--criteria-file", str(rubric)]
        _, published, _ = _run_judge(flags, ds, ocr, stamp_existing_provenance=False)
        if change == "ocr_text":
            ds._columns["col_a"][0] = "Changed OCR transcription for the same source page"
        elif change == "judge":
            flags += ["--model", "different-judge"]
        elif change == "criteria":
            flags += ["--criteria", "table-fidelity"]
        elif change == "criteria_file":
            # Same file name, new content: the prompt hash changes.
            rubric.write_text("A NEW rubric. A={ocr_text_a} B={ocr_text_b}")
        elif change == "text_cap":
            flags += ["--max-ocr-text-len", "5000"]
        elif change == "text_mode":
            flags += ["--judge-text-mode", "raw"]
        elif change == "image_dim":
            flags += ["--judge-image-dim", "2048"]
        else:
            flags += ["--split", "validation"]
        judge = FakeJudge()
        with pytest.raises(SystemExit):
            _run_judge(
                flags,
                ds,
                ocr,
                existing=self._saved_results(published),
                judge=judge,
                stamp_existing_provenance=False,
                checkpoint_resume=True,
            )
        assert judge.judged == 0
        assert "resume provenance mismatch" in capsys.readouterr().out

    def test_default_judge_change_names_previous_judge(self, capsys):
        """A board judged with an earlier default fails closed and names that judge."""
        import json

        from ocr_bench.publish import hash_judge_specs

        old = "novita:org/old-default"
        ds, ocr = make_ds(n=2, models=("a", "b"))
        flags = ["--no-adaptive", "--checkpoint-every", "0"]
        _, published, _ = _run_judge(
            [*flags, "--model", old], ds, ocr, stamp_existing_provenance=False
        )
        judge = FakeJudge()
        with pytest.raises(SystemExit):
            _run_judge(
                flags,
                ds,
                ocr,
                existing=self._saved_results(published),
                judge=judge,
                stamp_existing_provenance=False,
                checkpoint_resume=True,
                existing_meta_override=[
                    {
                        "judge_models": json.dumps([old]),
                        "judge_spec_hashes": json.dumps(hash_judge_specs([old])),
                    }
                ],
            )
        assert judge.judged == 0
        out = " ".join(capsys.readouterr().out.split())  # undo console line wrapping
        assert "resume provenance mismatch" in out
        assert f"--model {old}" in out
        assert f"--judge-model {old}" in out

    def test_comparisons_only_checkpoint_resumes_from_row_provenance(self):
        ds, ocr = make_ds(n=4, models=("a", "b"))
        _, first_publish, _ = _run_judge(
            ["--no-adaptive", "--checkpoint-every", "0"],
            ds,
            ocr,
            stamp_existing_provenance=False,
        )
        rows = first_publish.call_args.args[1].comparison_log[:2]
        existing = [
            ComparisonResult(
                sample_idx=row["sample_idx"],
                model_a=row["model_a"],
                model_b=row["model_b"],
                winner=row["winner"],
                col_a=row["col_a"],
                col_b=row["col_b"],
                provenance_hash=row["provenance_hash"],
            )
            for row in rows
        ]

        judge, _, _ = _run_judge(
            ["--no-adaptive", "--checkpoint-every", "0"],
            ds,
            ocr,
            existing=existing,
            stamp_existing_provenance=False,
            checkpoint_resume=True,
        )
        assert judge.judged == 2

    def test_missing_provenance_requires_full_rejudge(self, capsys):
        ds, ocr = make_ds(n=2, models=("a", "b"))
        existing = [
            ComparisonResult(
                sample_idx=0,
                model_a="model-a",
                model_b="model-b",
                winner="A",
                col_a="col_a",
                col_b="col_b",
            )
        ]
        with pytest.raises(SystemExit) as exc:
            _run_judge(
                ["--no-adaptive", "--checkpoint-every", "0"],
                ds,
                ocr,
                existing=existing,
                stamp_existing_provenance=False,
            )
        assert exc.value.code == 1
        assert "lack resume provenance" in capsys.readouterr().out

    def test_mismatched_provenance_requires_full_rejudge(self, capsys):
        ds, ocr = make_ds(n=2, models=("a", "b"))
        existing = [
            ComparisonResult(
                sample_idx=0,
                model_a="model-a",
                model_b="model-b",
                winner="A",
                col_a="col_a",
                col_b="col_b",
                provenance_hash="different-run",
            )
        ]
        with pytest.raises(SystemExit) as exc:
            _run_judge(
                ["--no-adaptive", "--checkpoint-every", "0"],
                ds,
                ocr,
                existing=existing,
                stamp_existing_provenance=False,
            )
        assert exc.value.code == 1
        assert "resume provenance mismatch" in capsys.readouterr().out

    def test_matching_checkpoint_hash_overrides_stale_completed_metadata(self):
        ds, ocr = make_ds(n=2, models=("a", "b"))
        existing = [
            ComparisonResult(
                sample_idx=0,
                model_a="model-a",
                model_b="model-b",
                winner="A",
                col_a="col_a",
                col_b="col_b",
            )
        ]
        stale_metadata = [
            {
                "criteria": "default",
                "prompt_hash": prompt_hash(CRITERIA_PROFILES["default"]),
                "judge_text_mode": "raw",
                "max_ocr_text_len": 1000,
                "judge_image_dim": 512,
            }
        ]

        judge, m_publish, _ = _run_judge(
            [
                "--criteria",
                "table-fidelity",
                "--no-adaptive",
                "--checkpoint-every",
                "0",
            ],
            ds,
            ocr,
            existing=existing,
            existing_meta_override=stale_metadata,
        )

        assert judge.judged == 1
        m_publish.assert_called_once()

    def test_resume_tops_up_partial_pairs(self):
        # A prior (checkpointed) run judged (model-a, model-b) on samples 0-3
        # only. Relaunch WITHOUT --full-rejudge: (pair, sample)-level skip means
        # (a,b) is topped up on samples 4-9, and (a,c)/(b,c) run on all 10.
        ds, ocr = make_ds(n=10)
        existing = [
            ComparisonResult(sample_idx=i, model_a="model-a", model_b="model-b", winner="A")
            for i in range(4)
        ]
        judge, m_publish, _ = _run_judge(
            ["--no-adaptive", "--checkpoint-every", "0"],
            ds,
            ocr,
            existing=existing,
        )
        # (a,b): 6 remaining samples + (a,c),(b,c): 10 each = 26.
        assert judge.judged == 26
        # (a,b) is topped up, not frozen — but only on the not-yet-judged samples.
        ab = ("model-a", "model-b")
        ab_samples = {s for (pair, s) in judge.pair_samples if pair == ab}
        assert ab_samples == {4, 5, 6, 7, 8, 9}
        m_publish.assert_called_once()

    def test_resume_fully_judged_pair_not_rejudged(self):
        # A pair judged on ALL samples is skipped entirely (nothing to top up).
        ds, ocr = make_ds(n=10)
        existing = [
            ComparisonResult(sample_idx=i, model_a="model-a", model_b="model-b", winner="A")
            for i in range(10)
        ]
        judge, _, _ = _run_judge(
            ["--no-adaptive", "--checkpoint-every", "0"], ds, ocr, existing=existing
        )
        # Only (a,c) and (b,c) remain: 20.
        assert judge.judged == 20
        assert ("model-a", "model-b") not in judge.pairs_seen

    def test_full_rejudge_ignores_existing(self):
        ds, ocr = make_ds(n=10)
        existing = [
            ComparisonResult(sample_idx=i, model_a="model-a", model_b="model-b", winner="A")
            for i in range(4)
        ]
        judge, _, _ = _run_judge(
            ["--no-adaptive", "--full-rejudge", "--checkpoint-every", "0"],
            ds,
            ocr,
            existing=existing,
        )
        # --full-rejudge drops the skip map, so all 3 pairs x 10 samples = 30.
        assert judge.judged == 30
        assert ("model-a", "model-b") in judge.pairs_seen


class TestFullRejudge:
    """--full-rejudge never loads existing results, so changed settings run."""

    _DEFAULT_HASH = prompt_hash(CRITERIA_PROFILES["default"])

    def _run(self, argv_extra, *, existing, existing_meta):
        """Drive cmd_judge with a 2-model dataset, catching a guard SystemExit."""
        ds, ocr = make_ds(n=4, models=("a", "b"))
        judge = FakeJudge()
        argv = [
            "judge",
            "user/ds",
            "--columns",
            *ocr.keys(),
            "--save-results",
            "user/results",
            "--no-adaptive",
            "--checkpoint-every",
            "0",
            *argv_extra,
        ]
        args = build_parser().parse_args(argv)
        for result in existing:
            result.col_a = result.col_a or "col_a"
            result.col_b = result.col_b or "col_b"
        exit_code: int | str | None = None
        with (
            patch.object(cli, "load_flat_dataset", return_value=(ds, ocr)),
            patch.object(cli, "parse_judge_spec", return_value=judge),
            patch.object(cli, "load_existing_comparisons", return_value=existing),
            patch.object(cli, "load_existing_metadata", return_value=existing_meta),
            patch.object(cli, "publish_results") as m_publish,
            patch.object(cli, "publish_checkpoint"),
        ):
            try:
                cli.cmd_judge(args)
            except SystemExit as exc:
                exit_code = exc.code
        return judge, m_publish, exit_code

    def _existing_one_pair(self):
        # (model-a, model-b) judged on sample 0 only.
        return [ComparisonResult(sample_idx=0, model_a="model-a", model_b="model-b", winner="A")]

    def test_full_rejudge_bypasses_guard(self):
        # Metadata says default, run requests table-fidelity — normally blocked,
        # but --full-rejudge never loads existing results, so no guard fires.
        _, m_publish, code = self._run(
            ["--criteria", "table-fidelity", "--full-rejudge"],
            existing=self._existing_one_pair(),
            existing_meta=[{"criteria": "default", "prompt_hash": self._DEFAULT_HASH}],
        )
        assert code is None
        m_publish.assert_called_once()
        assert m_publish.call_args.args[2].criteria == "table-fidelity"


class OrderRecordingJudge(FakeJudge):
    """A judge that records which way round it was shown each comparison."""

    def __init__(self):
        super().__init__(winner="tie")
        self.saw_model_b_first: dict[tuple[int, str, str], bool] = {}

    def judge(self, comparisons):
        for comp in comparisons:
            self.saw_model_b_first[(comp.sample_idx, comp.model_a, comp.model_b)] = comp.swapped
        return super().judge(comparisons)


def _published_display_order(m_publish) -> dict[tuple[int, str, str], str]:
    """The ``shown_first`` value of each comparison row passed to publish_results."""
    board = m_publish.call_args.args[1]
    return {
        (row["sample_idx"], row["model_a"], row["model_b"]): row["shown_first"]
        for row in board.comparison_log
    }


class TestDisplayOrderAcrossRuns:
    """The display order of a comparison does not depend on how the run was built."""

    MODELS = ("a", "b", "c", "d")

    def _run(self, flags, **kwargs):
        ds, ocr = make_ds(n=40, models=self.MODELS)
        judge = OrderRecordingJudge()
        _, m_publish, _ = _run_judge(
            [*flags, "--checkpoint-every", "0"], ds, ocr, judge=judge, **kwargs
        )
        return judge, m_publish

    def test_adaptive_batches_match_a_single_pass(self):
        adaptive_judge, adaptive_publish = self._run([])
        single_judge, single_publish = self._run(["--no-adaptive"])
        assert len(adaptive_judge.saw_model_b_first) == 40 * 6
        assert adaptive_judge.saw_model_b_first == single_judge.saw_model_b_first
        assert _published_display_order(adaptive_publish) == _published_display_order(
            single_publish
        )

    def test_published_rows_record_what_the_judge_saw(self):
        judge, m_publish = self._run([])
        published = _published_display_order(m_publish)
        assert set(published.values()) == {"A", "B"}
        assert published == {
            key: "B" if model_b_first else "A"
            for key, model_b_first in judge.saw_model_b_first.items()
        }

    def test_adaptive_batches_do_not_replay_one_order(self):
        # 8 batches of 5 pages. Replaying one order per batch shows each pair
        # the same way round on every 5th page: only 5 distinct decisions.
        judge, _ = self._run([])
        ds_order = cli.judging_order(cli.sample_indices(40, None, 42), 42)
        position_in_batch = {page: n % 5 for n, page in enumerate(ds_order)}
        replayed = 0
        for model_a, model_b in [("model-a", "model-b"), ("model-c", "model-d")]:
            by_position: dict[int, set[bool]] = {}
            for (page, a, b), model_b_first in judge.saw_model_b_first.items():
                if (a, b) == (model_a, model_b):
                    by_position.setdefault(position_in_batch[page], set()).add(model_b_first)
            replayed += all(len(seen) == 1 for seen in by_position.values())
        assert replayed == 0

    def test_resume_keeps_the_order_of_the_remaining_comparisons(self):
        full_judge, _ = self._run([])
        # A first run judged one pair on half of the pages.
        existing = [
            ComparisonResult(sample_idx=page, model_a="model-a", model_b="model-b", winner="tie")
            for page in range(20)
        ]
        resumed_judge, _ = self._run([], existing=existing)
        assert len(resumed_judge.saw_model_b_first) == 40 * 6 - 20
        assert all(
            full_judge.saw_model_b_first[key] == model_b_first
            for key, model_b_first in resumed_judge.saw_model_b_first.items()
        )

    @pytest.mark.parametrize(("flags", "adaptive"), [([], True), (["--no-adaptive"], False)])
    def test_metadata_records_run_mode_and_version(self, flags, adaptive):
        from ocr_bench import __version__

        _, m_publish = self._run(flags)
        metadata = _published_metadata(m_publish)
        assert metadata.adaptive is adaptive
        assert metadata.ocr_bench_version == __version__

    def test_a_refit_with_nothing_to_judge_also_records_run_mode_and_version(self):
        from ocr_bench import __version__

        ds, ocr = make_ds(n=4, models=("a", "b"))
        existing = [
            ComparisonResult(
                sample_idx=page, model_a="model-a", model_b="model-b", winner="A", shown_first="A"
            )
            for page in range(4)
        ]
        judge, m_publish, _ = _run_judge(
            ["--no-adaptive", "--checkpoint-every", "0"], ds, ocr, existing=existing
        )
        assert judge.judged == 0
        metadata = _published_metadata(m_publish)
        assert metadata.adaptive is False
        assert metadata.ocr_bench_version == __version__

    def test_resume_says_when_stored_comparisons_have_no_display_order(self, capsys):
        # Rows from a version that did not record the order: the user cannot
        # see from the board that their order may be unbalanced.
        existing = [
            ComparisonResult(sample_idx=page, model_a="model-a", model_b="model-b", winner="tie")
            for page in range(20)
        ]
        self._run([], existing=existing)
        out = " ".join(capsys.readouterr().out.split())  # undo console line wrapping
        assert "20 of them have no recorded display order" in out
        assert "--full-rejudge" in out

    def test_resume_is_quiet_when_every_judged_comparison_has_its_display_order(self, capsys):
        existing = [
            ComparisonResult(
                sample_idx=page, model_a="model-a", model_b="model-b", winner="tie", shown_first="A"
            )
            for page in range(19)
        ]
        # An auto-tie has no display order and is not a row from an older version.
        existing.append(
            ComparisonResult(
                sample_idx=19, model_a="model-a", model_b="model-b", winner="tie", agreement="auto"
            )
        )
        self._run([], existing=existing)
        assert "no recorded display order" not in " ".join(capsys.readouterr().out.split())
