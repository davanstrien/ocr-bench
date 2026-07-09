"""Tests for --max-comparisons budget and --checkpoint-every checkpointing.

These drive ``cli.cmd_judge`` end-to-end with a fake dataset + fake judge and
mocked Hub publish/checkpoint, so they exercise the real budget-trim, chunked
checkpoint, and resume-skip control flow without any network I/O.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from PIL import Image

from ocr_bench import cli
from ocr_bench.cli import build_parser
from ocr_bench.elo import ComparisonResult
from ocr_bench.judge import CRITERIA_PROFILES, _normalize_pair, prompt_hash


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


class TestCheckpointFullRejudge:
    def test_default_disabled_under_full_rejudge(self, capsys):
        # No explicit --checkpoint-every + --full-rejudge -> checkpointing off,
        # with an explanatory message (avoids clobbering complete published data).
        ds, ocr = make_ds(n=10)
        _, _, m_checkpoint = _run_judge(
            ["--no-adaptive", "--full-rejudge"], ds, ocr
        )
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
    def test_resume_tops_up_partial_pairs(self):
        # A prior (checkpointed) run judged (model-a, model-b) on samples 0-3
        # only. Relaunch WITHOUT --full-rejudge: (pair, sample)-level skip means
        # (a,b) is topped up on samples 4-9, and (a,c)/(b,c) run on all 10.
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
            ComparisonResult(
                sample_idx=i, model_a="model-a", model_b="model-b", winner="A"
            )
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
        # --full-rejudge drops the skip map, so all 3 pairs x 10 samples = 30.
        assert judge.judged == 30
        assert ("model-a", "model-b") in judge.pairs_seen


class TestCriteriaProvenanceGuard:
    """cmd_judge must refuse to mix criteria rubrics on one results repo (#44 review).

    Judging an existing results repo under a different --criteria than its
    comparisons were scored with would merge incompatible rubrics into one ELO
    board and mislabel the metadata. The guard exits before judging/publishing;
    --full-rejudge (which discards existing results) is the only safe rubric swap.
    """

    _DEFAULT_HASH = prompt_hash(CRITERIA_PROFILES["default"])
    _TABLE_HASH = prompt_hash(CRITERIA_PROFILES["table-fidelity"])

    def _run(self, argv_extra, *, existing, existing_meta):
        """Drive cmd_judge with a 2-model dataset, catching a guard SystemExit."""
        ds, ocr = make_ds(n=4, models=("a", "b"))
        judge = FakeJudge()
        argv = [
            "judge", "user/ds", "--columns", *ocr.keys(),
            "--save-results", "user/results", "--no-adaptive",
            "--checkpoint-every", "0", *argv_extra,
        ]
        args = build_parser().parse_args(argv)
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
        # (model-a, model-b) judged on sample 0 only, so a matching-criteria run
        # still has samples 1-3 to top up (proves it proceeds to judge+publish).
        return [
            ComparisonResult(sample_idx=0, model_a="model-a", model_b="model-b", winner="A")
        ]

    def test_mismatch_exits_without_judging_or_publishing(self):
        judge, m_publish, code = self._run(
            ["--criteria", "table-fidelity"],
            existing=self._existing_one_pair(),
            existing_meta=[{"criteria": "default", "prompt_hash": self._DEFAULT_HASH}],
        )
        assert code == 1
        assert judge.judged == 0  # exited before any judge call
        m_publish.assert_not_called()

    def test_pre_44_none_rows_treated_as_default_so_default_run_proceeds(self):
        # Genuinely pre-#44 metadata: no criteria/prompt_hash columns → default.
        _, m_publish, code = self._run(
            [],  # no --criteria → default
            existing=self._existing_one_pair(),
            existing_meta=[{"source_dataset": "user/ds"}],
        )
        assert code is None
        m_publish.assert_called_once()

    def test_matching_criteria_proceeds(self):
        _, m_publish, code = self._run(
            ["--criteria", "table-fidelity"],
            existing=self._existing_one_pair(),
            existing_meta=[{"criteria": "table-fidelity", "prompt_hash": self._TABLE_HASH}],
        )
        assert code is None
        m_publish.assert_called_once()
        assert m_publish.call_args.args[2].criteria == "table-fidelity"

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

    def test_same_custom_file_rerun_matches(self, tmp_path):
        # A repo judged under a custom prompt file, re-run with the same file:
        # identical hash → proceeds (guard compares hashes, not names).
        f = tmp_path / "rubric.txt"
        f.write_text("Custom rubric. A={ocr_text_a} B={ocr_text_b}")
        file_hash = prompt_hash(f.read_text())
        _, m_publish, code = self._run(
            ["--criteria-file", str(f)],
            existing=self._existing_one_pair(),
            existing_meta=[{"criteria": "custom:rubric.txt", "prompt_hash": file_hash}],
        )
        assert code is None
        m_publish.assert_called_once()
        assert m_publish.call_args.args[2].criteria == "custom:rubric.txt"

    def test_different_custom_content_blocks(self, tmp_path, capsys):
        # Same basename, DIFFERENT content → different hash → blocked, and the
        # error tells the user to re-supply the same file (not a --criteria name).
        f = tmp_path / "rubric.txt"
        f.write_text("A NEW rubric. A={ocr_text_a} B={ocr_text_b}")
        judge, m_publish, code = self._run(
            ["--criteria-file", str(f)],
            existing=self._existing_one_pair(),
            existing_meta=[{"criteria": "custom:rubric.txt", "prompt_hash": "0badc0ffee00"}],
        )
        assert code == 1
        assert judge.judged == 0
        m_publish.assert_not_called()
        assert "custom prompt file" in capsys.readouterr().out
