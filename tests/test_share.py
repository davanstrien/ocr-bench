"""`share`: a scores-only board from a results repo, with fail-closed privacy checks."""

from __future__ import annotations

import io
import json
import random
import re
import string
import sys
from itertools import combinations
from types import SimpleNamespace

import pytest
from datasets import Dataset, Image
from huggingface_hub import CommitOperationAdd, CommitOperationDelete
from huggingface_hub.errors import RepositoryNotFoundError
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from PIL import Image as PILImage

from ocr_bench import cli, dataset, share
from ocr_bench.dataset import evaluation_input_fingerprints, load_config_dataset
from ocr_bench.elo import ComparisonResult, compute_elo
from ocr_bench.judge import sample_indices
from ocr_bench.publish import build_leaderboard_rows
from ocr_bench.share import (
    CHUNK,
    PrivacyError,
    PrivateEvaluation,
    ShareError,
    check_no_page_text,
    check_plan,
    check_publish_order,
    load_private_evaluation,
    load_slice_values,
    plan_share,
    push_share,
    write_local,
)

MODELS = ["org/model-good", "org/model-mid", "org/model-weak"]
STRENGTH = {"org/model-good": 0.8, "org/model-mid": 0.5, "org/model-weak": 0.2}
NOW = "2026-11-06T23:00:00+00:00"
RESULTS = "me/private-results"
SOURCE = "me/private-source"


@pytest.fixture(autouse=True)
def _few_bootstrap_resamples(monkeypatch):
    # Each resample is a scipy fit. Ratings and counts do not depend on the resamples.
    monkeypatch.setattr(share, "N_BOOTSTRAP", 20)


def _results(pages, models=MODELS, *, pairs=None, text=None, reason=None, seed=0):
    """One comparison per model pair per page. Stronger models win more often."""
    rng = random.Random(seed)
    results = []
    for page in pages:
        for a, b in pairs or combinations(models, 2):
            p_a = STRENGTH.get(a, 0.5) / (STRENGTH.get(a, 0.5) + STRENGTH.get(b, 0.5))
            roll = rng.random()
            winner = "A" if roll < p_a - 0.05 else "B" if roll > p_a + 0.05 else "tie"
            results.append(
                ComparisonResult(
                    sample_idx=page,
                    model_a=a,
                    model_b=b,
                    winner=winner,
                    reason=reason(page) if reason else f"Output {winner} reads page {page} better",
                    text_a=text(page, a) if text else f"text of page {page} by {a}",
                    text_b=text(page, b) if text else f"text of page {page} by {b}",
                    provenance_hash="h",
                )
            )
    return results


def _metadata(n_rows, **overrides):
    metadata = {
        "source_dataset": SOURCE,
        "source_split": "train",
        "source_configs": json.dumps([]),
        "source_revisions": "{}",
        "source_output_columns": "{}",
        "judge_models": json.dumps(["novita:Qwen/Qwen3.8-27B"]),
        "seed": 42,
        "max_samples": n_rows,
        "adaptive_strategy": "balanced",
        "failed_models": "[]",
        "failed_outputs": "{}",
        "criteria": "body-text",
        "prompt_hash": "ab" * 32,
        "timestamp": "2026-11-06T12:00:00+00:00",
        "max_ocr_text_len": 8000,
        "judge_image_dim": 1024,
        "judge_text_mode": "normalized",
        "min_chars": 10,
        "source_fingerprint": "cd" * 32,
    }
    metadata.update(overrides)
    return metadata


def _evaluation(results, models=MODELS, *, n_rows=None, failed_models=(), failed_outputs=None):
    """A results repo as judge publishes it.

    Judge fits the board without failed models and appends them as FAILED rows.
    Comparisons with a model outside the fit stay in the repo as history.
    """
    fitted = [m for m in models if m not in failed_models]
    grid = set(fitted)
    board = compute_elo(
        [r for r in results if r.model_a in grid and r.model_b in grid], fitted, n_bootstrap=0
    )
    rows = build_leaderboard_rows(board, list(failed_models), failed_outputs or {})
    n_rows = n_rows or max(r.sample_idx for r in results) + 1
    metadata = _metadata(
        n_rows,
        failed_models=json.dumps(list(failed_models)),
        failed_outputs=json.dumps(failed_outputs or {}),
    )
    return PrivateEvaluation(leaderboard=rows, comparisons=results, metadata=metadata)


def _plan(evaluation, **kwargs):
    kwargs.setdefault("alias", "gov-scans-v1")
    return plan_share(evaluation, results_repo=RESULTS, now=NOW, **kwargs)


def _labels(n_rows, groups):
    """Slice values by row: groups maps label -> row ids."""
    values = [None] * n_rows
    for label, rows in groups.items():
        for row in rows:
            values[row] = label
    return values


def _rewrite_rows(plan, path, change):
    rows = [json.loads(line) for line in plan.files[path].decode().splitlines()]
    change(rows)
    plan.files[path] = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode()


# ---------------------------------------------------------------------------
# Boards
# ---------------------------------------------------------------------------


class TestBoards:
    def test_overall_board_reproduces_the_private_board(self):
        evaluation = _evaluation(_results(range(20)))
        plan = _plan(evaluation)
        private = {row["model"]: row["elo"] for row in evaluation.leaderboard}
        assert {row["model"]: row["elo"] for row in plan.overall_rows} == private
        assert [row["rank"] for row in plan.overall_rows] == [1, 2, 3]
        assert all(row["pages"] == 20 for row in plan.overall_rows)
        assert plan.settings["pages"] == 20
        assert plan.settings["comparisons"] == 60
        assert (plan.settings["seed"], plan.settings["adaptive_strategy"]) == (42, "balanced")

    def test_judge_spec_hashes_are_published(self):
        evaluation = _evaluation(_results(range(20)))
        evaluation.metadata["judge_spec_hashes"] = json.dumps(["ef" * 32])
        plan = _plan(evaluation)
        assert json.loads(plan.files["board.json"])["judge_spec_hashes"] == ["ef" * 32]

    def test_a_board_with_other_ratings_is_refused(self):
        evaluation = _evaluation(_results(range(20)))
        evaluation.leaderboard[0]["elo"] += 40
        with pytest.raises(ShareError, match="does not match its comparisons"):
            _plan(evaluation)

    def test_extra_verdicts_with_the_same_ratings_are_refused(self):
        # A checkpoint can add pages whose outcomes keep every rating the same.
        evaluation = _evaluation(_results(range(20)))
        for result in _results(range(20)):
            result.sample_idx += 20
            evaluation.comparisons.append(result)
        refit = _evaluation(evaluation.comparisons).leaderboard
        assert [r["elo"] for r in refit] == [r["elo"] for r in evaluation.leaderboard]
        with pytest.raises(ShareError, match="does not match its comparisons"):
            _plan(evaluation)

    def test_failed_model_history_stays_out_of_the_fit(self):
        """Judge fits without a failed model; its older comparisons remain as history."""
        models = [*MODELS, "org/model-broken"]
        history = _results(range(20), ["org/model-good", "org/model-broken"], seed=1)
        evaluation = _evaluation(
            _results(range(20)) + history,
            models,
            failed_models=["org/model-broken"],
            failed_outputs={"org/model-broken": 20},
        )
        plan = _plan(evaluation)
        failed = plan.overall_rows[-1]
        assert failed["model"] == "org/model-broken"
        assert (failed["rank"], failed["elo"], failed["status"]) == (None, None, "failed")
        assert failed["failed_outputs"] == 20
        assert history[0].text_b in plan.private_texts
        assert "FAILED" in plan.files["README.md"].decode()

    def test_out_of_grid_history_is_not_ranked_but_is_still_scanned(self):
        history = _results(range(20), ["org/model-good", "org/old-model"], seed=1)
        for result in history:
            result.text_b = f"OLD MODEL TRANSCRIPT of page {result.sample_idx}: " + "Qx7" * 30
        plan = _plan(_evaluation(_results(range(20)) + history))
        assert "org/old-model" not in {row["model"] for row in plan.overall_rows}
        assert history[0].text_b in plan.private_texts

    def test_no_slices_without_a_slice_column(self):
        plan = _plan(_evaluation(_results(range(20))))
        assert plan.slice_rows == []
        assert "slices.jsonl" not in plan.files
        assert "config_name: slices" not in plan.files["README.md"].decode()

    def test_a_custom_criteria_file_name_is_not_published(self):
        evaluation = _evaluation(_results(range(20)))
        evaluation.metadata["criteria"] = "custom:idl-drug-rubric.txt"
        plan = _plan(evaluation)
        assert plan.settings["criteria"] == "custom"
        assert "rubric" not in b"".join(plan.files.values()).decode()

    def test_models_never_compared_across_groups_are_refused(self):
        # Two independent pairs: nothing orders A-B against C-D.
        models = ["org/a", "org/b", "org/c", "org/d"]
        results = _results(range(20), models, pairs=[("org/a", "org/b"), ("org/c", "org/d")])
        with pytest.raises(ShareError, match="cannot rank every leaderboard model"):
            _plan(_evaluation(results, models))

    def test_a_model_whose_only_non_losses_are_ties_is_rankable(self):
        # A tie counts half each way, so one tie gives the weaker model a finite rating.
        results = [
            ComparisonResult(
                sample_idx=page,
                model_a="org/a",
                model_b="org/b",
                winner="tie" if page == 0 else "A",
                provenance_hash="h",
            )
            for page in range(20)
        ]
        plan = _plan(_evaluation(results, ["org/a", "org/b"]))
        assert [row["model"] for row in plan.overall_rows] == ["org/a", "org/b"]

    def test_a_model_that_won_every_comparison_is_refused(self):
        # The unregularised fit puts such a model at an unbounded rating with a
        # zero-width interval.
        results = [
            ComparisonResult(
                sample_idx=page, model_a="org/a", model_b="org/b", winner="A", provenance_hash="h"
            )
            for page in range(20)
        ]
        with pytest.raises(ShareError, match="won \\(or lost\\)"):
            _plan(_evaluation(results, ["org/a", "org/b"]))

    @pytest.mark.parametrize(
        "alias", ["", "   ", "x" * 65, "two\nlines", "carriage\rreturn", "line\u2028separator"]
    )
    def test_alias_must_be_a_short_single_line(self, alias):
        with pytest.raises(ShareError, match="--alias"):
            _plan(_evaluation(_results(range(20))), alias=alias)


class TestSlices:
    def _plan(self, groups, n_rows=40, pages=None, results=None, **kwargs):
        values = _labels(n_rows, groups)
        results = results or _results(pages or range(40))
        evaluation = _evaluation(results, n_rows=n_rows, **kwargs)
        return _plan(evaluation, slice_column="doc_type", slice_values=lambda m, c: values)

    def test_one_board_per_label_with_enough_pages(self):
        plan = self._plan({"letters": range(0, 15), "forms": range(15, 30), "maps": range(30, 33)})
        assert plan.settings["slices"] == ["forms", "letters"]
        assert plan.settings["slice_pages"] == [15, 15]
        # 3 pages of maps (too few) and 7 pages with no label.
        assert plan.settings["pages_not_in_a_slice_board"] == 10
        assert {row["slice"] for row in plan.slice_rows} == {"forms", "letters"}
        assert "maps" not in b"".join(plan.files.values()).decode()

    def test_slice_board_is_the_fit_on_that_label_only(self):
        plan = self._plan({"letters": range(0, 15), "forms": range(15, 40)})
        letters = [r for r in _results(range(40)) if r.sample_idx < 15]
        expected = compute_elo(letters, MODELS, n_bootstrap=0)
        got = {row["model"]: row["elo"] for row in plan.slice_rows if row["slice"] == "letters"}
        assert got == {model: round(elo) for model, elo in expected.elo.items()}

    def test_a_slice_whose_models_were_never_compared_across_is_left_out(self):
        # Targeted runs judge only some pairs. On these pages good-vs-mid and
        # weak-vs-other were judged, but the two pairs were never linked.
        models = [*MODELS, "org/model-other"]
        linked = _results(range(20), models)
        split = _results(
            range(20, 40),
            pairs=[("org/model-good", "org/model-mid"), ("org/model-weak", "org/model-other")],
        )
        plan = self._plan(
            {"letters": range(0, 20), "forms": range(20, 40)},
            results=linked + split,
            models=models,
        )
        assert plan.settings["slices"] == ["letters"]
        assert plan.settings["pages_not_in_a_slice_board"] == 20

    def test_a_model_with_few_pages_in_a_slice_is_left_out_of_its_board(self):
        few = _results(range(5), ["org/model-good", "org/model-new"], seed=2)
        plan = self._plan(
            {"letters": range(0, 20), "forms": range(20, 40)},
            results=_results(range(40)) + few,
            models=[*MODELS, "org/model-new"],
        )
        assert "org/model-new" in {row["model"] for row in plan.overall_rows}
        letters = {row["model"] for row in plan.slice_rows if row["slice"] == "letters"}
        assert letters == set(MODELS)

    def test_slice_rows_do_not_repeat_whole_sample_failure_counts(self):
        plan = self._plan(
            {"letters": range(0, 20), "forms": range(20, 40)},
            failed_outputs={"org/model-weak": 5},
        )
        weak = [row for row in plan.overall_rows if row["model"] == "org/model-weak"]
        assert weak[0]["status"] == "degraded"
        assert all("status" not in row and "failed_outputs" not in row for row in plan.slice_rows)

    def test_a_low_coverage_cascade_leaves_the_slice_out(self):
        # good-mid share 1 page; good-weak and mid-other 9 pages each. weak and
        # other have 9 pages and drop out; then good and mid have 1 page each.
        models = [*MODELS, "org/model-other"]
        cascade = _results(range(20, 21), pairs=[("org/model-good", "org/model-mid")])
        cascade += _results(range(21, 30), pairs=[("org/model-good", "org/model-weak")])
        cascade += _results(range(30, 39), pairs=[("org/model-mid", "org/model-other")])
        plan = self._plan(
            {"letters": range(0, 20), "forms": range(20, 39)},
            results=_results(range(20), models) + cascade,
            models=models,
        )
        assert plan.settings["slices"] == ["letters"]

    def test_a_slice_where_one_model_won_every_comparison_is_left_out(self):
        # On the forms pages good beats both others every time: no finite rating.
        sweep = [
            ComparisonResult(
                sample_idx=page,
                model_a="org/model-good",
                model_b=other,
                winner="A",
                provenance_hash="h",
            )
            for page in range(20, 40)
            for other in ("org/model-mid", "org/model-weak")
        ]
        sweep += _results(range(20, 40), pairs=[("org/model-mid", "org/model-weak")])
        plan = self._plan(
            {"letters": range(0, 20), "forms": range(20, 40)},
            results=_results(range(20)) + sweep,
        )
        assert plan.settings["slices"] == ["letters"]
        assert plan.settings["pages_not_in_a_slice_board"] == 20

    def test_slice_pages_count_only_pages_on_the_board(self):
        # forms: 10 pages compare good-mid-weak, 9 pages compare only new-other.
        # new and other have 9 pages (< 10) and leave the board, with their pages.
        models = [*MODELS, "org/model-new", "org/model-other"]
        linking = _results(range(0, 20), models)
        forms = _results(range(20, 30))
        forms += _results(range(30, 39), pairs=[("org/model-new", "org/model-other")])
        plan = self._plan(
            {"letters": range(0, 20), "forms": range(20, 39)},
            results=linking + forms,
            models=models,
        )
        pages = dict(zip(plan.settings["slices"], plan.settings["slice_pages"], strict=True))
        assert pages["forms"] == 10
        assert plan.settings["pages_not_in_a_slice_board"] == 9

    @pytest.mark.parametrize("label", ["line\u2028separator", "carriage\rreturn", "form\x0cfeed"])
    def test_labels_with_any_line_break_are_refused(self, label):
        with pytest.raises(ShareError, match="on one line"):
            self._plan({label: range(0, 20), "forms": range(20, 40)})

    def test_markdown_in_a_label_is_shown_as_text(self):
        label = "[x](https://example.com)<img src=x>"
        plan = self._plan({label: range(0, 20), "forms": range(20, 40)})
        readme = plan.files["README.md"].decode()
        assert "\\[x\\](https://example.com)\\<img src=x\\>" in readme
        assert not re.search(r"(?<!\\)\[x|(?<!\\)<img", readme)  # no unescaped markup

    def test_comparisons_outside_the_drawn_sample_are_refused(self):
        drawn = sorted(sample_indices(100, 40, 42))
        others = sorted(set(range(100)) - set(drawn))[:40]
        values = ["letters"] * 100
        for pages, ok in ((drawn, True), (others, False)):
            evaluation = _evaluation(_results(pages), n_rows=100)
            evaluation.metadata["max_samples"] = 40

            def plan(evaluation=evaluation):
                return _plan(evaluation, slice_column="doc_type", slice_values=lambda m, c: values)

            if ok:
                plan()
            else:
                with pytest.raises(ShareError, match="outside the judged sample"):
                    plan()

    def test_a_per_page_column_is_refused(self):
        with pytest.raises(ShareError, match="per-page column"):
            self._plan({f"title {page}": [page] for page in range(60)}, n_rows=60, pages=range(60))

    def test_long_labels_are_refused(self):
        with pytest.raises(ShareError, match="64 characters or fewer"):
            self._plan({"x" * 65: range(40)})

    def test_a_source_with_fewer_rows_is_refused(self):
        with pytest.raises(ShareError, match="source changed"):
            self._plan({"letters": range(30)}, n_rows=30, pages=range(40))

    def test_a_label_that_names_the_source_repo_is_refused(self):
        with pytest.raises(PrivacyError, match="names the results repo or the source"):
            self._plan({SOURCE: range(0, 20), "forms": range(20, 40)})


class TestLocalFiles:
    def test_a_stale_slice_file_is_removed_from_the_dry_run_folder(self, tmp_path):
        values = _labels(40, {"letters": range(0, 20), "forms": range(20, 40)})
        evaluation = _evaluation(_results(range(40)), n_rows=40)
        with_slices = _plan(evaluation, slice_column="doc_type", slice_values=lambda m, c: values)
        write_local(with_slices, tmp_path)
        assert (tmp_path / "slices.jsonl").exists()
        write_local(_plan(evaluation), tmp_path)
        assert sorted(p.name for p in tmp_path.iterdir()) == sorted(BOARD_FILES)


class FakeHistory:
    """The commit history of a results repo: folder -> last commit, newest first."""

    def __init__(self, last, commits=("c3", "c2", "c1"), sha="c3"):
        self.last, self.commits, self.sha = last, list(commits), sha

    def dataset_info(self, repo_id):
        return SimpleNamespace(sha=self.sha)

    def get_paths_info(self, repo_id, paths, expand=False, revision=None, repo_type=None):
        return [
            SimpleNamespace(path=path, last_commit=SimpleNamespace(oid=self.last[path]))
            for path in paths
            if path in self.last
        ]

    def list_repo_commits(self, repo_id, repo_type=None, revision=None):
        return [SimpleNamespace(commit_id=commit) for commit in self.commits]


class TestPublishOrder:
    """A judge run pushes comparisons, then the leaderboard, then the metadata."""

    def test_a_finished_run_passes(self):
        check_publish_order(
            FakeHistory({"comparisons": "c1", "data": "c2", "metadata": "c3"}), RESULTS, "c3"
        )

    def test_one_commit_for_all_three_passes(self):
        history = FakeHistory({"comparisons": "c3", "data": "c3", "metadata": "c3"})
        check_publish_order(history, RESULTS, "c3")

    def test_an_unchanged_leaderboard_is_not_a_failed_run(self):
        # An identical leaderboard is not uploaded again, so its folder looks older.
        history = FakeHistory({"comparisons": "c2", "data": "c1", "metadata": "c3"})
        check_publish_order(history, RESULTS, "c3")

    @pytest.mark.parametrize(
        "last",
        [
            {"comparisons": "c3", "data": "c2", "metadata": "c2"},  # a checkpoint after the run
            {"comparisons": "c2", "data": "c3", "metadata": "c1"},  # stopped before the metadata
        ],
    )
    def test_an_unfinished_run_is_refused(self, last):
        with pytest.raises(ShareError, match="did not finish publishing"):
            check_publish_order(FakeHistory(last), RESULTS, "c3")

    def test_a_repo_without_the_judge_folders_is_refused(self):
        with pytest.raises(ShareError, match="last commit of the metadata folder"):
            check_publish_order(FakeHistory({"comparisons": "c1", "data": "c2"}), RESULTS, "c3")

    def test_all_three_are_read_at_the_checked_revision(self, monkeypatch):
        evaluation = _evaluation(_results(range(20)))
        reads = []

        def fake_load_dataset(repo, split=None, revision=None):
            reads.append(("leaderboard", revision))
            return evaluation.leaderboard

        def fake_comparisons(repo, revision=None):
            reads.append(("comparisons", revision))
            return evaluation.comparisons

        def fake_metadata(repo, revision=None):
            reads.append(("metadata", revision))
            return [evaluation.metadata]

        monkeypatch.setattr(share, "load_dataset", fake_load_dataset)
        monkeypatch.setattr(share, "load_existing_comparisons", fake_comparisons)
        monkeypatch.setattr(share, "load_existing_metadata", fake_metadata)
        history = FakeHistory({"comparisons": "c1", "data": "c2", "metadata": "c3"}, sha="c3")
        loaded = load_private_evaluation(RESULTS, api=history)
        assert reads == [("leaderboard", "c3"), ("comparisons", "c3"), ("metadata", "c3")]
        assert loaded.metadata == evaluation.metadata


def _png(shade):
    buffer = io.BytesIO()
    PILImage.new("L", (2, 2), shade).save(buffer, format="PNG")
    return buffer.getvalue()


def _source(n_rows=30, order=None, labels=None, images=True):
    """A flat source with page ids, page images and slice labels; order permutes the rows."""
    order = order or list(range(n_rows))
    columns = {
        "id": [f"page-{i:03d}" for i in order],
        "doc_type": [(labels or {}).get(i, "letters" if i < 15 else "forms") for i in order],
        "markdown": [f"text {i}" for i in order],
    }
    if not images:
        return Dataset.from_dict(columns)
    columns["image"] = [{"bytes": _png(i), "path": None} for i in order]
    return Dataset.from_dict(columns).cast_column("image", Image())


class TestSliceSource:
    """The slice column is read only after proving the rows are the judged pages."""

    def _metadata(self, judged_source, max_samples=20, **overrides):
        judged = sample_indices(len(judged_source), max_samples, 42)
        fingerprint, _ = evaluation_input_fingerprints(judged_source, {}, judged)
        return _metadata(max_samples, source_fingerprint=fingerprint, **overrides)

    def _loader(self, monkeypatch, dataset, calls=None):
        def fake_load_dataset(repo, name=None, split=None, revision=None):
            if calls is not None:
                calls.append((repo, name, split, revision))
            return dataset

        monkeypatch.setattr(share, "load_dataset", fake_load_dataset)

    def test_unchanged_source_gives_the_labels(self, monkeypatch):
        source = _source()
        self._loader(monkeypatch, source)
        assert load_slice_values(self._metadata(source), "doc_type") == source["doc_type"]

    @pytest.mark.parametrize(
        "changed",
        [
            _source(order=[*range(15, 30), *range(15)]),  # same rows, other order
            _source(order=[*range(29), 99]),  # one page replaced
            _source(n_rows=31),  # one page added
        ],
    )
    def test_a_changed_source_is_refused(self, monkeypatch, changed):
        metadata = self._metadata(_source())
        self._loader(monkeypatch, changed)
        with pytest.raises(ShareError, match="not the pages judge compared"):
            load_slice_values(metadata, "doc_type")

    def test_relabelled_pages_give_the_new_labels(self, monkeypatch):
        metadata = self._metadata(_source())
        self._loader(monkeypatch, _source(labels={0: "maps"}))
        assert load_slice_values(metadata, "doc_type")[0] == "maps"

    def test_the_reference_config_and_its_revision_are_read(self, monkeypatch):
        source = _source()
        calls = []
        self._loader(monkeypatch, source, calls)
        metadata = self._metadata(
            source,
            source_configs=json.dumps(["no-text", "model-b", "model-a"]),
            source_output_columns=json.dumps({"model-a": "markdown", "model-b": "markdown"}),
            source_revisions=json.dumps({"model-b": "refs/pr/3", "model-a": "main"}),
        )
        load_slice_values(metadata, "doc_type")
        assert calls == [(SOURCE, "model-b", "train", "refs/pr/3")]

    def test_a_source_without_images_is_refused(self, monkeypatch):
        # Ids alone do not show that a row is still the judged page: the labels and
        # text could be replaced under the same ids.
        source = _source(images=False)
        self._loader(monkeypatch, source)
        with pytest.raises(ShareError, match="no image column"):
            load_slice_values(self._metadata(source), "doc_type")

    def test_an_image_column_the_fingerprint_does_not_read_is_refused(self, monkeypatch):
        # The fingerprint reads the column named like an image; "scan" is not.
        source = _source().rename_column("image", "scan")
        self._loader(monkeypatch, source)
        with pytest.raises(ShareError, match="no image column"):
            load_slice_values(self._metadata(source), "doc_type")

    def test_a_missing_column_is_an_error(self, monkeypatch):
        source = _source()
        self._loader(monkeypatch, source)
        with pytest.raises(ShareError, match="'decade' is not in the judged source"):
            load_slice_values(self._metadata(source), "decade")

    def test_no_fingerprint_is_an_error(self, monkeypatch):
        self._loader(monkeypatch, _source())
        with pytest.raises(ShareError, match="no source fingerprint"):
            load_slice_values(_metadata(20, source_fingerprint=""), "doc_type")

    def test_a_read_error_does_not_repeat_the_loader_message(self, monkeypatch):
        def failing(*args, **kwargs):
            raise OSError("page-0042: CONFIDENTIAL memo text")

        monkeypatch.setattr(share, "load_dataset", failing)
        with pytest.raises(ShareError) as exc:
            load_slice_values(self._metadata(_source()), "doc_type")
        assert "CONFIDENTIAL" not in str(exc.value)
        assert "page-0042" not in str(exc.value)


def _config(n_rows, *, ids=True):
    """One OCR output config: page image, label, OCR text (and page ids)."""
    columns = {
        "image": [{"bytes": _png(i), "path": None} for i in range(n_rows)],
        "doc_type": ["letters" if i < n_rows // 2 else "forms" for i in range(n_rows)],
        "markdown": [f"text of page {i}" for i in range(n_rows)],
    }
    if ids:
        columns = {"id": [f"page-{i:03d}" for i in range(n_rows)], **columns}
    return Dataset.from_dict(columns).cast_column("image", Image())


class TestFingerprintMatchesJudge:
    """share must reproduce the fingerprint judge stored, from judge's own merge."""

    def _check(self, monkeypatch, configs, *, ids=True):
        tables = {name: _config(30, ids=ids) for name in configs}

        def fake_load_dataset(path=None, name=None, split=None, revision=None, **kwargs):
            return tables[name]

        monkeypatch.setattr(dataset, "load_dataset", fake_load_dataset)
        monkeypatch.setattr(share, "load_dataset", fake_load_dataset)
        outputs: dict[str, str] = {}
        merged, _ = load_config_dataset(SOURCE, configs, source_output_columns=outputs)
        judged = sample_indices(len(merged), 20, 42)
        fingerprint, _ = evaluation_input_fingerprints(merged, {}, judged)
        metadata = _metadata(
            20,
            source_fingerprint=fingerprint,
            source_configs=json.dumps(configs),
            source_output_columns=json.dumps(outputs, sort_keys=True),
            source_revisions=json.dumps(dict.fromkeys(configs, "main")),
        )
        return load_slice_values(metadata, "doc_type")

    def test_ids_and_images_from_the_reference_config(self, monkeypatch):
        labels = self._check(monkeypatch, ["model-a", "model-b"])
        assert labels[:2] == ["letters", "letters"]

    def test_a_config_named_like_an_id_column(self, monkeypatch):
        # Judge's merge adds a column named "id" (that config's OCR text), which
        # judge fingerprints as a page id. share rebuilds the merge for this case.
        labels = self._check(monkeypatch, ["id", "other"], ids=False)
        assert len(labels) == 30

    def test_an_unreadable_image_does_not_show_its_path(self, monkeypatch):
        broken = _config(30).cast_column("image", Image(decode=False))
        broken = broken.map(
            lambda row: {"image": {"bytes": None, "path": "/tmp/CONFIDENTIAL-page.png"}}
        )
        monkeypatch.setattr(share, "load_dataset", lambda *a, **k: broken)
        with pytest.raises(ShareError) as exc:
            load_slice_values(_metadata(20), "doc_type")
        assert "CONFIDENTIAL" not in str(exc.value)


# ---------------------------------------------------------------------------
# Privacy
# ---------------------------------------------------------------------------

SECRET = st.text(alphabet=string.ascii_letters + string.digits, min_size=16, max_size=40)


def _all_text(files):
    """Raw file text plus every decoded JSON string, as the checks see it."""
    return "\n".join("\n".join(share._file_views(path, data)) for path, data in files.items())


class TestPrivacy:
    @given(secret=SECRET, data=st.data())
    @settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_no_page_text_or_reason_reaches_the_files(self, secret, data):
        """Whatever the outputs and reasons hold, no file holds them."""
        filler = data.draw(st.text(min_size=0, max_size=40))
        results = _results(
            range(24),
            text=lambda page, model: f"{filler}{secret}\n{page}{model}{secret[::-1]}",
            reason=lambda page: f'Output A keeps "{secret}"; B drops it\n{filler}',
        )
        values = ["letters" if page % 2 else "forms" for page in range(24)]
        plan = _plan(
            _evaluation(results, n_rows=24),
            slice_column="doc_type",
            slice_values=lambda metadata, column: values,
        )  # plan_share runs every check, so this also shows there is no false alarm
        everything = _all_text(plan.files)
        assert secret not in everything
        assert secret[::-1] not in everything
        assert SOURCE not in everything
        assert RESULTS not in everything

    @given(seed=st.integers(0, 2**32 - 1), length=st.integers(63, 200), data=st.data())
    @settings(max_examples=60, deadline=None)
    def test_a_copied_fragment_of_63_characters_is_found(self, seed, length, data):
        """Holds for page text: every aligned chunk has 4 or more distinct letters."""
        rng = random.Random(seed)
        text = "".join(rng.choice(string.ascii_letters + " \n") for _ in range(400))
        start = data.draw(st.integers(0, len(text) - length))
        fragment = text[start : start + length]
        where = data.draw(st.sampled_from(["README.md", "overall.jsonl", "board.json"]))
        if where == "README.md":
            files = {where: f"# Board\n\nnote: {fragment}\n".encode()}
        elif where == "overall.jsonl":
            files = {where: (json.dumps({"model": fragment}) + "\n").encode()}
        else:
            files = {where: json.dumps({"alias": fragment}, indent=2).encode()}
        with pytest.raises(PrivacyError, match="holds text from a compared page"):
            check_no_page_text(files, [text])

    def test_markup_and_layout_runs_are_not_alarms(self):
        # A markdown table in an OCR output shares its rule with the board's tables.
        # Repeat it at every offset, so one copy lines up with a 32-character chunk.
        table = "|---" * 9 + "|\n| 1 | "
        page = "".join("x" * shift + "\n" + table for shift in range(CHUNK))
        page += "Contents" + "." * 64 + "3\n" + "-" * 64 + " " * 64
        page += "<tr><td></td><td></td><td></td><td></td><td></td></tr>" * 3
        plan = _plan(_evaluation(_results(range(20), text=lambda page_id, model: page)))
        files = dict(plan.files)
        files["README.md"] += ("." * 80 + "\n" + "-" * 80 + " " * 80).encode()
        check_no_page_text(files, [page])

    def test_an_extra_column_is_refused(self):
        plan = _plan(_evaluation(_results(range(20))))
        _rewrite_rows(plan, "overall.jsonl", lambda rows: rows[0].update(reason="Output A"))
        with pytest.raises(PrivacyError, match="not the allowed columns"):
            check_plan(plan)

    def test_a_value_that_is_not_board_content_is_refused(self):
        plan = _plan(_evaluation(_results(range(20))))
        _rewrite_rows(plan, "overall.jsonl", lambda rows: rows[1].update(model="a page line"))
        with pytest.raises(PrivacyError, match="'model'"):
            check_plan(plan)

    def test_a_readme_that_is_not_the_rendered_card_is_refused(self):
        # Short ids and short text escape the chunk scan, so the card is rendered again.
        plan = _plan(_evaluation(_results(range(20))))
        plan.files["README.md"] += b"Private page id: manuscript-0042\n"
        with pytest.raises(PrivacyError, match="README.md is not the card"):
            check_plan(plan)

    def test_extra_settings_text_is_refused(self):
        plan = _plan(_evaluation(_results(range(20))))
        plan.files["board.json"] = plan.files["board.json"].replace(
            b'"alias"', b'"note": "box 12",\n  "alias"'
        )
        with pytest.raises(PrivacyError, match="unexpected keys"):
            check_plan(plan)

    def test_an_unknown_file_is_refused(self):
        plan = _plan(_evaluation(_results(range(20))))
        plan.files["comparisons.jsonl"] = b"{}\n"
        with pytest.raises(PrivacyError, match="Unexpected files"):
            check_plan(plan)

    def test_an_alias_that_names_the_results_repo_is_refused(self):
        with pytest.raises(PrivacyError, match="names the results repo or the source"):
            _plan(_evaluation(_results(range(20))), alias=f"board of {RESULTS}")

    def test_the_files_never_name_the_source_or_results_repo(self):
        plan = _plan(_evaluation(_results(range(20))))
        everything = _all_text(plan.files)
        assert "private-source" not in everything
        assert "private-results" not in everything
        assert plan.settings["source_fingerprint"] == "cd" * 32


# ---------------------------------------------------------------------------
# Push
# ---------------------------------------------------------------------------

SHARE_CARD = "---\ntags:\n  - ocr-bench\n  - ocr-bench-share\n---\n# OCR board\n"


def _missing(repo_id):
    response = SimpleNamespace(status_code=404, headers={}, request=None)
    return RepositoryNotFoundError(f"{repo_id} not found", response=response)


class RecordingApi:
    """Records every Hub call. ``repos`` maps id -> {"private", "files", "readme", ...}."""

    def __init__(self, repos=None):
        self.repos = repos or {}
        self.calls: list[tuple] = []

    def repo_info(self, repo_id, repo_type=None):
        self.calls.append(("repo_info", repo_id))
        if repo_id not in self.repos:
            raise _missing(repo_id)
        repo = self.repos[repo_id]
        return SimpleNamespace(private=repo["private"], sha=repo.get("sha", "sha-1"))

    def list_repo_files(self, repo_id, repo_type=None, revision=None):
        self.calls.append(("list_repo_files", repo_id, revision))
        return list(self.repos[repo_id]["files"])

    def hf_hub_download(self, repo_id, filename, repo_type=None, revision=None):
        self.calls.append(("hf_hub_download", repo_id, filename, revision))
        repo = self.repos[repo_id]
        path = repo["tmp"] / filename
        if filename == "README.md":
            path.write_text(repo["readme"])
        else:
            path.write_text(json.dumps({"alias": repo.get("alias", "gov-scans-v1")}))
        return str(path)

    def create_repo(self, repo_id, repo_type=None, private=None, exist_ok=False):
        self.calls.append(("create_repo", repo_id, private))
        self.repos[repo_id] = {"private": bool(private), "files": [".gitattributes"]}

    def create_commit(
        self, repo_id, repo_type=None, operations=(), commit_message="", parent_commit=None
    ):
        self.calls.append(("create_commit", repo_id, list(operations), parent_commit))


def _existing(tmp_path, files, readme=SHARE_CARD, private=False, alias="gov-scans-v1"):
    return {
        "me/board": {
            "private": private,
            "sha": "sha-checked",
            "files": [".gitattributes", *files],
            "readme": readme,
            "alias": alias,
            "tmp": tmp_path,
        }
    }


BOARD_FILES = ["README.md", "overall.jsonl", "board.json"]


class TestPush:
    def _plan(self, **kwargs):
        return _plan(_evaluation(_results(range(20))), **kwargs)

    def _push(self, api, dest="me/board", **kwargs):
        return push_share(self._plan(), dest, results_repo=RESULTS, api=api, **kwargs)

    def test_a_failed_check_makes_no_hub_call(self):
        plan = self._plan()
        plan.files["README.md"] += plan.private_texts[0].encode() * 3
        api = RecordingApi()
        with pytest.raises(PrivacyError):
            push_share(plan, "me/board", results_repo=RESULTS, api=api)
        assert api.calls == []

    def test_a_new_repo_is_created_then_written_in_one_commit(self):
        api = RecordingApi()
        result = self._push(api)
        assert [call[0] for call in api.calls] == [
            "repo_info",
            "create_repo",
            "repo_info",
            "list_repo_files",
            "create_commit",
            "repo_info",
        ]
        assert api.calls[1] == ("create_repo", "me/board", True)  # private by default
        operations = api.calls[4][2]
        assert sorted(op.path_in_repo for op in operations) == sorted(BOARD_FILES)
        assert all(isinstance(op, CommitOperationAdd) for op in operations)
        assert result.created and result.private

    def test_a_commit_by_another_writer_after_creation_is_refused(self, tmp_path):
        class RacingApi(RecordingApi):
            def create_repo(self, repo_id, repo_type=None, private=None, exist_ok=False):
                super().create_repo(repo_id, repo_type, private, exist_ok)
                self.repos[repo_id].update(
                    sha="sha-other",
                    files=[".gitattributes", "private-pages.parquet"],
                    tmp=tmp_path,
                    readme="",
                )

        api = RacingApi()
        with pytest.raises(ShareError, match="did not write"):
            self._push(api)
        assert "create_commit" not in [call[0] for call in api.calls]

    def test_public_flag_creates_a_public_repo(self):
        api = RecordingApi()
        result = self._push(api, public=True)
        assert api.calls[1] == ("create_repo", "me/board", False)
        assert not result.private

    def test_an_existing_public_board_stays_public(self, tmp_path):
        api = RecordingApi(_existing(tmp_path, BOARD_FILES, private=False))
        result = self._push(api)
        assert "create_repo" not in [call[0] for call in api.calls]
        assert not result.private and not result.created

    def test_an_existing_private_board_stays_private(self, tmp_path):
        api = RecordingApi(_existing(tmp_path, BOARD_FILES, private=True))
        result = self._push(api)
        assert "create_repo" not in [call[0] for call in api.calls]
        assert result.private and not result.created

    def test_the_commit_is_on_the_checked_revision(self, tmp_path):
        api = RecordingApi(_existing(tmp_path, BOARD_FILES))
        self._push(api)
        reads = [call for call in api.calls if call[0] in ("list_repo_files", "hf_hub_download")]
        assert {call[-1] for call in reads} == {"sha-checked"}
        commit = next(call for call in api.calls if call[0] == "create_commit")
        assert commit[3] == "sha-checked"

    def test_stale_slice_file_is_deleted(self, tmp_path):
        api = RecordingApi(_existing(tmp_path, [*BOARD_FILES, "slices.jsonl"]))
        self._push(api)
        commit = next(call for call in api.calls if call[0] == "create_commit")
        deletes = [op.path_in_repo for op in commit[2] if isinstance(op, CommitOperationDelete)]
        assert deletes == ["slices.jsonl"]

    def test_a_board_of_another_alias_needs_replace(self, tmp_path):
        api = RecordingApi(_existing(tmp_path, BOARD_FILES, alias="idl-v1"))
        with pytest.raises(ShareError, match="holds the board of 'idl-v1'"):
            self._push(api)
        assert "create_commit" not in [call[0] for call in api.calls]
        self._push(api, replace=True)
        assert "create_commit" in [call[0] for call in api.calls]

    @pytest.mark.parametrize(
        ("files", "readme"),
        [
            # A results repo: tagged ocr-bench, holds comparisons.
            (
                ["README.md", "comparisons/train-0.parquet"],
                "---\ntags:\n  - ocr-bench\n  - leaderboard\n---\n",
            ),
            # The share tag in prose only.
            (["README.md"], "# Notes\n\n- ocr-bench-share\n"),
            # A share card next to a file share does not write.
            ([*BOARD_FILES, "private-pages.parquet"], SHARE_CARD),
        ],
    )
    def test_a_repo_share_did_not_write_is_refused(self, tmp_path, files, readme):
        api = RecordingApi(_existing(tmp_path, files, readme=readme))
        with pytest.raises(ShareError, match="did not write"):
            self._push(api)
        assert "create_commit" not in [call[0] for call in api.calls]

    @pytest.mark.parametrize("dest", [RESULTS, SOURCE, RESULTS.upper()])
    def test_the_results_or_source_repo_is_refused(self, dest):
        api = RecordingApi()
        with pytest.raises(ShareError, match="separate repo"):
            self._push(api, dest=dest)
        assert api.calls == []


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestCli:
    def test_alias_is_required(self):
        with pytest.raises(SystemExit):
            cli.build_parser().parse_args(["share", "me/results", "me/board"])

    def _dry_run(self, monkeypatch, alias, *extra):
        evaluation = _evaluation(_results(range(20)))
        monkeypatch.setattr(share, "load_private_evaluation", lambda repo: evaluation)

        def no_push(*args, **kwargs):
            raise AssertionError("dry run must not push")

        monkeypatch.setattr(share, "push_share", no_push)
        argv = ["ocr-bench", "share", RESULTS, "me/board", "--alias", alias, "--dry-run"]
        monkeypatch.setattr(sys, "argv", [*argv, *extra])
        cli.main()

    def test_dry_run_writes_files_and_pushes_nothing(self, monkeypatch, tmp_path):
        out = tmp_path / "board"
        self._dry_run(monkeypatch, "v1", "--output-dir", str(out))
        assert sorted(p.name for p in out.iterdir()) == sorted(BOARD_FILES)
        assert "# OCR board: v1" in (out / "README.md").read_text()

    def test_dry_run_with_slices_prints_and_writes_the_slice_boards(
        self, monkeypatch, tmp_path, capsys
    ):
        values = ["letters" if page < 10 else "forms" for page in range(20)]
        monkeypatch.setattr(share, "load_slice_values", lambda metadata, column: values)
        out = tmp_path / "board"
        self._dry_run(monkeypatch, "v1", "--slice-column", "doc_type", "--output-dir", str(out))
        assert sorted(p.name for p in out.iterdir()) == sorted([*BOARD_FILES, "slices.jsonl"])
        printed = capsys.readouterr().out
        assert "v1: letters" in printed and "2 slice boards" in printed

    def test_the_default_folder_stays_in_the_working_directory(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        self._dry_run(monkeypatch, "../../escape")
        [folder] = tmp_path.iterdir()
        assert folder.name.startswith("ocr-bench-share-") and "/" not in folder.name
        assert (folder / "README.md").exists()

    def test_a_privacy_error_exits_with_an_error(self, monkeypatch, capsys):
        def leaky(*args, **kwargs):
            raise PrivacyError("README.md holds text from a compared page")

        monkeypatch.setattr(share, "load_private_evaluation", lambda repo: None)
        monkeypatch.setattr(share, "plan_share", leaky)
        monkeypatch.setattr(sys, "argv", ["ocr-bench", "share", "me/r", "me/b", "--alias", "v1"])
        with pytest.raises(SystemExit) as exc:
            cli.main()
        assert exc.value.code == 1
        assert "holds text from a compared page" in capsys.readouterr().out
