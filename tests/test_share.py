"""`share`: a scores-only board from a results repo, with fail-closed privacy checks."""

from __future__ import annotations

import json
import random
import string
import sys
from itertools import combinations
from types import SimpleNamespace

import pytest
from huggingface_hub import CommitOperationAdd, CommitOperationDelete
from huggingface_hub.errors import RepositoryNotFoundError
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from ocr_bench import cli, share
from ocr_bench.elo import ComparisonResult, compute_elo
from ocr_bench.publish import build_leaderboard_rows
from ocr_bench.share import (
    CHUNK,
    PrivacyError,
    PrivateEvaluation,
    ShareError,
    check_no_page_text,
    check_plan,
    load_slice_values,
    plan_share,
    push_share,
)

MODELS = ["org/model-good", "org/model-mid", "org/model-weak"]
STRENGTH = {"org/model-good": 0.8, "org/model-mid": 0.5, "org/model-weak": 0.2}
NOW = "2026-11-06T23:00:00+00:00"


@pytest.fixture(autouse=True)
def _few_bootstrap_resamples(monkeypatch):
    # Each resample is a scipy fit. Ratings do not depend on the resamples.
    monkeypatch.setattr(share, "N_BOOTSTRAP", 20)


def _results(pages, models=MODELS, *, text=None, reason=None, seed=0):
    """One comparison per model pair per page. Stronger models win more often."""
    rng = random.Random(seed)
    results = []
    for page in pages:
        for a, b in combinations(models, 2):
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
        "source_dataset": "me/private-source",
        "source_split": "train",
        "source_configs": json.dumps([]),
        "source_revisions": "{}",
        "judge_models": json.dumps(["novita:Qwen/Qwen3.8-27B"]),
        "seed": 42,
        "max_samples": n_rows,
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
    """A results repo as judge publishes it: the board fitted on the grid."""
    grid = set(models)
    board = compute_elo(
        [r for r in results if r.model_a in grid and r.model_b in grid], models, n_bootstrap=0
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
    return plan_share(evaluation, results_repo="me/private-results", now=NOW, **kwargs)


def _labels(n_rows, groups):
    """Slice values by row: groups maps label -> row ids."""
    values = [None] * n_rows
    for label, rows in groups.items():
        for row in rows:
            values[row] = label
    return values


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

    def test_a_board_that_differs_from_its_comparisons_is_refused(self):
        evaluation = _evaluation(_results(range(20)))
        evaluation.leaderboard[0]["elo"] += 40  # e.g. a later run stopped after a checkpoint
        with pytest.raises(ShareError, match="does not match its comparisons"):
            _plan(evaluation)

    def test_out_of_grid_history_is_not_ranked_but_is_still_scanned(self):
        grid = _results(range(20))
        history = _results(range(20), ["org/model-good", "org/old-model"], seed=1)
        for result in history:
            result.text_b = f"OLD MODEL TRANSCRIPT of page {result.sample_idx}: " + "Qx7" * 30
        evaluation = _evaluation(grid + history)
        plan = _plan(evaluation)
        assert "org/old-model" not in {row["model"] for row in plan.overall_rows}
        assert history[0].text_b in plan.private_texts

    def test_failed_model_is_listed_without_a_rank(self):
        models = [*MODELS, "org/model-broken"]
        evaluation = _evaluation(
            _results(range(20)),
            models,
            failed_models=["org/model-broken"],
            failed_outputs={"org/model-broken": 20},
        )
        plan = _plan(evaluation)
        failed = [row for row in plan.overall_rows if row["model"] == "org/model-broken"]
        assert failed == [
            {
                "rank": None,
                "model": "org/model-broken",
                "params": None,
                "elo": None,
                "elo_low": None,
                "elo_high": None,
                "pages": 0,
                "wins": 0,
                "losses": 0,
                "ties": 0,
                "win_pct": None,
                "status": "failed",
                "failed_outputs": 20,
            }
        ]
        assert "FAILED" in plan.files["README.md"].decode()

    def test_no_slices_without_a_slice_column(self):
        plan = _plan(_evaluation(_results(range(20))))
        assert plan.slice_rows == []
        assert "slices.jsonl" not in plan.files
        assert "config_name: slices" not in plan.files["README.md"].decode()

    @pytest.mark.parametrize("alias", ["", "   ", "x" * 65, "two\nlines"])
    def test_alias_must_be_a_short_single_line(self, alias):
        with pytest.raises(ShareError, match="--alias"):
            _plan(_evaluation(_results(range(20))), alias=alias)


class TestSlices:
    def _plan(self, groups, n_rows=40, pages=None):
        values = _labels(n_rows, groups)
        evaluation = _evaluation(_results(pages or range(40)), n_rows=n_rows)
        return _plan(
            evaluation, slice_column="doc_type", slice_values=lambda metadata, column: values
        )

    def test_one_board_per_slice_with_enough_pages(self):
        plan = self._plan({"letters": range(0, 15), "forms": range(15, 30), "maps": range(30, 33)})
        assert plan.settings["slices"] == ["forms", "letters"]
        # 3 pages of maps (too few) and 7 pages with no label.
        assert plan.settings["pages_not_in_a_slice_board"] == 10
        assert {row["slice"] for row in plan.slice_rows} == {"forms", "letters"}
        letters = [row for row in plan.slice_rows if row["slice"] == "letters"]
        assert {row["pages"] for row in letters} == {15}
        assert "maps" not in plan.files["README.md"].decode()
        assert "maps" not in plan.files["slices.jsonl"].decode()

    def test_slice_board_is_the_fit_on_that_slice_only(self):
        plan = self._plan({"letters": range(0, 15), "forms": range(15, 40)})
        evaluation = _evaluation(_results(range(40)), n_rows=40)
        letters = [r for r in evaluation.comparisons if r.sample_idx < 15]
        expected = compute_elo(letters, MODELS, n_bootstrap=0)
        got = {row["model"]: row["elo"] for row in plan.slice_rows if row["slice"] == "letters"}
        assert got == {model: round(elo) for model, elo in expected.elo.items()}

    def test_a_per_page_column_is_refused(self):
        with pytest.raises(ShareError, match="per-page column"):
            self._plan({f"title {page}": [page] for page in range(60)}, n_rows=60, pages=range(60))

    def test_long_labels_are_refused(self):
        with pytest.raises(ShareError, match="longer than"):
            self._plan({"x" * 65: range(40)})

    def test_a_source_with_fewer_rows_is_refused(self):
        with pytest.raises(ShareError, match="source changed"):
            self._plan({"letters": range(30)}, n_rows=30, pages=range(40))

    def test_a_source_whose_sample_moved_is_refused(self):
        # Judged 40 of 100 rows; the source now has 120 rows, so seed 42 picks other rows.
        pages = sorted(share.sample_indices(100, 40, 42))
        values = ["letters"] * 120
        evaluation = _evaluation(_results(pages), n_rows=100)
        evaluation.metadata["max_samples"] = 40
        with pytest.raises(ShareError, match="not the sample"):
            _plan(evaluation, slice_column="doc_type", slice_values=lambda m, c: values)

    def test_the_slice_column_is_read_from_the_first_config_that_has_it(self, monkeypatch):
        calls = []

        def fake_load_dataset(repo, name=None, split=None, revision=None, columns=None):
            calls.append((repo, name, split, revision, columns))
            if name == "model-a":
                raise ValueError("no such column")
            return _Col({"doc_type": ["letters", "forms"]})

        monkeypatch.setattr(share, "load_dataset", fake_load_dataset)
        metadata = _metadata(
            2,
            source_configs=json.dumps(["model-a", "model-b"]),
            source_revisions=json.dumps({"model-a": "main", "model-b": "refs/pr/3"}),
        )
        assert load_slice_values(metadata, "doc_type") == ["letters", "forms"]
        assert calls == [
            ("me/private-source", "model-a", "train", "main", ["doc_type"]),
            ("me/private-source", "model-b", "train", "refs/pr/3", ["doc_type"]),
        ]

    def test_a_missing_slice_column_is_an_error(self, monkeypatch):
        def fake_load_dataset(*args, **kwargs):
            raise ValueError("no such column")

        monkeypatch.setattr(share, "load_dataset", fake_load_dataset)
        with pytest.raises(ShareError, match="'doc_type' was not found"):
            load_slice_values(_metadata(2), "doc_type")


class _Col:
    def __init__(self, columns):
        self.columns = columns
        self.column_names = list(columns)

    def __getitem__(self, key):
        return self.columns[key]


# ---------------------------------------------------------------------------
# Privacy
# ---------------------------------------------------------------------------

SECRET = st.text(alphabet=string.ascii_letters + string.digits, min_size=16, max_size=40)


def _all_text(files):
    """Raw file text plus every decoded JSON string, as check_no_page_text sees it."""
    return "\n".join("\n".join(share._file_views(path, data)) for path, data in files.items())


class TestPrivacy:
    @given(secret=SECRET, data=st.data())
    @settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_no_page_text_or_reason_reaches_the_files(self, secret, data):
        """Whatever the outputs and reasons hold, no file holds them."""
        filler = data.draw(st.text(min_size=0, max_size=40))
        results = _results(
            range(12),
            text=lambda page, model: f"{filler}{secret}\n{page}{model}{secret[::-1]}",
            reason=lambda page: f'Output A keeps "{secret}"; B drops it\n{filler}',
        )
        values = ["letters" if page % 2 else "forms" for page in range(24)]
        evaluation = _evaluation(results, n_rows=24)
        evaluation.metadata["max_samples"] = 24
        plan = _plan(
            evaluation,
            slice_column="doc_type",
            slice_values=lambda metadata, column: values,
        )  # plan_share runs both checks: no false alarm on any of these texts
        everything = _all_text(plan.files)
        assert secret not in everything
        assert secret[::-1] not in everything
        assert "me/private-source" not in everything
        assert "me/private-results" not in everything

    @given(seed=st.integers(0, 2**32 - 1), length=st.integers(63, 200), data=st.data())
    @settings(max_examples=60, deadline=None)
    def test_any_copied_fragment_of_63_characters_is_found(self, seed, length, data):
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
        board = plan_share(
            _evaluation(_results(range(20), text=lambda page_id, model: page)),
            results_repo="me/private-results",
            alias="v1",
            now=NOW,
        )
        files = dict(board.files)
        files["README.md"] += ("." * 80 + "\n" + "-" * 80 + " " * 80).encode()
        check_no_page_text(files, [page])

    def test_an_extra_column_is_refused(self):
        plan = _plan(_evaluation(_results(range(20))))
        rows = [json.loads(line) for line in plan.files["overall.jsonl"].decode().splitlines()]
        rows[0]["reason"] = "Output A"
        plan.files["overall.jsonl"] = "".join(json.dumps(r) + "\n" for r in rows).encode()
        with pytest.raises(PrivacyError, match="not the allowed columns"):
            check_plan(plan)

    def test_a_value_that_is_not_board_content_is_refused(self):
        plan = _plan(_evaluation(_results(range(20))))
        rows = [json.loads(line) for line in plan.files["overall.jsonl"].decode().splitlines()]
        rows[1]["model"] = "a line from a page"
        plan.files["overall.jsonl"] = "".join(json.dumps(r) + "\n" for r in rows).encode()
        with pytest.raises(PrivacyError, match="'model'"):
            check_plan(plan)

    def test_an_unknown_file_is_refused(self):
        plan = _plan(_evaluation(_results(range(20))))
        plan.files["comparisons.jsonl"] = b"{}\n"
        with pytest.raises(PrivacyError, match="Unexpected files"):
            check_plan(plan)

    def test_settings_never_name_the_source_or_results_repo(self):
        plan = _plan(_evaluation(_results(range(20))))
        everything = _all_text(plan.files)
        assert "private-source" not in everything
        assert "private-results" not in everything
        assert plan.settings["source_fingerprint"] == "cd" * 32


# ---------------------------------------------------------------------------
# Push
# ---------------------------------------------------------------------------


def _missing(repo_id):
    response = SimpleNamespace(status_code=404, headers={}, request=None)
    return RepositoryNotFoundError(f"{repo_id} not found", response=response)


class RecordingApi:
    """Records every Hub call. ``repos`` maps repo id -> {"private", "files", "readme"}."""

    def __init__(self, repos=None):
        self.repos = repos or {}
        self.calls: list[tuple] = []

    def list_repo_files(self, repo_id, repo_type=None):
        self.calls.append(("list_repo_files", repo_id))
        if repo_id not in self.repos:
            raise _missing(repo_id)
        return list(self.repos[repo_id]["files"])

    def hf_hub_download(self, repo_id, filename, repo_type=None):
        self.calls.append(("hf_hub_download", repo_id, filename))
        path = self.repos[repo_id]["tmp"] / filename
        path.write_text(self.repos[repo_id]["readme"])
        return str(path)

    def create_repo(self, repo_id, repo_type=None, private=None, exist_ok=False):
        self.calls.append(("create_repo", repo_id, private))
        self.repos[repo_id] = {"private": bool(private), "files": [".gitattributes"]}

    def create_commit(self, repo_id, repo_type=None, operations=(), commit_message=""):
        self.calls.append(("create_commit", repo_id, list(operations)))

    def repo_info(self, repo_id, repo_type=None):
        self.calls.append(("repo_info", repo_id))
        return SimpleNamespace(private=self.repos[repo_id]["private"])


class TestPush:
    def _plan(self, **kwargs):
        return _plan(_evaluation(_results(range(20))), **kwargs)

    def test_a_failed_check_makes_no_hub_call(self):
        plan = self._plan()
        plan.files["README.md"] += plan.private_texts[0].encode() * 3
        api = RecordingApi()
        with pytest.raises(PrivacyError):
            push_share(plan, "me/public-board", results_repo="me/private-results", api=api)
        assert api.calls == []

    def test_a_new_repo_is_created_then_written_in_one_commit(self):
        api = RecordingApi()
        result = push_share(
            self._plan(), "me/public-board", results_repo="me/private-results", api=api
        )
        assert [call[0] for call in api.calls] == [
            "list_repo_files",
            "create_repo",
            "create_commit",
            "repo_info",
        ]
        assert api.calls[1] == ("create_repo", "me/public-board", False)
        operations = api.calls[2][2]
        assert sorted(op.path_in_repo for op in operations) == [
            "README.md",
            "board.json",
            "overall.jsonl",
        ]
        assert all(isinstance(op, CommitOperationAdd) for op in operations)
        assert result.created and not result.private

    def test_private_flag_creates_a_private_repo(self):
        api = RecordingApi()
        result = push_share(
            self._plan(), "me/board", results_repo="me/private-results", private=True, api=api
        )
        assert api.calls[1] == ("create_repo", "me/board", True)
        assert result.private

    def test_an_existing_private_board_stays_private(self, tmp_path):
        api = RecordingApi(
            {
                "me/board": {
                    "private": True,
                    "files": [".gitattributes", "README.md", "overall.jsonl"],
                    "readme": "---\ntags:\n  - ocr-bench\n  - ocr-bench-share\n---\n",
                    "tmp": tmp_path,
                }
            }
        )
        result = push_share(self._plan(), "me/board", results_repo="me/private-results", api=api)
        assert "create_repo" not in [call[0] for call in api.calls]
        assert result.private and not result.created

    def test_stale_slice_file_is_deleted(self, tmp_path):
        api = RecordingApi(
            {
                "me/board": {
                    "private": False,
                    "files": [".gitattributes", "README.md", "overall.jsonl", "slices.jsonl"],
                    "readme": "---\ntags:\n  - ocr-bench-share\n---\n",
                    "tmp": tmp_path,
                }
            }
        )
        push_share(self._plan(), "me/board", results_repo="me/private-results", api=api)
        commit = next(call for call in api.calls if call[0] == "create_commit")
        deletes = [op.path_in_repo for op in commit[2] if isinstance(op, CommitOperationDelete)]
        assert deletes == ["slices.jsonl"]

    def test_a_repo_share_did_not_write_is_refused(self, tmp_path):
        api = RecordingApi(
            {
                "me/results": {
                    "private": False,
                    "files": [".gitattributes", "README.md", "comparisons/train-0.parquet"],
                    "readme": "---\ntags:\n  - ocr-bench\n  - leaderboard\n---\n",
                    "tmp": tmp_path,
                }
            }
        )
        with pytest.raises(ShareError, match="was not written by `ocr-bench share`"):
            push_share(self._plan(), "me/results", results_repo="me/private-results", api=api)
        assert "create_commit" not in [call[0] for call in api.calls]

    @pytest.mark.parametrize("dest", ["me/private-results", "me/private-source"])
    def test_the_results_or_source_repo_is_refused(self, dest):
        api = RecordingApi()
        with pytest.raises(ShareError, match="separate repo"):
            push_share(self._plan(), dest, results_repo="me/private-results", api=api)
        assert api.calls == []


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestCli:
    def test_alias_is_required(self, capsys):
        with pytest.raises(SystemExit):
            cli.build_parser().parse_args(["share", "me/results", "me/board"])

    def test_dry_run_writes_files_and_pushes_nothing(self, monkeypatch, tmp_path):
        evaluation = _evaluation(_results(range(20)))
        monkeypatch.setattr(share, "load_private_evaluation", lambda repo: evaluation)

        def no_push(*args, **kwargs):
            raise AssertionError("dry run must not push")

        monkeypatch.setattr(share, "push_share", no_push)
        out = tmp_path / "board"
        argv = ["ocr-bench", "share", "me/private-results", "me/board", "--alias", "v1"]
        monkeypatch.setattr(sys, "argv", [*argv, "--dry-run", "--output-dir", str(out)])
        cli.main()
        assert sorted(p.name for p in out.iterdir()) == [
            "README.md",
            "board.json",
            "overall.jsonl",
        ]
        assert "# OCR board: v1" in (out / "README.md").read_text()

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
