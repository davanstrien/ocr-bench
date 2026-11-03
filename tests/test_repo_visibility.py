"""Destination repo visibility: new repos for private sources must not become public."""

from types import SimpleNamespace

import pytest
from huggingface_hub.errors import RepositoryNotFoundError

from ocr_bench import cli
from ocr_bench.cli import build_parser
from ocr_bench.dataset import DatasetError
from ocr_bench.publish import RepoVisibility, ensure_dataset_repo


class FakeApi:
    """Minimal stand-in for HfApi: repo_info and create_repo only."""

    def __init__(self, repos: dict[str, bool]):
        # repo id -> private flag; missing ids raise RepositoryNotFoundError.
        self.repos = dict(repos)
        self.created: list[tuple[str, bool]] = []

    def repo_info(self, repo_id, repo_type=None):
        if repo_id not in self.repos:
            response = SimpleNamespace(status_code=404, headers={}, request=None)
            raise RepositoryNotFoundError(f"{repo_id} not found", response=response)
        return SimpleNamespace(private=self.repos[repo_id])

    def create_repo(self, repo_id, repo_type=None, private=False, exist_ok=False):
        self.created.append((repo_id, private))
        self.repos[repo_id] = private


class TestEnsureDatasetRepo:
    def test_missing_repo_with_flag_is_created_private(self):
        api = FakeApi({"u/src": False})
        result = ensure_dataset_repo("u/out", private=True, source_repo="u/src", api=api)
        assert api.created == [("u/out", True)]
        assert result == RepoVisibility(private=True, created=True, inherited=False)

    def test_missing_repo_inherits_private_source(self):
        api = FakeApi({"u/src": True})
        result = ensure_dataset_repo("u/out", private=False, source_repo="u/src", api=api)
        assert api.created == [("u/out", True)]
        assert result == RepoVisibility(private=True, created=True, inherited=True)

    def test_missing_repo_with_public_source_stays_public(self):
        api = FakeApi({"u/src": False})
        result = ensure_dataset_repo("u/out", private=False, source_repo="u/src", api=api)
        assert api.created == [("u/out", False)]
        assert result == RepoVisibility(private=False, created=True, inherited=False)

    def test_existing_private_repo_is_left_alone(self):
        api = FakeApi({"u/src": False, "u/out": True})
        result = ensure_dataset_repo("u/out", private=True, source_repo="u/src", api=api)
        assert api.created == []
        assert result == RepoVisibility(private=True)

    def test_existing_public_repo_with_flag_raises_before_writing(self):
        api = FakeApi({"u/src": False, "u/out": False})
        with pytest.raises(DatasetError, match="already exists and is public"):
            ensure_dataset_repo("u/out", private=True, source_repo="u/src", api=api)
        assert api.created == []

    def test_existing_public_repo_with_private_source_is_reported(self):
        api = FakeApi({"u/src": True, "u/out": False})
        result = ensure_dataset_repo("u/out", private=False, source_repo="u/src", api=api)
        assert api.created == []
        assert result == RepoVisibility(private=False, public_with_private_source=True)

    def test_without_source_only_the_flag_decides(self):
        api = FakeApi({})
        result = ensure_dataset_repo("u/out", private=False, api=api)
        assert api.created == [("u/out", False)]
        assert result.private is False


class TestCliWiring:
    def test_run_prepares_output_repo_before_launch(self, monkeypatch):
        order: list[str] = []
        monkeypatch.setattr(
            cli,
            "ensure_dataset_repo",
            lambda repo_id, **kw: (
                order.append(f"ensure:{repo_id}:{kw['private']}") or RepoVisibility(private=True)
            ),
        )
        monkeypatch.setattr(
            "ocr_bench.run.launch_ocr_jobs", lambda *a, **k: order.append("launch") or []
        )
        args = build_parser().parse_args(
            ["run", "in/ds", "out/repo", "--models", "glm-ocr", "--private", "--no-wait"]
        )
        cli.cmd_run(args)
        assert order == ["ensure:out/repo:True", "launch"]

    def test_run_refusal_stops_before_launch(self, monkeypatch):
        def refuse(repo_id, **kw):
            raise DatasetError("already exists and is public")

        launched: list[str] = []
        monkeypatch.setattr(cli, "ensure_dataset_repo", refuse)
        monkeypatch.setattr(
            "ocr_bench.run.launch_ocr_jobs", lambda *a, **k: launched.append("x") or []
        )
        args = build_parser().parse_args(["run", "in/ds", "out/repo", "--private"])
        with pytest.raises(DatasetError):
            cli.cmd_run(args)
        assert launched == []

    def test_run_dry_run_touches_no_repo(self, _offline_destination_repos):
        args = build_parser().parse_args(["run", "in/ds", "out/repo", "--dry-run"])
        cli.cmd_run(args)
        assert _offline_destination_repos == []

    def test_judge_prepares_results_repo_with_source(self, monkeypatch):
        seen: list[tuple[str, dict]] = []

        def record(repo_id, **kw):
            seen.append((repo_id, kw))
            raise DatasetError("stop here")  # stop before any dataset loading

        monkeypatch.setattr(cli, "ensure_dataset_repo", record)
        args = build_parser().parse_args(["judge", "u/ocr", "--private"])
        with pytest.raises(DatasetError, match="stop here"):
            cli.cmd_judge(args)
        assert seen == [("u/ocr-results", {"private": True, "source_repo": "u/ocr"})]

    def test_judge_no_publish_skips_destination(self, monkeypatch, _offline_destination_repos):
        def stop_loading(*args, **kwargs):
            raise DatasetError("stop at load")

        monkeypatch.setattr(cli, "load_flat_dataset", stop_loading)
        args = build_parser().parse_args(
            ["judge", "u/ocr", "--columns", "a", "b", "--no-publish", "--private"]
        )
        with pytest.raises(DatasetError, match="stop at load"):
            cli.cmd_judge(args)
        assert _offline_destination_repos == []

    def test_bench_forwards_private_to_run_and_judge(self, monkeypatch):
        from ocr_bench.run import JobRun

        calls: list[tuple[str, object]] = []
        monkeypatch.setattr(
            cli,
            "cmd_run",
            lambda a: calls.append(("run", a)) or [JobRun("glm-ocr", "j", "u", "completed")],
        )
        monkeypatch.setattr(cli, "cmd_judge", lambda a: calls.append(("judge", a)))
        monkeypatch.setattr(cli, "cmd_view", lambda a: calls.append(("view", a)))
        cli.cmd_bench(build_parser().parse_args(["bench", "u/imgs", "u/out", "--private"]))
        run_args, judge_args = calls[0][1], calls[1][1]
        assert run_args.private is True
        assert judge_args.private is True

    def test_bench_defaults_to_not_private(self, monkeypatch):
        from ocr_bench.run import JobRun

        calls: list[tuple[str, object]] = []
        monkeypatch.setattr(
            cli,
            "cmd_run",
            lambda a: calls.append(("run", a)) or [JobRun("glm-ocr", "j", "u", "completed")],
        )
        monkeypatch.setattr(cli, "cmd_judge", lambda a: calls.append(("judge", a)))
        monkeypatch.setattr(cli, "cmd_view", lambda a: calls.append(("view", a)))
        cli.cmd_bench(build_parser().parse_args(["bench", "u/imgs", "u/out"]))
        assert calls[0][1].private is False
        assert calls[1][1].private is False


class TestPublishSpaceVisibility:
    def _args(self, private=False):
        return SimpleNamespace(results="u/results", space=None, private=private)

    def _api(self, monkeypatch, results_private):
        captured: dict = {}

        class Api:
            def repo_info(self, repo_id, repo_type=None):
                return SimpleNamespace(private=results_private)

            def duplicate_space(self, **kwargs):
                captured.update(kwargs)

            def add_space_variable(self, **kwargs):
                pass

        monkeypatch.setattr("huggingface_hub.HfApi", Api)
        monkeypatch.setattr(cli, "load_existing_metadata", lambda repo: [])
        monkeypatch.setattr("huggingface_hub.metadata_update", lambda *a, **k: None)
        return captured

    def test_private_results_give_private_space(self, monkeypatch):
        captured = self._api(monkeypatch, results_private=True)
        cli.cmd_publish(self._args())
        assert captured["private"] is True

    def test_public_results_keep_template_visibility(self, monkeypatch):
        captured = self._api(monkeypatch, results_private=False)
        cli.cmd_publish(self._args())
        assert captured["private"] is None
