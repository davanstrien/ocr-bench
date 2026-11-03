"""Destination repo visibility: data derived from restricted sources must not become public."""

from types import SimpleNamespace

import pytest
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

from ocr_bench import cli
from ocr_bench.cli import build_parser
from ocr_bench.dataset import DatasetError
from ocr_bench.publish import RepoVisibility, ensure_dataset_repo


def _missing(repo_id, error=RepositoryNotFoundError):
    response = SimpleNamespace(status_code=404, headers={}, request=None)
    return error(f"{repo_id} not found", response=response)


class FakeApi:
    """Stand-in for HfApi with the Hub semantics ensure_dataset_repo relies on.

    ``repos`` maps repo id -> {"private": bool, "gated": False|"auto"|"manual"}.
    ``create_repo(private=None)`` uses ``org_default_private``. ``exist_ok=True``
    never changes an existing repo. ``race_public`` lists ids that another
    process creates as public between the lookup and the create.
    """

    def __init__(
        self, repos=None, *, org_default_private=False, race_public=(), gated_no_access=()
    ):
        self.repos = {k: dict(v) for k, v in (repos or {}).items()}
        self.org_default_private = org_default_private
        self.race_public = set(race_public)
        self.gated_no_access = set(gated_no_access)
        self.created: list[tuple[str, bool | None]] = []

    def repo_info(self, repo_id, repo_type=None):
        if repo_id in self.gated_no_access:
            raise _missing(repo_id, GatedRepoError)
        if repo_id not in self.repos:
            raise _missing(repo_id)
        return SimpleNamespace(**self.repos[repo_id])

    def create_repo(self, repo_id, repo_type=None, private=None, exist_ok=False):
        self.created.append((repo_id, private))
        if repo_id in self.race_public:
            self.repos.setdefault(repo_id, {"private": False, "gated": False})
        if repo_id in self.repos:
            return
        actual = self.org_default_private if private is None else private
        self.repos[repo_id] = {"private": actual, "gated": False}


PUBLIC = {"private": False, "gated": False}
PRIVATE = {"private": True, "gated": False}
GATED = {"private": False, "gated": "manual"}


class TestEnsureDatasetRepo:
    def test_missing_repo_with_flag_is_created_private(self):
        api = FakeApi({"u/src": PUBLIC})
        result = ensure_dataset_repo("u/out", private=True, source_repo="u/src", api=api)
        assert api.created == [("u/out", True)]
        assert result == RepoVisibility(private=True, created=True, inherited=False)

    def test_missing_repo_inherits_private_source(self):
        api = FakeApi({"u/src": PRIVATE})
        result = ensure_dataset_repo("u/out", private=False, source_repo="u/src", api=api)
        assert api.created == [("u/out", True)]
        assert result == RepoVisibility(private=True, created=True, inherited=True)

    def test_gated_source_counts_as_restricted(self):
        api = FakeApi({"u/src": GATED})
        result = ensure_dataset_repo("u/out", private=False, source_repo="u/src", api=api)
        assert api.created == [("u/out", True)]
        assert result.private and result.inherited

    def test_public_source_keeps_org_default_visibility(self):
        # private=None lets an organization's "private by default" setting apply.
        api = FakeApi({"u/src": PUBLIC}, org_default_private=True)
        result = ensure_dataset_repo("u/out", private=False, source_repo="u/src", api=api)
        assert api.created == [("u/out", None)]
        assert result == RepoVisibility(private=True, created=True, inherited=False)

    def test_public_source_without_org_default_is_public(self):
        api = FakeApi({"u/src": PUBLIC})
        result = ensure_dataset_repo("u/out", private=False, source_repo="u/src", api=api)
        assert api.created == [("u/out", None)]
        assert result == RepoVisibility(private=False, created=True)

    def test_create_race_with_public_repo_is_refused(self):
        api = FakeApi({"u/src": PUBLIC}, race_public={"u/out"})
        with pytest.raises(DatasetError, match="created public by another process"):
            ensure_dataset_repo("u/out", private=True, source_repo="u/src", api=api)

    def test_existing_private_repo_is_left_alone(self):
        api = FakeApi({"u/src": PUBLIC, "u/out": PRIVATE})
        result = ensure_dataset_repo("u/out", private=True, source_repo="u/src", api=api)
        assert api.created == []
        assert result == RepoVisibility(private=True)

    def test_existing_public_repo_with_flag_raises_before_writing(self):
        api = FakeApi({"u/src": PUBLIC, "u/out": PUBLIC})
        with pytest.raises(DatasetError, match="--private was set"):
            ensure_dataset_repo("u/out", private=True, source_repo="u/src", api=api)
        assert api.created == []

    def test_existing_public_repo_with_private_source_fails_closed(self):
        api = FakeApi({"u/src": PRIVATE, "u/out": PUBLIC})
        with pytest.raises(DatasetError, match="--public"):
            ensure_dataset_repo("u/out", private=False, source_repo="u/src", api=api)
        assert api.created == []

    def test_allow_public_permits_public_repo_for_private_source(self):
        api = FakeApi({"u/src": PRIVATE, "u/out": PUBLIC})
        result = ensure_dataset_repo(
            "u/out", private=False, allow_public=True, source_repo="u/src", api=api
        )
        assert result == RepoVisibility(private=False, public_with_restricted_source=True)

    def test_allow_public_new_repo_uses_org_default(self):
        api = FakeApi({"u/src": PRIVATE})
        result = ensure_dataset_repo(
            "u/out", private=False, allow_public=True, source_repo="u/src", api=api
        )
        assert api.created == [("u/out", None)]
        assert result.public_with_restricted_source is True

    def test_gated_destination_without_access_is_refused(self):
        api = FakeApi({"u/src": PUBLIC}, gated_no_access={"u/out"})
        with pytest.raises(DatasetError, match="gated"):
            ensure_dataset_repo("u/out", private=False, source_repo="u/src", api=api)
        assert api.created == []

    def test_without_source_only_the_flag_decides(self):
        api = FakeApi({})
        result = ensure_dataset_repo("u/out", private=False, api=api)
        assert api.created == [("u/out", None)]
        assert result.private is False


class TestCliWithRealHelper:
    """CLI paths exercising the real ensure_dataset_repo against a fake HfApi."""

    @pytest.fixture
    def fake_api(self, monkeypatch):
        api = FakeApi()
        monkeypatch.setattr("ocr_bench.publish.HfApi", lambda: api)
        monkeypatch.setattr(cli, "ensure_dataset_repo", ensure_dataset_repo)
        return api

    def test_run_refuses_public_output_for_private_input(self, monkeypatch, fake_api):
        fake_api.repos = {"in/ds": PRIVATE, "out/repo": PUBLIC}
        launched: list[str] = []
        monkeypatch.setattr(
            "ocr_bench.run.launch_ocr_jobs", lambda *a, **k: launched.append("x") or []
        )
        args = build_parser().parse_args(["run", "in/ds", "out/repo", "--no-wait"])
        with pytest.raises(DatasetError, match="private or gated"):
            cli.cmd_run(args)
        assert launched == []

    def test_bench_results_conflict_stops_before_any_launch(self, monkeypatch, fake_api):
        fake_api.repos = {"in/ds": PUBLIC, "out/repo-results": PUBLIC}
        calls: list[str] = []
        monkeypatch.setattr(cli, "cmd_run", lambda a: calls.append("run") or [])
        monkeypatch.setattr(cli, "cmd_judge", lambda a: calls.append("judge"))
        args = build_parser().parse_args(["bench", "in/ds", "out/repo", "--private"])
        with pytest.raises(DatasetError, match="--private was set"):
            cli.cmd_bench(args)
        assert calls == []

    def test_bench_no_publish_skips_results_check(self, monkeypatch, fake_api):
        from ocr_bench.run import JobRun

        fake_api.repos = {"in/ds": PUBLIC, "out/repo-results": PUBLIC}
        calls: list[str] = []
        monkeypatch.setattr(
            cli,
            "cmd_run",
            lambda a: calls.append("run") or [JobRun("glm-ocr", "j", "u", "completed")],
        )
        monkeypatch.setattr(cli, "cmd_judge", lambda a: calls.append("judge"))
        args = build_parser().parse_args(
            ["bench", "in/ds", "out/repo", "--private", "--no-publish"]
        )
        cli.cmd_bench(args)
        assert calls == ["run", "judge"]


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

    def test_run_dry_run_touches_no_repo(self, _offline_destination_repos):
        args = build_parser().parse_args(["run", "in/ds", "out/repo", "--dry-run"])
        cli.cmd_run(args)
        assert _offline_destination_repos == []

    def test_private_and_public_together_are_rejected(self, _offline_destination_repos):
        args = build_parser().parse_args(["run", "in/ds", "out/repo", "--private", "--public"])
        with pytest.raises(DatasetError, match="cannot be used together"):
            cli.cmd_run(args)
        assert _offline_destination_repos == []

    def _judge_until_prepare(self, monkeypatch, argv):
        """Run cmd_judge on a tiny in-memory dataset and stop at the repo check."""
        seen: list[tuple[str, dict]] = []

        def record(repo_id, **kw):
            seen.append((repo_id, kw))
            raise DatasetError("stop here")

        ds = [{"image": None, "a": "alpha text", "b": "beta text"}]
        monkeypatch.setattr(
            cli, "load_flat_dataset", lambda *a, **k: (ds, {"a": "model-a", "b": "model-b"})
        )
        monkeypatch.setattr(cli, "ensure_dataset_repo", record)
        monkeypatch.setattr(
            cli, "evaluation_input_fingerprints", lambda *a, **k: ("fp", {"a": "x", "b": "y"})
        )
        with pytest.raises(DatasetError, match="stop here"):
            cli.cmd_judge(build_parser().parse_args(argv))
        return seen

    def test_judge_prepares_results_repo_after_loading_source(self, monkeypatch):
        seen = self._judge_until_prepare(
            monkeypatch, ["judge", "u/ocr", "--columns", "a", "b", "--private"]
        )
        assert seen == [
            (
                "u/ocr-results",
                {"private": True, "allow_public": False, "source_repo": "u/ocr"},
            )
        ]

    def test_judge_bad_source_creates_no_results_repo(
        self, monkeypatch, _offline_destination_repos
    ):
        def stop_loading(*args, **kwargs):
            raise DatasetError("stop at load")

        monkeypatch.setattr(cli, "load_flat_dataset", stop_loading)
        args = build_parser().parse_args(["judge", "u/ocr", "--columns", "a", "b"])
        with pytest.raises(DatasetError, match="stop at load"):
            cli.cmd_judge(args)
        assert _offline_destination_repos == []

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

    def test_bench_forwards_flags_to_run_and_judge(self, monkeypatch):
        from ocr_bench.run import JobRun

        for flag in ("--private", "--public"):
            calls: list[tuple[str, object]] = []
            monkeypatch.setattr(
                cli,
                "cmd_run",
                lambda a, calls=calls: (
                    calls.append(("run", a)) or [JobRun("glm-ocr", "j", "u", "completed")]
                ),
            )
            monkeypatch.setattr(cli, "cmd_judge", lambda a, calls=calls: calls.append(("judge", a)))
            monkeypatch.setattr(cli, "cmd_view", lambda a, calls=calls: calls.append(("view", a)))
            cli.cmd_bench(build_parser().parse_args(["bench", "u/imgs", "u/out", flag]))
            attr = flag.lstrip("-")
            assert getattr(calls[0][1], attr) is True
            assert getattr(calls[1][1], attr) is True

    def test_bench_defaults_to_neither_flag(self, monkeypatch):
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
        for _, sub in calls[:2]:
            assert sub.private is False
            assert sub.public is False


class TestConsoleMessages:
    def _prepare(self, monkeypatch, visibility, capsys):
        monkeypatch.setattr(cli, "ensure_dataset_repo", lambda repo_id, **kw: visibility)
        cli._prepare_destination(
            "u/out", private=False, allow_public=False, source_repo="u/src", label="output"
        )
        return capsys.readouterr().out

    def test_inherited_privacy_is_explained(self, monkeypatch, capsys):
        out = self._prepare(
            monkeypatch, RepoVisibility(private=True, created=True, inherited=True), capsys
        )
        assert "Created private output repo" in out
        assert "private or gated" in out

    def test_public_with_restricted_source_warns(self, monkeypatch, capsys):
        out = self._prepare(
            monkeypatch, RepoVisibility(private=False, public_with_restricted_source=True), capsys
        )
        assert "Warning" in out
        assert "--public" in out


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

    def test_private_results_give_private_space_and_token_note(self, monkeypatch, capsys):
        captured = self._api(monkeypatch, results_private=True)
        cli.cmd_publish(self._args())
        assert captured["private"] is True
        assert "HF_TOKEN" in capsys.readouterr().out

    def test_public_results_keep_template_visibility(self, monkeypatch, capsys):
        captured = self._api(monkeypatch, results_private=False)
        cli.cmd_publish(self._args())
        assert captured["private"] is None
        assert "HF_TOKEN" not in capsys.readouterr().out
