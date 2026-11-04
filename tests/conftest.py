"""Shared test fixtures."""

import pytest

from ocr_bench.publish import RepoVisibility


@pytest.fixture(autouse=True)
def _offline_destination_repos(monkeypatch):
    """Keep CLI tests off the Hub when ``run``/``judge`` prepare their destination repo.

    ``ocr_bench.cli.ensure_dataset_repo`` calls ``HfApi.repo_info`` and may call
    ``create_repo``. Tests that exercise it patch it themselves or call
    ``ocr_bench.publish.ensure_dataset_repo`` with a fake API.
    """
    calls: list[tuple[str, dict]] = []

    def fake_ensure(repo_id, **kwargs):
        calls.append((repo_id, kwargs))
        return RepoVisibility(private=False)

    monkeypatch.setattr("ocr_bench.cli.ensure_dataset_repo", fake_ensure)
    return calls


@pytest.fixture(autouse=True)
def _offline_source_repo_check(monkeypatch):
    """Keep ``judge`` and ``audit`` tests off the Hub when they check the source repo exists.

    ``require_dataset_repo`` calls ``HfApi.repo_info``. Tests that exercise it
    call ``ocr_bench.dataset.require_dataset_repo`` with a fake API, or patch
    these names themselves.
    """
    calls: list[str] = []

    def fake_require(repo_id, api=None):
        calls.append(repo_id)

    monkeypatch.setattr("ocr_bench.cli.require_dataset_repo", fake_require)
    monkeypatch.setattr("ocr_bench.integrity.require_dataset_repo", fake_require)
    return calls
