"""Tests for the publish (Space deployment) CLI command."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ocr_bench.cli import cmd_publish


def _make_args(results: str, space: str | None = None, private: bool = False):
    args = MagicMock()
    args.results = results
    args.space = space
    args.private = private
    return args


@pytest.fixture(autouse=True)
def hub_stubs():
    """Keep publish tests hermetic: stub the Hub reads/writes cmd_publish makes
    beyond duplicate_space/add_space_variable (metadata config read + Space card
    metadata write). Yields both mocks so tests can drive/assert them."""
    with (
        patch("ocr_bench.cli.load_existing_metadata", return_value=[]) as mock_meta,
        patch("huggingface_hub.metadata_update") as mock_md,
    ):
        yield mock_meta, mock_md


@patch("huggingface_hub.HfApi")
def test_publish_default_space_id(mock_api_cls):
    """Default space id is {results}-viewer."""
    mock_api = mock_api_cls.return_value
    cmd_publish(_make_args("user/my-results"))

    mock_api.duplicate_space.assert_called_once()
    call_kwargs = mock_api.duplicate_space.call_args
    assert call_kwargs.kwargs["to_id"] == "user/my-results-viewer"


@patch("huggingface_hub.HfApi")
def test_publish_explicit_space_id(mock_api_cls):
    """Explicit --space overrides the default."""
    mock_api = mock_api_cls.return_value
    cmd_publish(_make_args("user/my-results", space="user/custom-space"))

    call_kwargs = mock_api.duplicate_space.call_args
    assert call_kwargs.kwargs["to_id"] == "user/custom-space"


@patch("huggingface_hub.HfApi")
def test_publish_sets_repos_variable(mock_api_cls):
    """REPOS env var is set to the results repo."""
    mock_api = mock_api_cls.return_value
    cmd_publish(_make_args("user/my-results"))

    mock_api.add_space_variable.assert_called_once_with(
        repo_id="user/my-results-viewer", key="REPOS", value="user/my-results"
    )


@patch("huggingface_hub.HfApi")
def test_publish_uses_template(mock_api_cls):
    """Duplicates from the correct template Space."""
    mock_api = mock_api_cls.return_value
    cmd_publish(_make_args("user/my-results"))

    call_kwargs = mock_api.duplicate_space.call_args
    assert call_kwargs.kwargs["from_id"] == "davanstrien/ocr-bench-space-template"
    assert call_kwargs.kwargs["exist_ok"] is True


@patch("huggingface_hub.HfApi")
def test_publish_private_flag(mock_api_cls):
    """--private is forwarded to duplicate_space."""
    mock_api = mock_api_cls.return_value
    cmd_publish(_make_args("user/my-results", private=True))

    call_kwargs = mock_api.duplicate_space.call_args
    assert call_kwargs.kwargs["private"] is True


@patch("huggingface_hub.HfApi")
def test_publish_crosslinks_results_and_source(mock_api_cls, hub_stubs):
    """Space card gets a datasets list linking both results and source (issue #38)."""
    mock_meta, mock_md = hub_stubs
    mock_meta.return_value = [{"source_dataset": "user/source-images"}]

    cmd_publish(_make_args("user/my-results"))

    mock_md.assert_called_once()
    space_id, payload = mock_md.call_args.args
    assert space_id == "user/my-results-viewer"
    assert payload["datasets"] == ["user/my-results", "user/source-images"]
    assert mock_md.call_args.kwargs["repo_type"] == "space"
    assert mock_md.call_args.kwargs["overwrite"] is True


@patch("huggingface_hub.HfApi")
def test_publish_datasets_filters_missing_source(mock_api_cls, hub_stubs):
    """No resolvable source → datasets list is just the results repo (Nones filtered)."""
    mock_meta, mock_md = hub_stubs
    mock_meta.return_value = []  # no metadata config → no source dataset

    cmd_publish(_make_args("user/my-results"))

    _, payload = mock_md.call_args.args
    assert payload["datasets"] == ["user/my-results"]


@patch("huggingface_hub.HfApi")
def test_publish_datasets_dedups_when_source_equals_results(mock_api_cls, hub_stubs):
    """A source_dataset equal to results doesn't duplicate the link."""
    mock_meta, mock_md = hub_stubs
    mock_meta.return_value = [{"source_dataset": "user/my-results"}]

    cmd_publish(_make_args("user/my-results"))

    _, payload = mock_md.call_args.args
    assert payload["datasets"] == ["user/my-results"]


@patch("huggingface_hub.HfApi")
def test_publish_sets_title(mock_api_cls, hub_stubs):
    """Space card gets a per-benchmark title derived from the results repo name."""
    _, mock_md = hub_stubs

    cmd_publish(_make_args("user/my-results"))

    _, payload = mock_md.call_args.args
    assert payload["title"] == "OCR Bench — my-results"
