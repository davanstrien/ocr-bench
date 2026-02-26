"""Tests for the publish (Space deployment) CLI command."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from ocr_bench.cli import cmd_publish


def _make_args(results: str, space: str | None = None, private: bool = False):
    args = MagicMock()
    args.results = results
    args.space = space
    args.private = private
    return args


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
