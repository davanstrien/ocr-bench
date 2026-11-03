"""Default judge spec and the resume hint shown after the default changed."""

import json
from unittest.mock import patch

from ocr_bench.backends import DEFAULT_JUDGE, InferenceProviderJudge, parse_judge_spec
from ocr_bench.cli import _previous_judge_hint
from ocr_bench.publish import hash_judge_specs

OLD_DEFAULT = "novita:Qwen/Qwen3.5-35B-A3B"


def _meta(names, spec_hashes=None):
    """A metadata row as publish writes it: JSON strings for both lists."""
    hashes = hash_judge_specs(names) if spec_hashes is None else spec_hashes
    return {"judge_models": json.dumps(names), "judge_spec_hashes": json.dumps(hashes)}


def test_default_judge_is_pinned_provider_spec():
    assert DEFAULT_JUDGE == "novita:Qwen/Qwen3.8-27B"
    with patch("ocr_bench.backends.InferenceClient") as client:
        backend = parse_judge_spec(DEFAULT_JUDGE)
    assert isinstance(backend, InferenceProviderJudge)
    assert backend.name == DEFAULT_JUDGE
    client.assert_called_once_with(model="Qwen/Qwen3.8-27B", provider="novita")


class TestPreviousJudgeHint:
    def test_old_default_gets_judge_and_bench_flags(self):
        hint = _previous_judge_hint([_meta([OLD_DEFAULT])], None)
        assert f"--model {OLD_DEFAULT}" in hint
        assert f"--judge-model {OLD_DEFAULT}" in hint
        assert DEFAULT_JUDGE in hint
        assert "last completed run" in hint

    def test_bare_spec_with_matching_hash_gets_exact_flag(self):
        hint = _previous_judge_hint([_meta(["Qwen/Qwen3.5-35B-A3B"])], None)
        assert "--model Qwen/Qwen3.5-35B-A3B" in hint

    def test_list_values_are_accepted(self):
        row = {"judge_models": [OLD_DEFAULT], "judge_spec_hashes": hash_judge_specs([OLD_DEFAULT])}
        assert f"--model {OLD_DEFAULT}" in _previous_judge_hint([row], None)

    def test_latest_metadata_row_wins(self):
        hint = _previous_judge_hint([_meta(["novita:org/older"]), _meta([OLD_DEFAULT])], None)
        assert OLD_DEFAULT in hint
        assert "org/older" not in hint

    def test_jury_lists_every_flag(self):
        jury = [OLD_DEFAULT, "novita:org/other"]
        hint = _previous_judge_hint([_meta(jury)], None)
        assert f"--model {OLD_DEFAULT} --model novita:org/other" in hint

    def test_endpoint_alias_without_matching_hash_gets_generic_advice(self):
        # An endpoint spec like http://host/v1/:novita:org/model is recorded as
        # "novita:org/model"; its hash is of the URL spec, so no exact flag.
        url_hash = hash_judge_specs(["http://localhost:8000/v1/:novita:org/model"])
        hint = _previous_judge_hint([_meta(["novita:org/model"], url_hash)], None)
        assert "novita:org/model" in hint
        assert "--model novita:org/model" not in hint
        assert "same judge spec(s)" in hint

    def test_missing_hashes_get_generic_advice(self):
        row = {"judge_models": json.dumps([OLD_DEFAULT])}
        hint = _previous_judge_hint([row], None)
        assert OLD_DEFAULT in hint
        assert f"--model {OLD_DEFAULT}" not in hint

    def test_no_hint_when_models_were_requested(self):
        assert _previous_judge_hint([_meta([OLD_DEFAULT])], ["novita:org/x"]) == ""

    def test_no_hint_when_repo_used_current_default(self):
        assert _previous_judge_hint([_meta([DEFAULT_JUDGE])], None) == ""

    def test_no_hint_without_metadata_or_with_bad_values(self):
        assert _previous_judge_hint([], None) == ""
        assert _previous_judge_hint([{"judge_models": "not json"}], None) == ""
        assert _previous_judge_hint([{"judge_models": json.dumps([])}], None) == ""
        assert _previous_judge_hint([{"judge_models": json.dumps([None])}], None) == ""
        assert _previous_judge_hint([{"judge_models": json.dumps([""])}], None) == ""
        assert _previous_judge_hint([{}], None) == ""
