"""Default judge spec and the resume hint shown after the default changed."""

import json
from unittest.mock import patch

from ocr_bench.backends import DEFAULT_JUDGE, InferenceProviderJudge, parse_judge_spec
from ocr_bench.cli import _previous_judge_hint

OLD_DEFAULT = "novita:Qwen/Qwen3.5-35B-A3B"


def test_default_judge_is_pinned_provider_spec():
    assert DEFAULT_JUDGE == "novita:Qwen/Qwen3.8-27B"
    with patch("ocr_bench.backends.InferenceClient") as client:
        backend = parse_judge_spec(DEFAULT_JUDGE)
    assert isinstance(backend, InferenceProviderJudge)
    assert backend.name == DEFAULT_JUDGE
    client.assert_called_once_with(model="Qwen/Qwen3.8-27B", provider="novita")


class TestPreviousJudgeHint:
    def test_old_default_gets_exact_flag(self):
        meta = [{"judge_models": json.dumps([OLD_DEFAULT])}]
        hint = _previous_judge_hint(meta, None)
        assert f"--model {OLD_DEFAULT}" in hint
        assert DEFAULT_JUDGE in hint

    def test_list_value_is_accepted(self):
        hint = _previous_judge_hint([{"judge_models": [OLD_DEFAULT]}], None)
        assert f"--model {OLD_DEFAULT}" in hint

    def test_latest_metadata_row_wins(self):
        meta = [
            {"judge_models": json.dumps(["novita:org/older"])},
            {"judge_models": json.dumps([OLD_DEFAULT])},
        ]
        hint = _previous_judge_hint(meta, None)
        assert OLD_DEFAULT in hint
        assert "org/older" not in hint

    def test_jury_lists_every_flag(self):
        jury = [OLD_DEFAULT, "novita:org/other"]
        hint = _previous_judge_hint([{"judge_models": json.dumps(jury)}], None)
        assert f"--model {OLD_DEFAULT} --model novita:org/other" in hint

    def test_endpoint_names_get_generic_advice(self):
        meta = [{"judge_models": json.dumps(["openai-endpoint-0123456789ab"])}]
        hint = _previous_judge_hint(meta, None)
        assert "openai-endpoint-0123456789ab" in hint
        assert "--model openai-endpoint" not in hint
        assert "same --model spec(s)" in hint

    def test_no_hint_when_models_were_requested(self):
        meta = [{"judge_models": json.dumps([OLD_DEFAULT])}]
        assert _previous_judge_hint(meta, ["novita:org/x"]) == ""

    def test_no_hint_when_repo_used_current_default(self):
        meta = [{"judge_models": json.dumps([DEFAULT_JUDGE])}]
        assert _previous_judge_hint(meta, None) == ""

    def test_no_hint_without_metadata_or_with_bad_value(self):
        assert _previous_judge_hint([], None) == ""
        assert _previous_judge_hint([{"judge_models": "not json"}], None) == ""
        assert _previous_judge_hint([{"judge_models": json.dumps([])}], None) == ""
        assert _previous_judge_hint([{}], None) == ""
