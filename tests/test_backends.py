"""Tests for judge backends."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from ocr_bench.backends import (
    InferenceProviderJudge,
    OpenAICompatibleJudge,
    aggregate_jury_votes,
    parse_judge_spec,
)
from ocr_bench.judge import Comparison


def _make_comparison(idx: int = 0) -> Comparison:
    return Comparison(
        sample_idx=idx,
        model_a="model-a",
        model_b="model-b",
        col_a="col_a",
        col_b="col_b",
        swapped=False,
        messages=[{"role": "user", "content": "test"}],
    )


# ---------------------------------------------------------------------------
# parse_judge_spec
# ---------------------------------------------------------------------------


class TestParseJudgeSpec:
    def test_bare_model(self):
        backend = parse_judge_spec("org/model-name")
        assert isinstance(backend, InferenceProviderJudge)
        assert backend.model == "org/model-name"
        assert backend.name == "org/model-name"

    def test_provider_model(self):
        backend = parse_judge_spec("novita:org/model-name")
        assert isinstance(backend, InferenceProviderJudge)
        assert backend.model == "org/model-name"
        assert "novita" in backend.name

    def test_http_url(self):
        backend = parse_judge_spec("http://localhost:8000/v1")
        assert isinstance(backend, OpenAICompatibleJudge)
        assert "localhost" in backend.name

    def test_https_url(self):
        backend = parse_judge_spec("https://api.example.com/v1")
        assert isinstance(backend, OpenAICompatibleJudge)


# ---------------------------------------------------------------------------
# aggregate_jury_votes
# ---------------------------------------------------------------------------


class TestAggregateJuryVotes:
    def test_majority_vote(self):
        results = [
            [{"winner": "A", "reason": "better"}],
            [{"winner": "A", "reason": "more complete"}],
            [{"winner": "B", "reason": "disagree"}],
        ]
        agg = aggregate_jury_votes(results, ["j1", "j2", "j3"])
        assert len(agg) == 1
        assert agg[0]["winner"] == "A"
        assert agg[0]["agreement"] == "2/3"

    def test_unanimity(self):
        results = [
            [{"winner": "B", "reason": "r1"}],
            [{"winner": "B", "reason": "r2"}],
        ]
        agg = aggregate_jury_votes(results, ["j1", "j2"])
        assert agg[0]["winner"] == "B"
        assert agg[0]["agreement"] == "2/2"

    def test_single_judge(self):
        results = [[{"winner": "tie", "reason": "equal"}]]
        agg = aggregate_jury_votes(results, ["j1"])
        assert agg[0]["winner"] == "tie"
        assert agg[0]["agreement"] == "1/1"

    def test_no_valid_votes(self):
        results = [[{}], [{}]]
        agg = aggregate_jury_votes(results, ["j1", "j2"])
        assert agg[0]["winner"] == "tie"
        assert agg[0]["agreement"] == "0/0"

    def test_empty_results(self):
        agg = aggregate_jury_votes([], [])
        assert agg == []

    def test_multiple_comparisons(self):
        results = [
            [{"winner": "A", "reason": "r1"}, {"winner": "B", "reason": "r2"}],
            [{"winner": "A", "reason": "r3"}, {"winner": "A", "reason": "r4"}],
        ]
        agg = aggregate_jury_votes(results, ["j1", "j2"])
        assert len(agg) == 2
        assert agg[0]["winner"] == "A"
        assert agg[1]["agreement"] in ("1/2", "1/2")  # tie broken by Counter.most_common


# ---------------------------------------------------------------------------
# InferenceProviderJudge
# ---------------------------------------------------------------------------


class TestInferenceProviderJudge:
    @patch("ocr_bench.backends.InferenceClient")
    def test_judge_parses_response(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"winner": "A", "reason": "better"}'
        mock_client.chat_completion.return_value = mock_response

        judge = InferenceProviderJudge(model="test-model", provider="novita")
        judge.client = mock_client

        comp = _make_comparison()
        results = judge.judge([comp])
        assert len(results) == 1
        assert results[0]["winner"] == "A"

    @patch("ocr_bench.backends.InferenceClient")
    def test_judge_multiple_comparisons(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        responses = []
        for winner in ["A", "B", "tie"]:
            r = MagicMock()
            r.choices = [MagicMock()]
            r.choices[0].message.content = f'{{"winner": "{winner}", "reason": "test"}}'
            responses.append(r)

        mock_client.chat_completion.side_effect = responses

        judge = InferenceProviderJudge(model="test-model")
        judge.client = mock_client

        comps = [_make_comparison(i) for i in range(3)]
        results = judge.judge(comps)
        assert [r["winner"] for r in results] == ["A", "B", "tie"]


# ---------------------------------------------------------------------------
# OpenAICompatibleJudge
# ---------------------------------------------------------------------------


class TestOpenAICompatibleJudge:
    @patch("ocr_bench.backends.OpenAI")
    def test_judge_parses_response(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"winner": "B", "reason": "more complete"}'
        mock_client.chat.completions.create.return_value = mock_response

        judge = OpenAICompatibleJudge(base_url="http://localhost:8000/v1")
        judge.client = mock_client

        comp = _make_comparison()
        results = judge.judge([comp])
        assert len(results) == 1
        assert results[0]["winner"] == "B"

    @patch("ocr_bench.backends.OpenAI")
    def test_passes_guided_json(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"winner": "tie", "reason": "equal"}'
        mock_client.chat.completions.create.return_value = mock_response

        judge = OpenAICompatibleJudge(base_url="http://localhost:8000/v1")
        judge.client = mock_client

        judge.judge([_make_comparison()])
        call_kwargs = mock_client.chat.completions.create.call_args
        assert "guided_json" in call_kwargs.kwargs.get("extra_body", {})
