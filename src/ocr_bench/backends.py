"""Judge backends — API-based (HF Inference Providers, OpenAI-compatible)."""

from __future__ import annotations

import abc
from collections import Counter
from typing import Any

import stamina
import structlog
from huggingface_hub import InferenceClient
from openai import OpenAI

from ocr_bench.judge import JUDGE_SCHEMA, Comparison, parse_judge_output

logger = structlog.get_logger()

# Retry on these exception types with exponential backoff + jitter.
_RETRYABLE = (Exception,)


class JudgeBackend(abc.ABC):
    """Base class for judge backends."""

    name: str

    @abc.abstractmethod
    def judge(self, comparisons: list[Comparison]) -> list[dict[str, str]]:
        """Run the judge on a list of comparisons.

        Returns a list of parsed results (one per comparison).
        Each result is a dict with ``winner`` and ``reason`` keys,
        or an empty dict on failure.
        """


DEFAULT_MAX_TOKENS = 512


class InferenceProviderJudge(JudgeBackend):
    """HF Inference Providers backend (Novita, Together, etc.)."""

    def __init__(
        self, model: str, provider: str | None = None, max_tokens: int = DEFAULT_MAX_TOKENS,
    ):
        self.name = f"{provider + ':' if provider else ''}{model}"
        self.model = model
        self.max_tokens = max_tokens
        self.client = InferenceClient(model=model, provider=provider)  # type: ignore[invalid-argument-type]

    def judge(self, comparisons: list[Comparison]) -> list[dict[str, str]]:
        results: list[dict[str, str]] = []
        for comp in comparisons:
            result = self._call_single(comp)
            results.append(result)
        return results

    @stamina.retry(on=_RETRYABLE, attempts=3)
    def _call_single(self, comp: Comparison) -> dict[str, str]:
        response = self.client.chat_completion(  # type: ignore[no-matching-overload]
            messages=comp.messages,
            max_tokens=self.max_tokens,
            temperature=0.0,
            response_format={"type": "json_object"},
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        raw = response.choices[0].message.content.strip()
        result = parse_judge_output(raw)
        if not result:
            logger.warning("empty_parse", backend=self.name, sample=comp.sample_idx)
        return result


class OpenAICompatibleJudge(JudgeBackend):
    """OpenAI-compatible endpoint (local vLLM server, Ollama, HF IE, etc.)."""

    def __init__(
        self,
        base_url: str,
        model: str = "default",
        max_tokens: int = DEFAULT_MAX_TOKENS,
        api_key: str = "not-needed",
        extra_body: dict | None = None,
    ):
        self.name = f"openai@{base_url}"
        self.model = model
        self.max_tokens = max_tokens
        self.extra_body = extra_body if extra_body is not None else {"guided_json": JUDGE_SCHEMA}
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def judge(self, comparisons: list[Comparison]) -> list[dict[str, str]]:
        results: list[dict[str, str]] = []
        for comp in comparisons:
            result = self._call_single(comp)
            results.append(result)
        return results

    @stamina.retry(on=_RETRYABLE, attempts=3)
    def _call_single(self, comp: Comparison) -> dict[str, str]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=comp.messages,  # type: ignore[invalid-argument-type]
            max_tokens=self.max_tokens,
            temperature=0.0,
            extra_body=self.extra_body,
        )
        raw = response.choices[0].message.content.strip()
        result = parse_judge_output(raw)
        if not result:
            logger.warning("empty_parse", backend=self.name, sample=comp.sample_idx)
        return result


# ---------------------------------------------------------------------------
# Spec parsing
# ---------------------------------------------------------------------------

DEFAULT_JUDGE = "novita:moonshotai/Kimi-K2.5"


def parse_judge_spec(spec: str, max_tokens: int = DEFAULT_MAX_TOKENS) -> JudgeBackend:
    """Parse a judge specification string into a backend.

    Formats:
      - ``"https://xxx.endpoints.huggingface.cloud"`` → :class:`OpenAICompatibleJudge`
        (HF Inference Endpoints, OpenAI-compatible with HF token auth)
      - ``"http://..."`` or ``"https://..."`` (other) → :class:`OpenAICompatibleJudge`
      - ``"provider:org/model"`` (colon before first ``/``) → :class:`InferenceProviderJudge`
      - anything else → :class:`InferenceProviderJudge` (no provider)
    """
    if spec.startswith("http://") or spec.startswith("https://"):
        # HF Inference Endpoints — OpenAI-compatible, auth via HF token
        if ".endpoints.huggingface." in spec:
            from huggingface_hub import get_token

            base_url = spec.rstrip("/")
            if not base_url.endswith("/v1"):
                base_url += "/v1"
            token = get_token() or "not-needed"
            return OpenAICompatibleJudge(
                base_url=base_url,
                api_key=token,
                max_tokens=max_tokens,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
        return OpenAICompatibleJudge(base_url=spec, max_tokens=max_tokens)

    if ":" in spec:
        # provider:model format — colon must come before first slash
        colon_idx = spec.index(":")
        slash_idx = spec.find("/")
        if slash_idx == -1 or colon_idx < slash_idx:
            provider, model = spec.split(":", 1)
            return InferenceProviderJudge(model=model, provider=provider, max_tokens=max_tokens)

    return InferenceProviderJudge(model=spec, max_tokens=max_tokens)


# ---------------------------------------------------------------------------
# Jury aggregation
# ---------------------------------------------------------------------------


def aggregate_jury_votes(
    all_results: list[list[dict[str, str]]],
    judge_names: list[str],
) -> list[dict[str, Any]]:
    """Aggregate votes from multiple judges using majority voting.

    Args:
        all_results: List of result lists, one per judge. Each inner list
            has one dict per comparison.
        judge_names: Names of the judges (same order as *all_results*).

    Returns:
        Aggregated results with ``winner``, ``reason``, and ``agreement`` fields.
    """
    if not all_results:
        return []

    n_comparisons = len(all_results[0])
    n_judges = len(all_results)
    aggregated: list[dict[str, Any]] = []

    for i in range(n_comparisons):
        votes: list[str] = []
        reasons: list[str] = []
        for j in range(n_judges):
            result = all_results[j][i] if i < len(all_results[j]) else {}
            winner = result.get("winner", "")
            if winner:
                votes.append(winner)
                reasons.append(f"{judge_names[j]}: {result.get('reason', '')}")

        if not votes:
            aggregated.append({"winner": "tie", "reason": "no valid votes", "agreement": "0/0"})
            continue

        counter = Counter(votes)
        majority_winner, majority_count = counter.most_common(1)[0]
        agreement = f"{majority_count}/{len(votes)}"

        aggregated.append({
            "winner": majority_winner,
            "reason": "; ".join(reasons),
            "agreement": agreement,
        })

    return aggregated
