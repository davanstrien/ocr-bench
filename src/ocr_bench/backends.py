"""Judge backends — API-based (HF Inference Providers, OpenAI-compatible)."""

from __future__ import annotations

import abc
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import stamina
import structlog
from huggingface_hub import InferenceClient
from openai import APIConnectionError, OpenAI

from ocr_bench.judge import JUDGE_SCHEMA, Comparison, parse_judge_output

logger = structlog.get_logger()

# Number of attempts for a transient judge call (connection/timeout/429/5xx).
_RETRY_ATTEMPTS = 5

# Transport-level failures worth retrying — no HTTP status (connection reset,
# DNS failure, read timeout). `APIConnectionError` also covers the openai
# client's `APITimeoutError` (a subclass). HTTP errors are handled by status
# code below.
_TRANSPORT_ERRORS: tuple[type[BaseException], ...] = (APIConnectionError,)
try:  # requests is always present via huggingface_hub; guard defensively
    from requests.exceptions import ConnectionError as _ReqConnectionError
    from requests.exceptions import Timeout as _ReqTimeout

    _TRANSPORT_ERRORS += (_ReqConnectionError, _ReqTimeout)
except Exception:  # pragma: no cover
    pass


def _is_retryable(exc: Exception) -> bool:
    """Retry only *transient* judge failures.

    Retries connection/timeout errors and HTTP 429 (rate limit) or 5xx
    (server) responses. Fatal errors — a bad/expired token, a typo'd model
    id, or any other 4xx — are NOT retried, so they surface immediately
    instead of after several pointless backoffs.
    """
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(getattr(exc, "response", None), "status_code", None)
    if status is not None:
        return status == 429 or status >= 500
    return isinstance(exc, _TRANSPORT_ERRORS)


class JudgeBackend(abc.ABC):
    """Base class for judge backends."""

    name: str
    concurrency: int = 1

    @abc.abstractmethod
    def _call_single(self, comp: Comparison) -> dict[str, str]:
        """Run the judge on a single comparison."""

    def judge(self, comparisons: list[Comparison]) -> list[dict[str, str]]:
        """Run the judge on a list of comparisons (concurrently if supported).

        Returns a list of parsed results (one per comparison).
        Each result is a dict with ``winner`` and ``reason`` keys,
        or an empty dict on failure.
        """
        if self.concurrency <= 1 or len(comparisons) <= 1:
            return [self._call_single(comp) for comp in comparisons]

        # Concurrent execution preserving order
        results: list[dict[str, str]] = [{}] * len(comparisons)
        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            future_to_idx = {
                pool.submit(self._call_single, comp): i
                for i, comp in enumerate(comparisons)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:
                    logger.warning("judge_call_failed", idx=idx, error=str(exc))
                    results[idx] = {}
        return results


DEFAULT_MAX_TOKENS = 1024


class InferenceProviderJudge(JudgeBackend):
    """HF Inference Providers backend (Novita, Together, etc.)."""

    def __init__(
        self, model: str, provider: str | None = None, max_tokens: int = DEFAULT_MAX_TOKENS,
    ):
        self.name = f"{provider + ':' if provider else ''}{model}"
        self.model = model
        self.max_tokens = max_tokens
        self.client = InferenceClient(model=model, provider=provider)  # type: ignore[invalid-argument-type]

    @stamina.retry(on=_is_retryable, attempts=_RETRY_ATTEMPTS)
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
        temperature: float = 0.0,
        concurrency: int = 1,
    ):
        self.name = model if model != "default" else f"openai@{base_url}"
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.extra_body = extra_body if extra_body is not None else {"guided_json": JUDGE_SCHEMA}
        self.concurrency = concurrency
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    @stamina.retry(on=_is_retryable, attempts=_RETRY_ATTEMPTS)
    def _call_single(self, comp: Comparison) -> dict[str, str]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=comp.messages,  # type: ignore[invalid-argument-type]
            max_tokens=self.max_tokens,
            temperature=self.temperature,
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

DEFAULT_JUDGE = "novita:Qwen/Qwen3.5-35B-A3B"


def parse_judge_spec(
    spec: str, max_tokens: int = DEFAULT_MAX_TOKENS, concurrency: int = 1,
) -> JudgeBackend:
    """Parse a judge specification string into a backend.

    Formats:
      - ``"https://xxx.endpoints.huggingface.cloud"`` → :class:`OpenAICompatibleJudge`
        (HF Inference Endpoints, OpenAI-compatible with HF token auth)
      - ``"http://..."`` or ``"https://..."`` (other) → :class:`OpenAICompatibleJudge`
      - ``"provider:org/model"`` (colon before first ``/``) → :class:`InferenceProviderJudge`
      - anything else → :class:`InferenceProviderJudge` (no provider)
    """
    if spec.startswith("http://") or spec.startswith("https://"):
        # Check for url:model format (e.g. https://...cloud/v1/:org/model)
        url_part = spec
        model_name = "default"
        # Split on /v1/: to separate URL from model name
        if "/v1/:" in spec:
            url_part, model_name = spec.split("/v1/:", 1)
            url_part += "/v1"

        # HF Inference Endpoints — OpenAI-compatible, auth via HF token
        if ".endpoints.huggingface." in url_part:
            from huggingface_hub import get_token

            base_url = url_part.rstrip("/")
            if not base_url.endswith("/v1"):
                base_url += "/v1"
            token = get_token() or "not-needed"
            return OpenAICompatibleJudge(
                base_url=base_url,
                model=model_name,
                api_key=token,
                max_tokens=max_tokens,
                temperature=0.7,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                concurrency=concurrency,
            )
        return OpenAICompatibleJudge(
            base_url=url_part, model=model_name, max_tokens=max_tokens,
            concurrency=concurrency,
        )

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
