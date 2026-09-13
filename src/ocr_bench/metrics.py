"""Ground-truth OCR metrics.

Character and word error rates complement the pairwise VLM judge when a
collection includes reference transcriptions. The implementation records the
edit-distance numerators and denominators so published scores remain auditable.
"""

from __future__ import annotations

import html
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeVar

from datasets import Dataset
from rapidfuzz.distance import Levenshtein

from ocr_bench.judge import is_sentinel, normalize_for_judge

_WHITESPACE_RE = re.compile(r"\s+")
_T = TypeVar("_T")


@dataclass(frozen=True)
class MetricSummary:
    """Corpus-level error totals for one OCR model."""

    model: str
    column: str
    char_errors: int
    reference_chars: int
    word_errors: int
    reference_words: int
    evaluated_samples: int
    skipped_samples: int
    failed_outputs: int

    @property
    def cer(self) -> float:
        return self.char_errors / self.reference_chars if self.reference_chars else 0.0

    @property
    def wer(self) -> float:
        return self.word_errors / self.reference_words if self.reference_words else 0.0

    def as_row(self) -> dict[str, object]:
        """Return the stable aggregate schema used by CLI and Hub publishing."""
        return {
            "model": self.model,
            "column": self.column,
            "cer": self.cer,
            "wer": self.wer,
            "char_errors": self.char_errors,
            "reference_chars": self.reference_chars,
            "word_errors": self.word_errors,
            "reference_words": self.reference_words,
            "evaluated_samples": self.evaluated_samples,
            "skipped_samples": self.skipped_samples,
            "failed_outputs": self.failed_outputs,
        }


@dataclass(frozen=True)
class MetricResult:
    """Aggregate and per-sample scores for a collection."""

    summaries: list[MetricSummary]
    details: list[dict[str, object]]
    reference_column: str


def normalize_metric_text(value: object) -> str:
    """Normalize Unicode, HTML, and whitespace while preserving case/punctuation."""
    if value is None:
        text = ""
    elif isinstance(value, str):
        text = value
    elif isinstance(value, (bytes, bytearray)):
        text = value.decode("utf-8", "replace")
    else:
        text = str(value)

    text = unicodedata.normalize("NFC", text)
    return _WHITESPACE_RE.sub(" ", html.unescape(normalize_for_judge(text))).strip()


def levenshtein_distance(reference: Sequence[_T], prediction: Sequence[_T]) -> int:
    """Return unit-cost Levenshtein distance for text or token sequences."""
    return Levenshtein.distance(reference, prediction)


def score_dataset(
    dataset: Dataset,
    ocr_columns: dict[str, str],
    reference_column: str,
) -> MetricResult:
    """Score every OCR column against a reference transcription column.

    Empty reference rows are skipped because CER/WER have no meaningful
    denominator. Error sentinels are evaluated as empty predictions, ensuring
    failed inference is penalized rather than making a model look better by
    silently dropping its hardest pages.
    """
    if reference_column not in dataset.column_names:
        raise ValueError(
            f"Reference column '{reference_column}' not found. "
            f"Available columns: {dataset.column_names}"
        )
    # Flat-dataset discovery intentionally treats names such as ``text`` and
    # ``ocr_*`` as likely model outputs. Ground-truth columns commonly use those
    # names too, so exclude the declared reference even if discovery selected it.
    scored_columns = {
        column: model for column, model in ocr_columns.items() if column != reference_column
    }
    if not scored_columns:
        raise ValueError(
            f"No OCR output columns remain after excluding reference column "
            f"'{reference_column}'"
        )
    missing = [column for column in scored_columns if column not in dataset.column_names]
    if missing:
        raise ValueError(f"OCR columns not found: {', '.join(missing)}")

    references = dataset[reference_column]
    identifiers = _identifier_values(dataset)
    summaries: list[MetricSummary] = []
    details: list[dict[str, object]] = []

    for column, model in scored_columns.items():
        predictions = dataset[column]
        char_errors = reference_chars = 0
        word_errors = reference_words = 0
        evaluated = skipped = failed = 0

        for sample_idx in range(len(dataset)):
            reference_value = references[sample_idx]
            prediction_value = predictions[sample_idx]
            reference = normalize_metric_text(reference_value)
            if not reference.strip():
                skipped += 1
                continue

            prediction_raw = "" if prediction_value is None else str(prediction_value)
            prediction_failed = is_sentinel(prediction_raw)
            prediction = "" if prediction_failed else normalize_metric_text(prediction_value)
            ref_words = reference.split()
            pred_words = prediction.split()
            sample_char_errors = levenshtein_distance(reference, prediction)
            sample_word_errors = levenshtein_distance(ref_words, pred_words)

            evaluated += 1
            failed += int(prediction_failed)
            char_errors += sample_char_errors
            reference_chars += len(reference)
            word_errors += sample_word_errors
            reference_words += len(ref_words)
            details.append(
                {
                    "sample_idx": sample_idx,
                    "id": identifiers[sample_idx],
                    "model": model,
                    "column": column,
                    "cer": sample_char_errors / len(reference),
                    "wer": sample_word_errors / len(ref_words),
                    "char_errors": sample_char_errors,
                    "reference_chars": len(reference),
                    "word_errors": sample_word_errors,
                    "reference_words": len(ref_words),
                    "failed_output": prediction_failed,
                }
            )

        summaries.append(
            MetricSummary(
                model=model,
                column=column,
                char_errors=char_errors,
                reference_chars=reference_chars,
                word_errors=word_errors,
                reference_words=reference_words,
                evaluated_samples=evaluated,
                skipped_samples=skipped,
                failed_outputs=failed,
            )
        )

    summaries.sort(key=lambda summary: (summary.cer, summary.wer, summary.model))
    return MetricResult(
        summaries=summaries,
        details=details,
        reference_column=reference_column,
    )


def _identifier_values(dataset: Dataset) -> list[str]:
    """Return stable display identifiers without decoding image rows."""
    for column in ("id", "source_row", "b_number", "page_index"):
        if column in dataset.column_names:
            return [str(value) if value is not None else "" for value in dataset[column]]
    return [str(index) for index in range(len(dataset))]
