"""Standard reference-based evaluation for thesis metadata extraction."""

from __future__ import annotations

import json
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

try:
    from rapidfuzz import fuzz
except ImportError:  # pragma: no cover - fallback for minimal environments
    fuzz = None  # type: ignore[assignment]

from ocr_bench.task_config import DEFAULT_GROUND_TRUTH_COLUMN

EXACT_MATCH_FIELDS = {"defense_year", "degree_type", "language"}

LIST_FIELDS = {
    "title",
    "subtitle",
    "author",
    "degree_type",
    "discipline",
    "granting_institution",
    "doctoral_school",
    "thesis_advisor",
    "jury_president",
    "reviewers",
    "committee_members",
}

FUZZY_THRESHOLD = 90
_JURY_FIELDS = ("jury_president", "reviewers", "committee_members")


@dataclass
class StandardEvalResult:
    """Aggregate metrics for one model output column."""

    model: str
    samples: int
    global_f1: float
    jury_global_f1: float
    metrics: dict[str, dict[str, float]]


def normalize_text(text: Any) -> str:
    """Normalize a field value for string matching."""
    if text is None:
        return ""
    return str(text).strip().lower()


def parse_ground_truth(gt_obj: Any) -> dict[str, Any]:
    """Parse GT row payload (JSON string or dict)."""
    if isinstance(gt_obj, dict):
        return gt_obj
    if gt_obj is None:
        return {}
    try:
        parsed = json.loads(str(gt_obj))
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def parse_prediction(pred_obj: Any) -> dict[str, Any]:
    """Parse model output payload (JSON string or dict)."""
    if isinstance(pred_obj, dict):
        return pred_obj
    if pred_obj is None:
        return {}
    try:
        parsed = json.loads(str(pred_obj))
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def split_list_field(value: Any) -> list[str]:
    """Split a list-like field represented as list or pipe-separated string."""
    if not value:
        return []
    if isinstance(value, list):
        return [normalize_text(v) for v in value if normalize_text(v)]
    return [normalize_text(v) for v in str(value).split("|") if normalize_text(v)]


def match_exact(pred: Any, gt: Any) -> bool:
    return normalize_text(pred) == normalize_text(gt)


def match_fuzzy(pred: Any, gt: Any) -> bool:
    pred_norm = normalize_text(pred)
    gt_norm = normalize_text(gt)
    if not pred_norm and not gt_norm:
        return True
    if fuzz is not None:
        score = fuzz.token_set_ratio(pred_norm, gt_norm)
    else:
        score = SequenceMatcher(a=pred_norm, b=gt_norm).ratio() * 100
    return score >= FUZZY_THRESHOLD


def match_list(pred: Any, gt: Any) -> tuple[int, int, int]:
    """Return list matching counts: tp, fp, fn."""
    pred_list = split_list_field(pred)
    gt_list = split_list_field(gt)

    matched = 0
    used_pred: set[int] = set()
    for gt_item in gt_list:
        for i, pred_item in enumerate(pred_list):
            if i in used_pred:
                continue
            if match_fuzzy(pred_item, gt_item):
                matched += 1
                used_pred.add(i)
                break
    tp = matched
    fp = len(pred_list) - matched
    fn = len(gt_list) - matched
    return tp, fp, fn


def build_jury_set(record: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for field in _JURY_FIELDS:
        names.extend(split_list_field(record.get(field)))
    return names


def evaluate_jury_global(pred: dict[str, Any], gt: dict[str, Any]) -> dict[str, int]:
    tp, fp, fn = match_list(build_jury_set(pred), build_jury_set(gt))
    return {"tp": tp, "fp": fp, "fn": fn}


def _field_scores(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def evaluate_record(
    pred: dict[str, Any], gt: dict[str, Any], observable_fields: set[str] | None = None
) -> dict[str, dict[str, int]]:
    """Compute per-field tp/fp/fn for one prediction vs one GT record."""
    results: dict[str, dict[str, int]] = {}

    for field, gt_val in gt.items():
        if field == "confidence":
            continue
        if observable_fields and field not in observable_fields:
            continue

        pred_val = pred.get(field)

        if field in LIST_FIELDS:
            tp, fp, fn = match_list(pred_val, gt_val)
        else:
            if not gt_val:
                tp, fp, fn = (0, 1, 0) if pred_val else (0, 0, 0)
            else:
                matched = (
                    match_exact(pred_val, gt_val)
                    if field in EXACT_MATCH_FIELDS
                    else match_fuzzy(pred_val, gt_val)
                )
                tp, fp, fn = (1, 0, 0) if matched else (0, 1, 1)

        results[field] = {"tp": tp, "fp": fp, "fn": fn}

    results["jury_global"] = evaluate_jury_global(pred, gt)
    return results


def aggregate_metrics(all_results: list[dict[str, dict[str, int]]]) -> dict[str, dict[str, float]]:
    """Aggregate tp/fp/fn across rows and compute precision/recall/f1 per field."""
    aggregate: dict[str, dict[str, int]] = {}
    for record in all_results:
        for field, scores in record.items():
            if field not in aggregate:
                aggregate[field] = {"tp": 0, "fp": 0, "fn": 0}
            aggregate[field]["tp"] += scores["tp"]
            aggregate[field]["fp"] += scores["fp"]
            aggregate[field]["fn"] += scores["fn"]

    return {
        field: _field_scores(scores["tp"], scores["fp"], scores["fn"])
        for field, scores in aggregate.items()
    }


def compute_global_score(metrics: dict[str, dict[str, float]]) -> float:
    """Mean F1 across fields (including jury_global when present)."""
    if not metrics:
        return 0.0
    f1s = [entry["f1"] for entry in metrics.values()]
    return sum(f1s) / len(f1s) if f1s else 0.0


def _parse_observable_fields(raw: Any) -> set[str] | None:
    if raw is None:
        return None
    if isinstance(raw, list):
        return {str(v) for v in raw}
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return {str(v) for v in parsed}
        except json.JSONDecodeError:
            pass
        return {v.strip() for v in raw.split("|") if v.strip()}
    return None


def _dataset_len(dataset: Any) -> int:
    try:
        return len(dataset)
    except Exception:
        return 0


def _get_column(dataset: Any, column: str) -> list[Any]:
    try:
        return list(dataset[column])
    except Exception:
        return []


def evaluate_against_ground_truth(
    dataset: Any,
    model_columns: dict[str, str],
    *,
    ground_truth_column: str = DEFAULT_GROUND_TRUTH_COLUMN,
) -> list[StandardEvalResult]:
    """Evaluate all prediction columns against the grounded truth JSON column."""
    if not hasattr(dataset, "column_names"):
        return []
    if ground_truth_column not in dataset.column_names:
        return []

    gt_rows = _get_column(dataset, ground_truth_column)
    n_rows = _dataset_len(dataset)
    if not gt_rows or n_rows == 0:
        return []

    observable_rows = (
        _get_column(dataset, "observable_fields")
        if "observable_fields" in dataset.column_names
        else [None] * n_rows
    )

    outputs: list[StandardEvalResult] = []
    for column_name, model_name in model_columns.items():
        if column_name not in dataset.column_names:
            continue
        pred_rows = _get_column(dataset, column_name)
        compared = min(n_rows, len(pred_rows), len(gt_rows), len(observable_rows))
        if compared == 0:
            continue

        all_results: list[dict[str, dict[str, int]]] = []
        for i in range(compared):
            gt = parse_ground_truth(gt_rows[i])
            pred = parse_prediction(pred_rows[i])
            observable_fields = _parse_observable_fields(observable_rows[i])
            all_results.append(evaluate_record(pred, gt, observable_fields))

        metrics = aggregate_metrics(all_results)
        jury_f1 = metrics.get("jury_global", {}).get("f1", 0.0)
        outputs.append(
            StandardEvalResult(
                model=model_name,
                samples=compared,
                global_f1=compute_global_score(metrics),
                jury_global_f1=jury_f1,
                metrics=metrics,
            )
        )

    return outputs
