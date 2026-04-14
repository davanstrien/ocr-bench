"""Standard reference-based evaluation (dummy scaffold)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StandardEvalResult:
    """Container for standard evaluation metrics for one model output column."""

    model: str
    samples: int
    exact_match: float
    normalized_overlap: float


def evaluate_against_ground_truth(
    dataset: object,
    model_columns: dict[str, str],
    *,
    ground_truth_column: str = "sudoc_record_templated",
) -> list[StandardEvalResult]:
    """Dummy standard eval comparing each model output to the ground truth column.

    This is intentionally a placeholder scaffold; metric implementations can be
    replaced with stricter JSON-aware field-level metrics later.
    """
    if not hasattr(dataset, "column_names"):
        return []
    if ground_truth_column not in dataset.column_names:
        return []

    gt_values = [str(v or "").strip() for v in dataset[ground_truth_column]]
    results: list[StandardEvalResult] = []

    for col, model_name in model_columns.items():
        if col not in dataset.column_names:
            continue
        preds = [str(v or "").strip() for v in dataset[col]]
        compared = min(len(gt_values), len(preds))
        if compared == 0:
            continue

        exact = sum(1 for i in range(compared) if preds[i] == gt_values[i]) / compared
        overlap = sum(
            1 for i in range(compared) if preds[i] and gt_values[i] and preds[i] in gt_values[i]
        ) / compared
        results.append(
            StandardEvalResult(
                model=model_name,
                samples=compared,
                exact_match=exact,
                normalized_overlap=overlap,
            )
        )

    return results
