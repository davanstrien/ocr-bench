"""Tests for standard reference-based evaluation metrics."""

from __future__ import annotations

import json

from ocr_bench.standard_eval import evaluate_against_ground_truth


class _FakeDataset:
    def __init__(self, rows: list[dict]):
        self._rows = rows
        self.column_names = list(rows[0].keys()) if rows else []

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, item):
        if isinstance(item, str):
            return [row.get(item) for row in self._rows]
        return self._rows[item]


def test_evaluate_against_ground_truth_single_row_perfect_match():
    gt = {
        "title": "Titre",
        "subtitle": "",
        "author": "Alice",
        "degree_type": "Thèse de doctorat",
        "discipline": "Droit",
        "granting_institution": "Université X",
        "doctoral_school": "",
        "defense_year": "2014",
        "thesis_advisor": "Prof Y",
        "jury_president": "Jean",
        "reviewers": "Paul|Marie",
        "committee_members": "",
        "language": "fre",
    }
    pred = dict(gt)
    ds = _FakeDataset(
        [
            {
                "sudoc_record_templated": json.dumps(gt, ensure_ascii=False),
                "model_output": json.dumps(pred, ensure_ascii=False),
            }
        ]
    )

    results = evaluate_against_ground_truth(ds, {"model_output": "ModelA"})
    assert len(results) == 1
    assert results[0].samples == 1
    assert 0.0 < results[0].global_f1 < 1.0
    assert results[0].jury_global_f1 == 1.0


def test_evaluate_against_ground_truth_missing_gt_column_returns_empty():
    ds = _FakeDataset([{"model_output": "{}"}])
    results = evaluate_against_ground_truth(ds, {"model_output": "ModelA"})
    assert results == []
