"""Prepare frozen raw predictions and call the unmodified official olmOCR scorer."""

import argparse
import contextlib
import importlib.metadata
import json
import random
import shutil
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download

from runner import CATEGORIES, DATA_REV, DATASET, digest, key, read_json, write_json

SCORER_REV = "f7cfe4c22098b154c76b6ec950d1c0a464eecf8d"
CANDIDATE = "kraken_ppocrv6_medium"


def main():
    from olmocr.bench import benchmark

    start = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", type=Path, default=Path("/bucket"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    args = parser.parse_args()
    pages = read_json(args.bucket / "manifest.json")
    if args.mode == "smoke":
        pages = [p for p in pages if p["smoke"]]
    selected = {(p["pdf"], p["page"]) for p in pages}
    workspace = Path("/tmp/scorer-input")
    output = args.bucket / "scores" / args.run_id
    workspace.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    test_categories = {}
    for category in CATEGORIES:
        source = Path(hf_hub_download(
            DATASET, f"bench_data/{category}.jsonl", repo_type="dataset", revision=DATA_REV,
        ))
        rows = [json.loads(line) for line in source.read_text().splitlines()]
        rows = [r for r in rows if (r["pdf"], r["page"]) in selected]
        with (workspace / f"{category}.jsonl").open("w") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                test_categories[row["id"]] = f"{category}.jsonl"
    coverage = []
    for page in pages:
        variant = "staged" if args.mode == "smoke" else f"shard-{page['shard']}"
        root = args.bucket / "runs" / args.run_id / variant
        result = read_json(Path(str(root / "pages" / key(page)) + ".result.json"))
        if result["status"] != "ok" or result["page"] != page:
            raise RuntimeError(f"Missing or inconsistent successful output: {key(page)}")
        candidate = root / "candidate" / f"{key(page)}_repeat1.md"
        # Score saved candidate bytes, checking that they are exactly the recorded raw output.
        text = candidate.read_text()
        if text != result["text"]:
            raise RuntimeError(f"Candidate text mismatch: {key(page)}")
        dest = workspace / CANDIDATE / f"{key(page)}_repeat1.md"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text)
        pdf = Path(hf_hub_download(
            DATASET, f"bench_data/pdfs/{page['pdf']}", repo_type="dataset", revision=DATA_REV,
        ))
        if digest(pdf) != result["render"]["pdf_sha256"]:
            raise RuntimeError(f"PDF identity mismatch: {page['pdf']}")
        dest_pdf = workspace / "pdfs" / page["pdf"]
        dest_pdf.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(pdf, dest_pdf)
        coverage.append({"page": page, "characters": len(text), "lines": result["n_lines"]})
    write_json(output / "coverage.json", coverage)
    print(f"Validated {len(coverage)} page predictions and PDF identities", flush=True)

    captured = {}
    original_evaluate = benchmark.evaluate_candidate
    original_ci = benchmark.calculate_bootstrap_ci

    def capture_evaluation(*positional, **kwargs):
        result = original_evaluate(*positional, **kwargs)
        captured["evaluation"] = result
        return result

    def capture_ci(*positional, **kwargs):
        ci = original_ci(*positional, **kwargs)
        captured["ci"] = [float(x) for x in ci]
        return ci

    # Observe the official calls without replacing tests, grouping, scoring or bootstrap code.
    benchmark.evaluate_candidate = capture_evaluation
    benchmark.calculate_bootstrap_ci = capture_ci
    random.seed(20260909)
    np.random.seed(20260909)
    sys.argv = [
        "olmocr.bench.benchmark", "--dir", str(workspace), "--candidate", CANDIDATE,
        "--bootstrap_samples", "1000", "--confidence_level", "0.95",
        "--output_failed", str(output / "failed-tests.jsonl"),
    ]
    # Keep the full official output (including individual failures) as a durable artifact.
    with (output / "official-output.txt").open("w") as stream:
        with contextlib.redirect_stdout(stream):
            benchmark.main()
    micro, total, errors, failures, by_type, _, details = captured["evaluation"]
    write_json(output / "candidate-errors.json", errors)
    if errors:
        raise RuntimeError(f"Official scorer reported {len(errors)} candidate errors")
    categories = defaultdict(lambda: {"passed": 0, "total": 0})
    records = []
    for pdf, pdf_pages in details.items():
        for page_number, tests in pdf_pages.items():
            for test, passed, explanation in tests:
                category = test_categories.get(test.id, "baseline")
                categories[category]["total"] += 1
                categories[category]["passed"] += int(passed)
                records.append({
                    "id": test.id, "pdf": pdf, "page": page_number, "category": category,
                    "type": test.type, "passed": passed, "explanation": explanation,
                    "test": asdict(test),
                })
    if len(records) != total:
        raise RuntimeError(f"Incomplete test-result capture: {len(records)} / {total}")
    for values in categories.values():
        values["score"] = values["passed"] / values["total"]
    score = sum(v["score"] for v in categories.values()) / len(categories)
    summary = {
        "candidate": CANDIDATE, "mode": args.mode, "pages": len(pages), "tests": total,
        "official_macro_score": score, "official_ci_95": captured.get("ci"),
        "micro_score": micro, "categories": dict(sorted(categories.items())),
        "test_types": {k: {"total": len(v), "score": sum(v) / len(v)} for k, v in by_type.items()},
        "failed_tests": len(failures), "candidate_errors": 0,
        "elapsed_seconds": time.perf_counter() - start,
        "scorer_revision": SCORER_REV, "dataset_revision": DATA_REV,
        "python": sys.version, "seed": 20260909,
        "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
        "lock_sha256": digest(Path("uv.lock")), "source_sha256": digest(Path(__file__)),
    }
    with (output / "test-results.jsonl").open("w") as stream:
        for record in sorted(records, key=lambda r: r["id"]):
            stream.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    write_json(output / "summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "packages"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
