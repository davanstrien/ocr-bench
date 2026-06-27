# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#     "olmocr[bench]",
#     "numpy",  # olmocr.bench.tests imports numpy but doesn't declare it
# ]
# ///
"""
Job 2 (CPU): score the candidate produced by convert.py with the official
olmocr.bench.benchmark harness. old_scans = text-present / text-absent /
reading-order tests only -> pure string matching, no KaTeX/chromium needed.

Reads from DATA (default /bucket). Mount the same bucket the convert job wrote to,
read-only is fine:

    hf jobs uv run --flavor cpu-upgrade -s HF_TOKEN \\
        -v hf://buckets/davanstrien/paddleocr-vl16-oldscans:/bucket:ro \\
        experiments/olmocr-bench-oldscans/score.py

Env:
  DATA   directory holding old_scans.jsonl + the candidate folder (default /bucket)
"""
import os
import subprocess
import sys

DATA = os.environ.get("DATA", "/bucket")

# --dir globs *.jsonl (only old_scans.jsonl is present) and treats each subdir
# other than "pdfs" as a candidate (only paddleocr_vl_16 is present).
proc = subprocess.run(
    [sys.executable, "-m", "olmocr.bench.benchmark", "--dir", DATA],
    text=True,
)
sys.exit(proc.returncode)
