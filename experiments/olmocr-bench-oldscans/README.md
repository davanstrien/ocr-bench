# PaddleOCR-VL-1.6 on olmOCR-bench (old_scans)

A standalone experiment — **not part of the `ocr_bench` library**. Scores
[PaddleOCR-VL-1.6](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6) on the
`old_scans` subset of [allenai/olmOCR-bench](https://huggingface.co/datasets/allenai/olmOCR-bench),
a number the PaddleOCR-VL-1.6 technical report (arXiv 2606.03264) never reports —
it only measures OmniDocBench v1.6 / Real5-OmniDocBench.

`old_scans` = 98 single-page Library-of-Congress scans, 526 tests
(text-present / text-absent / reading-order). No math, no tables → pure string
matching, so the scorer needs **no KaTeX/chromium**.

## Fidelity (why the number is fair)

- **Scoring** is stock `olmocr.bench.benchmark`, untouched.
- **Conversion** mirrors olmOCR-bench's own runner
  [`olmocr/bench/runners/run_paddlevl.py`](https://github.com/allenai/olmocr/blob/main/olmocr/bench/runners/run_paddlevl.py)
  exactly: `res.markdown["markdown_texts"]`, per page, **bare default pipeline,
  no tuning** (no `max_pixels` / prompts / dpi). The only intentional difference
  is `pipeline_version="v1.6"` — what the model card tells you to pass.
- We run inside **PaddlePaddle's own docker image**, so paddle/paddleocr are the
  vendor's exact builds, not a PyPI guess. Arguably *more* faithful than
  assembling the stack ourselves.

## Design

Two HF Jobs, one bucket as the handoff. The two stacks never share an env:

| Job | Command | Hardware | Stack | Does |
|-----|---------|----------|-------|------|
| `convert.py` | `hf jobs run` | GPU (`l4x1`) | PaddlePaddle image | PaddleOCR-VL-1.6 → markdown into the bucket |
| `score.py` | `hf jobs uv run` | CPU (`cpu-upgrade`) | olmocr (PyPI) | `olmocr.bench.benchmark` → prints the score |

The candidate path is written as `{splitext(pdf_field)}_pg{page}_repeat1.md` —
the literal string transform `benchmark.py` uses to locate it — so the layout is
guaranteed to match without going through olmocr's convert machinery.

### Image (probed, not guessed)

`ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-vl:latest-nvidia-gpu`
(~8 GB; **the Baidu registry is pullable by HF Jobs**). Probed on `cpu-basic`:

```
PY=/usr/local/bin/python3        # uv: none -> use `hf jobs run`, not `uv run`
paddleocr 3.6.0                  # paddlex + huggingface_hub 1.16.4 also present
site-packages: /usr/local/lib/python3.10/site-packages
```

Because the image lacks `uv`, `convert.py` is a **plain python script** (no PEP 723
header) run by the image's python; every import it needs is already in the image.
`hf jobs run` has no local-file upload, so the script is delivered via the bucket
(mounted read-only). The image runs as the **non-root `paddleocr` user**, which
can't write the root-owned bucket FUSE mount — so `convert.py` writes to a local
dir and pushes results with `sync_bucket()` (mount-free HTTP upload).

Handoff bucket: `hf://buckets/davanstrien/paddleocr-vl16-oldscans`

## Run

```bash
BUCKET=hf://buckets/davanstrien/paddleocr-vl16-oldscans
IMAGE=ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-vl:latest-nvidia-gpu

# deliver the script into the bucket (re-cp whenever you edit it)
hf buckets cp convert.py $BUCKET/convert.py

# 1a. smoke-test the plumbing on 3 PDFs first (mount ro: script delivery only;
#     results go back to the bucket via sync_bucket over HTTP)
hf jobs run --flavor l4x1 -s HF_TOKEN -e LIMIT=3 -v $BUCKET:/bucket:ro \
    $IMAGE python3 /bucket/convert.py

# 1b. full convert — 98 PDFs. USE l4x1 (see gotchas) and a generous --timeout.
hf jobs run --flavor l4x1 --timeout 1h -s HF_TOKEN -v $BUCKET:/bucket:ro \
    $IMAGE python3 /bucket/convert.py

# 2. score (CPU; olmocr from PyPI; read-only mount)
hf jobs uv run --flavor cpu-upgrade -s HF_TOKEN -v $BUCKET:/bucket:ro \
    score.py
```

`hf jobs run`/`uv run` accept `-d` to detach; then block on the job with
`hf jobs wait <job_id>` and read `hf jobs logs <job_id>`. Inspect the bucket
between steps: `hf buckets ls $BUCKET`.

## Notes / gotchas (all hit during bring-up)

- **`/data` is reserved** by Jobs for local-script artifacts → mount the bucket
  at `/bucket`.
- **Transient `Volume mount failed`** on a fresh bucket (CSI driver not ready on
  a fresh node, unrelated to the bucket being empty) → just re-run the job.
- **Non-root image can't write the bucket mount**: this image runs as
  `paddleocr`, the FUSE mount is root-owned → `PermissionError` on write. Fix:
  write locally, upload via `sync_bucket()` (HTTP, no FUSE). Mount the bucket
  `:ro` in job 1 since it's only used to deliver the script.
- **Why the image, not a uv script**: paddlex pulls the GUI build of opencv
  (`libGL.so.1`, absent in the slim uv image) *and* the VL pipeline needs the
  `paddlex[ocr]` extra; the 1.8 GB paddle wheel rebuilds every run. The vendor
  image sidesteps all of it. (`uv run --image` does *not* help: uv still
  reinstalls declared deps, and reusing the image's packages needs
  `--system-site-packages`, which uv lacks. The `--python`+`PYTHONPATH` trick
  needs uv *in* the image, which this one doesn't have.)
- **Use `l4x1` for convert, not bigger GPUs**: this paddle image's CUDA build
  matches the older `l4x1` driver. On `l40sx1` the model hung on the first PDF
  (driver/CUDA mismatch). More compute doesn't help anyway — a 0.9B model over 98
  pages is bound by image-pull + model-load, not GPU throughput.
- **Convert `--timeout`**: the full run can outlast the default job timeout. It
  still finishes and `sync_bucket`s before being killed (so the bucket is
  complete), but the job shows `ERROR: Job timeout` — pass `--timeout 1h` to keep
  the status clean.
- **`numpy` for scoring**: `olmocr[bench]` imports numpy without declaring it
  (assumes their conda env) → `score.py` adds `numpy` explicitly.
- **old_scans only.** For `old_scans_math` (458 math tests): change `JSONL_PATH`
  in `convert.py`, and `score.py` then needs `playwright install chromium` for
  KaTeX.

## Result

PaddleOCR-VL-1.6, default v1.6 pipeline, no tuning (run 2026-06-27):

| Category | Pass rate | Tests |
|---|---|---|
| **old_scans (present/absent/order)** | **38.6%** | 203/526 |
| → present | 31.2% | 279 |
| → absent | 95.7% | 70 |
| → order | 27.7% | 177 |
| baseline (auto-generated, 1/PDF) | 84.7% | 83/98 |
| tool "Average Score" (mean of the two jsonl files) | 61.6% ± 4.0% | 624 |

**Headline = 38.6%** — the leaderboard-comparable `old_scans` number. (The tool's
"61.6%" averages in the easy auto-baseline tests; don't quote it as the score.)
For reference, olmOCR-bench's published OldScan column: olmOCR-Ours 44.5, GPT-4o
40.7, Qwen2.5-VL 38.6, Gemini-Flash-2 34.2, most others 17–29. So PaddleOCR-VL-1.6
is **mid-pack on degraded historical scans** despite being SOTA on OmniDocBench.

**Notable:** ~15 baseline failures are `Text contains disallowed characters`
(CJK: 场, 景, 民, 生, …) — the model **hallucinates Chinese characters on English
handwritten scans**. Clean-benchmark SOTA ≠ real-world historical data.
