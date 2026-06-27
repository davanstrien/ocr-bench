# PaddleOCR-VL-1.6 on olmOCR-bench (old_scans)

Scores [PaddleOCR-VL-1.6](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6) on the
`old_scans` subset of [`allenai/olmOCR-bench`](https://huggingface.co/datasets/allenai/olmOCR-bench).
Standalone experiment — not part of the `ocr_bench` library.

`old_scans` = 98 single-page Library-of-Congress scans, 526 tests
(text-present / text-absent / reading-order). No math or tables, so scoring needs
no KaTeX/chromium.

## Fidelity

- **Scoring**: stock `olmocr.bench.benchmark`, unmodified.
- **Conversion**: matches olmOCR-bench's own runner
  [`run_paddlevl.py`](https://github.com/allenai/olmocr/blob/main/olmocr/bench/runners/run_paddlevl.py)
  — `res.markdown["markdown_texts"]`, per page, default pipeline, no tuning. The
  only difference is `pipeline_version="v1.6"` (as the model card specifies).
- Runs inside PaddlePaddle's own image, so paddle/paddleocr are the vendor builds.

## Method

Two HF Jobs with a bucket as the handoff:

| Step | Command | Hardware | Does |
|------|---------|----------|------|
| `convert.py` | `hf jobs run` | GPU `l4x1` | PaddleOCR-VL-1.6 → markdown → `sync_bucket` to the bucket |
| `score.py` | `hf jobs uv run` | CPU `cpu-upgrade` | `olmocr.bench.benchmark` → score |

Candidate files are written as `{splitext(pdf_field)}_pg{page}_repeat1.md`, the
path `benchmark.py` looks them up by.

- **Image**: `ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-vl:latest-nvidia-gpu`
  — paddle 3.2.1 + paddleocr 3.6.0 + 1.6 weights baked in; python `/usr/local/bin/python3` (3.10); no `uv`.
- **Bucket**: `hf://buckets/davanstrien/paddleocr-vl16-oldscans`

## Run

```bash
BUCKET=hf://buckets/davanstrien/paddleocr-vl16-oldscans
IMAGE=ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-vl:latest-nvidia-gpu

# deliver the script into the bucket (re-cp after editing)
hf buckets cp convert.py $BUCKET/convert.py

# convert — 98 PDFs (add -e LIMIT=3 for a 3-PDF smoke test)
hf jobs run --flavor l4x1 --timeout 1h -s HF_TOKEN -v $BUCKET:/bucket:ro \
    $IMAGE python3 /bucket/convert.py

# add the source PDFs the scorer requires under pdfs/
hf download allenai/olmOCR-bench --repo-type dataset \
    --include "bench_data/pdfs/old_scans/*" --local-dir /tmp/olm
hf buckets sync /tmp/olm/bench_data/pdfs $BUCKET/pdfs

# score
hf jobs uv run --flavor cpu-upgrade -s HF_TOKEN -v $BUCKET:/bucket:ro score.py
```

Add `-d` to detach, then `hf jobs wait <id>` / `hf jobs logs <id>`.

## Configuration

- **Flavor `l4x1`**: the image's CUDA build matches the `l4x1` driver; larger GPUs (l40s / a100) do not.
- **Mount path `/bucket`**: `/data` is reserved by Jobs for local-script artifacts.
- **`sync_bucket`, not a FUSE write**: the image runs as a non-root user that cannot write the mount, so `convert.py` writes locally and uploads over HTTP; the mount is `:ro` (script delivery only).
- **`pdfs/` folder**: `benchmark.py` requires `<dir>/pdfs` to exist; the source PDFs are synced there (run step above).
- **`numpy`**: declared in `score.py` because `olmocr[bench]` imports it without declaring it.
- **`old_scans_math`** variant: change `JSONL_PATH` in `convert.py`; `score.py` then also needs `playwright install chromium`.

## Reproducibility

This run uses floating refs. To make it bit-stable, pin:

- the **image by digest** (`...paddleocr-vl@sha256:...`) instead of `:latest` — this pins paddle, paddleocr, and the weights together;
- `allenai/olmOCR-bench` by `revision`;
- `olmocr` to an exact version in `score.py`.

Decoding is already **greedy** (the model's `generation_config.json` has no
`do_sample`/`temperature`, so transformers defaults to greedy), so runs are
deterministic modulo GPU-kernel nondeterminism — no sampling seed to pin.

## Result

PaddleOCR-VL-1.6, default v1.6 pipeline, no tuning (2026-06-27):

| Category | Pass rate | Tests |
|---|---|---|
| **old_scans (present / absent / order)** | **38.6%** | 203 / 526 |
| → present | 31.2% | 279 |
| → absent | 95.7% | 70 |
| → order | 27.7% | 177 |
| baseline (auto-generated, 1/PDF) | 84.7% | 83 / 98 |

For reference, olmOCR-bench's published OldScan column (no-anchor): olmOCR 43.7,
GPT-4o 40.9, Qwen2.5-VL 38.6, Gemini-Flash-2 27.8, GOT-OCR (0.58B) 22.1. At 0.9B,
PaddleOCR-VL-1.6 ties the 7B Qwen2.5-VL.

~15 baseline failures are `disallowed characters`: the model emits CJK glyphs
(场, 景, 民, 生, …) on English handwritten scans.

> **Status: preliminary.** Decoding is greedy (deterministic) and the candidate
> outputs were spot-checked against the source scans (real, untruncated). Not yet
> validated by reproducing a published olmOCR-bench number through this harness —
> do that before quoting the figure externally.
