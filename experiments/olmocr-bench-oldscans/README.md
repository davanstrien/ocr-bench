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

# convert — 98 PDFs (add -e LIMIT=3 for a 3-PDF smoke test). On the first run this
# also stages the source PDFs into the bucket under pdfs/ (which the scorer needs);
# later runs reuse them from the mount instead of re-downloading.
hf jobs run --flavor l4x1 --timeout 1h -s HF_TOKEN -v $BUCKET:/bucket:ro \
    $IMAGE python3 /bucket/convert.py

# score (ranks every candidate folder in the bucket together)
hf jobs uv run --flavor cpu-upgrade -s HF_TOKEN -v $BUCKET:/bucket:ro score.py
```

Add `-d` to detach, then `hf jobs wait <id>` / `hf jobs logs <id>`.

## Configuration

- **Flavor `l4x1`**: the image's CUDA build matches the `l4x1` driver; larger GPUs (l40s / a100) do not.
- **Mount path `/bucket`**: `/data` is reserved by Jobs for local-script artifacts.
- **`sync_bucket`, not a FUSE write**: the image runs as a non-root user that cannot write the mount, so `convert.py` writes locally and uploads over HTTP; the mount is `:ro` (script delivery only).
- **`pdfs/` folder**: `benchmark.py` requires `<dir>/pdfs` to exist; `convert.py` stages the source PDFs into the bucket on the first run and reuses them from the mount after, so no separate sync step is needed.
- **`numpy`**: declared in `score.py` because `olmocr[bench]` imports it without declaring it.
- **Versions**: `convert.py` takes `PIPELINE_VERSION` (default `v1.6`) and `CANDIDATE`. Run `-e PIPELINE_VERSION=v1 -e CANDIDATE=paddleocr_vl_orig` to also convert the original 0.9B PaddleOCR-VL (the leaderboard's 37.8); both candidates then sit in the bucket and the score job ranks them together.
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

PaddleOCR-VL on olmOCR-bench `old_scans` — default pipeline, no tuning, greedy
(2026-06-27). Both versions scored through the same harness:

| Version | **old_scans** | present | absent | order | baseline |
|---|---|---|---|---|---|
| v1.6 | **38.6%** (203/526) | 31.2 | 95.7 | 27.7 | 84.7 |
| v1 (original 0.9B) | **38.2%** (201/526) | 32.3 | 95.7 | 24.9 | 88.8 |

`old_scans` = the present/absent/order tests = the leaderboard's "Old scans" column.

**Harness validated against the published figure.** olmOCR-bench lists the original
`PaddleOCR-VL` (unversioned; its `run_paddlevl` runner is dated 2025-10-20, pre-1.6)
at **Old scans = 37.8**. Running that same original (`v1`) through this harness gives
**38.2** — within 0.4 pt, inside the ±3.6 % CI. So our convert + scoring reproduce
the published number; the harness is sound.

**v1.6's gains don't transfer to old scans.** v1.6 (38.6) and the original v1 (38.2)
are statistically indistinguishable here — the upgrade that made v1.6 SOTA on
OmniDocBench buys nothing on degraded historical scans. v1.6 even regresses slightly
on `baseline` (84.7 vs 88.8): it emits *more* CJK/Japanese disallowed-character
hallucinations (场, 景, 民, 生, ら …) on English scans than the original did. See
`samples.html` (regenerate via `gen_samples.py`) for scan↔output pairs with the
glyphs highlighted.

**Reading the harness output.** "38.6" / "38.2" are the `old_scans.jsonl` sub-scores.
`olmocr.bench.benchmark` *also* prints an `Average Score` (61.6 % / 63.5 %) = mean of
the old_scans sub-score and the auto-baseline category — that is **not** the
leaderboard "Old scans" figure; don't quote it. And the per-version `baseline` here
(auto-BaselineTest over only the 98 old_scans PDFs) is **not** the leaderboard's
"Base" column (the same test over the whole ~1,400-PDF benchmark, 98.5 %).

**Size context** (published no-anchor Old scans): olmOCR 43.7, GPT-4o 40.9,
Qwen2.5-VL 38.6, Gemini-Flash-2 27.8, GOT-OCR (0.58B) 22.1. At 0.9B, PaddleOCR-VL
ties the 7B Qwen2.5-VL.

> **Status: validated.** The harness reproduces the published original-PaddleOCR-VL
> figure (37.8 → 38.2, within CI), and v1.6 (38.6) is statistically the same on
> old_scans. Greedy/deterministic decoding; outputs spot-checked vs source scans.
> Pin the image digest (see Reproducibility) before citing externally.
