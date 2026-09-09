# Kraken on olmOCR-bench

This experiment runs bundled BLLA segmentation and the PP-OCRv6 medium recognizer
through Kraken's Python API on Hugging Face Jobs, then scores the raw outputs with
the pinned official olmOCR-bench implementation. It includes a 21-page smoke
test (three fixed pages per category) and the full 1,403-page benchmark.

## Method

- Pin Kraken to `3e6893a0d5eb06273494bfcdb9d3cac6159e0611`, which includes the GPU
  tensor-placement fix. The bundled segmenter's identity follows this pin.
- Pin the model and dataset revisions in `runner.py`; verify recognition weight
  bytes against SHA-256 before loading.
- Render PDF pages to RGB at 300 DPI using pypdfium2. No PDF text extraction.
- The pinned `old_scans/48.pdf`, `51.pdf` and `57.pdf` pages render to
  253,864,905, 331,857,945 and 323,914,932 pixels at this DPI. Their cached PNGs
  have explicit Pillow size allowances and dimension checks; other pages retain
  Pillow's default limit. No page is downscaled to bypass it.
- Use fp32, batch size 32, in-process line extraction, default reading order,
  bidi reordering and 16-pixel recognition padding. Torch CPU threads are limited
  to four on A10G-small.
- Select smoke pages by a deterministic hash of the page identifier, before
  observing predictions. Assign the full manifest to four category-balanced
  shards. There is no official modern-only subset.
- The stock arm invokes Kraken's high-level segmentation and recognition methods
  for each page. The staged arm configures each model once, runs segmentation
  across the manifest, and then recognizes the saved segmentations. It retains
  the high-level merge, reading-order and decoding paths.
- Compare every coordinate and ordering field plus raw line-joined text. Random
  line/region UUIDs are canonicalized while preserving identity relationships;
  image filenames are excluded from geometry comparison. Persist geometry without
  discarding any fields, and check its JSON round-trip before recognition.
- Exercise resume with prediction methods replaced by failure sentinels: completed
  pages must be loaded without invoking either model again.

Preparation is implemented with an instance-local guard around
`prepare_for_inference`; a different configuration raises an error. This does not
change network weights, image scaling, thresholds, polygonization or decoding.
Both models currently remain loaded during the staged passes.

## Run on Jobs

All runtime checks and tests run on Jobs, including Ruff. The image is pinned by
digest in `launch.sh` and in provenance. The committed `uv.lock` and
`scoring/uv.lock` were resolved and tested on Jobs. Bootstraps prefer these locks
and use `uv sync --locked`. Their fallback, for initial development without a
committed lock, saves a Jobs-generated lock to the Bucket. The compatible pre-1.0
Hub client is used only for downloads; mounted Bucket I/O does not depend on its
newer Bucket APIs.

```sh
experiment=experiments/kraken-olmocr-bench
bucket=YOUR_NAME/YOUR_BUCKET
hf buckets create "$bucket" --private
sh "$experiment/launch.sh" smoke "$bucket" smoke-v1
# Replace these uppercase Job ID placeholders with the IDs printed by launch.sh.
hf jobs wait SMOKE_JOB_ID
sh "$experiment/launch.sh" score-smoke "$bucket" smoke-v1
hf jobs wait SMOKE_SCORER_JOB_ID
sh "$experiment/launch.sh" full "$bucket" full-v1
hf jobs wait SHARD_0_JOB_ID SHARD_1_JOB_ID SHARD_2_JOB_ID SHARD_3_JOB_ID
sh "$experiment/launch.sh" score-full "$bucket" full-v1
hf jobs wait FULL_SCORER_JOB_ID
```

Each phase is asynchronous. Require a `COMPLETED` Job state, passing smoke checks
and zero scorer candidate errors before moving on; waiting alone is not evidence
of success. The full entrypoint uses the same `runner.py` with `--mode full
--shard 0` through `3`. Give each new configuration a new run identifier; resume
only identical code and settings. The smoke's `--verify-only` option compares
previously saved outputs and tests resumption without repeating predictions.

Defaults cap the GPU smoke at 50 minutes, each full shard at three hours, and each
CPU scoring Job at one hour. These are per-Job limits. Account for all attempts
when applying a cumulative budget; do not blindly relaunch all shards after one
fails. Inspect `errors.json` and resume only the incomplete shard with the same
run identifier and inference settings. Read retries are bounded at 30 seconds for
mounted-Bucket visibility delays; missing/corrupt artifacts remain hard failures.
For example, `sh "$experiment/launch.sh" resume-0 "$bucket" full-v1` resumes
shard zero with a 45-minute cap. A successful complete resume preserves the earlier
error ledger as `errors-before-successful-resume.json` before clearing its active
`errors.json` marker. Include both Job attempts in the cost total.
Hub downloads retry transient transport errors and HTTP 408/429/500/502/503/504 responses
up to five attempts, with bounded backoff. Permanent errors still fail the Job;
successful retries retain the same revision and subsequent byte-identity checks.

## Artifacts and interpretation

The Bucket stores the manifest, environment/package versions, input hashes,
geometry, raw candidate Markdown, line predictions, stage timings, errors and
fixed-sample overlays. Candidate names follow the official scorer convention,
`{pdf_stem}_pg{page}_repeat1.md`. Empty text is valid if segmentation found no
lines; exceptions are separately reported and make the run fail.

`summary.json` records inference-stage totals and elapsed time. The first page
also has a CPU profile, so its runtime includes profiling overhead. Stock runs
before staged in this smoke: cold-start and cache effects mean the result cannot
isolate the benefit of model preparation or support a precise speedup claim.
Use Job running durations, including setup and failed attempts, for compute cost.

`score.py` uses official scorer revision
`f7cfe4c22098b154c76b6ec950d1c0a464eecf8d` in `allenai/olmocr`, including baseline
tests and its category-aware bootstrap. It checks complete candidate coverage,
raw-text identity and PDF hashes before calling the official `main()` function.
Thin observers capture its returned test results and confidence interval without
replacing scoring code. The official overall is the mean of per-JSONL category
scores, including the automatic baseline category; it is not the micro-average.
Both are exported, with raw test results and the complete official console output.
NumPy/Python seeds are fixed at 20260909 for 1,000 bootstrap samples.

Scoring artifacts live under `scores/RUN_ID/`: `summary.json`,
`test-results.jsonl`, `coverage.json`, `candidate-errors.json`,
`official-output.txt`, a PNG/SVG category chart, and bounded diagnostic examples
with original page, line overlay, prediction and test explanation. Diagnostic
selection happens after scoring and never changes the scored set. Full math/table
tests assess structured output that plain line text does not provide. Smoke
scores validate execution and should not be presented as benchmark estimates.

To retrieve private artifacts after a successful run:

```sh
hf buckets sync "hf://buckets/$bucket/scores/full-v1" ./kraken-results
```

The recognizer contains approximately 15.92 million parameters; the complete
pipeline also includes a segmenter. Its training card mentions historical,
contemporary and born-digital material without proportions, so results should
not be framed as proof of transfer from exclusively or mostly historical data.
