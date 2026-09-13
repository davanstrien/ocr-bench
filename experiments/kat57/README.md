# Kat57 ground-truth conversion

This experiment converts Lund University Library's 36 GB
[Kat57 Zenodo release](https://zenodo.org/records/14679534) into a data-only
Hugging Face dataset that `ocr-bench run` can consume directly.

The source contains 10,695 PNG images paired with eScriptorium PAGE XML. The
converter verifies the published ZIP MD5, extracts human-transcribed
line text in XML order, retains the original PAGE XML, writes bounded-memory
Parquet shards, and uploads each shard before creating the next one.

## Published artifacts

- [`tadad/kat57-ground-truth`](https://huggingface.co/datasets/tadad/kat57-ground-truth)
  is the complete conversion: 10,695 rows in 84 Parquet shards. The Hub dataset
  server reports the expected row count and no pending, failed, or partial work.
- [`tadad/kat57-ocr-bench-results`](https://huggingface.co/datasets/tadad/kat57-ocr-bench-results)
  is a 50-card end-to-end smoke test. Tesseract OCR outputs were scored against
  the references and published as `metrics`, `metric_details`, and
  `metric_metadata` configs.
- [`tadad/kat57-ocr-bench-500-results`](https://huggingface.co/datasets/tadad/kat57-ocr-bench-500-results)
  is the comparative acceptance run: 16 OCR models scored on a deterministic
  500-card sample, for 8,000 reference-based evaluations. It contains CER/WER
  results only; no VLM judge or ELO results are part of the Kat57 benchmark.

Run it on a cheap CPU Job; `cpu-basic` has just enough disk because only the
source archive and one output shard coexist:

```bash
hf jobs uv run \
  --flavor cpu-basic \
  --timeout 8h \
  --secrets HF_TOKEN \
  experiments/kat57/prepare.py \
  tadad/kat57-ground-truth
```

For a disposable integration test, publish a small private subset to a fresh
repository:

```bash
hf jobs uv run \
  --flavor cpu-basic \
  --timeout 8h \
  --secrets HF_TOKEN \
  experiments/kat57/prepare.py \
  tadad/kat57-ground-truth-smoke \
  --max-samples 10 \
  --private
```

For a reproducible benchmark sample drawn across the full archive, combine
`--max-samples` with `--sample-seed`:

```bash
hf jobs uv run \
  --flavor cpu-basic \
  --timeout 8h \
  --secrets HF_TOKEN \
  experiments/kat57/prepare.py \
  tadad/kat57-ground-truth-500 \
  --max-samples 500 \
  --sample-seed 57
```

The converter refuses to overwrite a repository that already contains data.
