# ocr-bench

**There is no single best OCR model.** Rankings change depending on your documents — manuscript cards, printed books, and historical texts all produce different winners.

ocr-bench creates **per-collection leaderboards** using a VLM-as-judge approach, so you can find what works best for *your* documents rather than relying on generic benchmarks.

## Why?

Generic OCR benchmarks tell you which model wins *on average*. But if you're digitising 18th-century encyclopaedias, that average doesn't help — the best model for your documents might be the worst on someone else's.

ocr-bench lets you run the same set of OCR models on a sample of *your* collection, then uses a vision-language model to judge which produces the best transcription for each document. The result is a leaderboard specific to your data.

| Model | BPL card catalog | Britannica 1771 |
|-------|:---:|:---:|
| LightOnOCR-2 (1B) | **#1** | **#1** (1788) |
| GLM-OCR (0.9B) | #4 | #2 (1757) |
| DeepSeek-OCR (4B) | #3 | #4 (1429) |
| dots.ocr (1.7B) | #2 | #5 (972) |

Rankings flip completely between collections. The model that's #2 on BPL cards is dead last on Britannica.

## Hub-native by design

The entire evaluation loop lives on the Hugging Face Hub:

1. **Your dataset** on the Hub (images + optional ground truth)
2. **OCR models** run via [HF Jobs](https://huggingface.co/docs/hub/jobs) → outputs written as PRs on a Hub dataset
3. **VLM judge** via [HF Inference Providers](https://huggingface.co/docs/inference-providers) — only needs an HF token
4. **Results** published to a Hub dataset (leaderboard + pairwise comparisons)
5. **Viewer** as a [HF Space](https://huggingface.co/spaces) for browsing and human validation

No third-party API keys. No local GPU required. Everything is shareable via Hub URLs.

## Quickstart

```bash
pip install ocr-bench[viewer]

# 1. Run OCR models on your dataset
ocr-bench run <input-dataset> <output-repo> --max-samples 50

# 2. Judge outputs pairwise with a VLM
ocr-bench judge <output-repo>

# 3. Browse results + validate
ocr-bench view <output-repo>-results
```

## How it works

**`ocr-bench run`** launches OCR models on your dataset via [HF Jobs](https://huggingface.co/docs/hub/jobs). Each model writes its output as a PR on the same Hub dataset, keeping everything together without merge conflicts.

**`ocr-bench judge`** runs pairwise comparisons using a VLM judge (default: [Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) via HF Inference Providers). For each document, the judge sees the original image and two OCR outputs (anonymised as A/B) and picks the better transcription. Results are fit to a [Bradley-Terry model](https://en.wikipedia.org/wiki/Bradley%E2%80%93Terry_model) to produce ELO ratings with bootstrap 95% confidence intervals. Adaptive stopping halts early when rankings are statistically resolved.

**`ocr-bench view`** serves a local web viewer with a leaderboard, comparison browser, and human validation. Vote on comparisons to cross-check the automated judge with human judgement.

## Example results

Browse these on the Hub:
- [davanstrien/ocr-bench-britannica-results](https://huggingface.co/datasets/davanstrien/ocr-bench-britannica-results) — Encyclopaedia Britannica 1771, 5 models, 50 samples
- [davanstrien/bpl-ocr-bench-results](https://huggingface.co/datasets/davanstrien/bpl-ocr-bench-results) — Boston Public Library card catalog, 4 models, 150 samples

## Install

```bash
pip install ocr-bench            # Core (run + judge)
pip install ocr-bench[viewer]    # With web UI
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv pip install ocr-bench[viewer]
```

Requires Python >= 3.11 and an [HF token](https://huggingface.co/settings/tokens).

## Status

Working proof of concept. The core pipeline (run → judge → view) is functional. Not polished production software — expect rough edges.
