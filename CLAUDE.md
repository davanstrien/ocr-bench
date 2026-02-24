# CLAUDE.md — ocr-bench

OCR model evaluation toolkit. VLM-as-judge with per-dataset leaderboards on Hugging Face Hub.

> Historical decisions, smoke tests, and completed phase details are in [ARCHIVE.md](ARCHIVE.md).

## What This Project Does

Lets anyone answer: **"Which OCR model works best for MY documents?"**

Rankings change by document type — manuscript cards, printed books, historical texts, tables all produce different winners. This tool creates per-collection leaderboards.

## Current State (2026-02-23)

**177 tests passing**, ruff clean. Full pipeline works:

```
ocr-bench run <input-ds> <output-repo> --max-samples 50
ocr-bench judge <output-repo> --from-prs --save-results <results-repo>
ocr-bench view <results-repo>
```

### What's built

| Module | What it does |
|--------|-------------|
| `elo.py` | Bradley-Terry MLE via scipy, bootstrap 95% CIs, ELO scale |
| `judge.py` | VLM-as-judge prompt, Comparison dataclass, structured output schema |
| `dataset.py` | Flat, config-per-model, PR-based dataset loading, OCR column discovery |
| `backends.py` | API backends: InferenceProvider + OpenAI-compatible |
| `publish.py` | Publish comparisons + leaderboard to Hub; incremental load from existing results |
| `run.py` | Orchestrator: launch N OCR models via HF Jobs |
| `validate.py` | Human A/B validation data layer, agreement stats, human ELO |
| `viewer.py` | Data loading for results viewer (pure functions) |
| `web.py` | FastAPI + HTMX unified viewer (browse + validate in one app) |
| `cli.py` | CLI: `judge` (incremental + `--full-rejudge`), `run`, `view` |

### Viewer (`ocr-bench view`)

FastAPI + HTMX, Tufte-inspired. Keyboard-first (`←`/`→` navigate, `a`/`b`/`t` vote, `r` reveal). Browsing IS validation — vote buttons on every comparison, voting reveals judge verdict. Leaderboard with judge ELO + human ELO. Document images lazy-loaded. HTMX partial updates.

### Dependencies

- Core: `datasets`, `huggingface-hub`, `openai`, `pillow`, `stamina`, `structlog`
- `pip install ocr-bench[viewer]` — FastAPI + uvicorn + jinja2 (the web viewer)

## Next Steps

### Immediate — Publication + polish
- [x] Incremental judge mode: `--save-results` to existing repo loads comparisons, skips judged pairs, judges only new ones, merges and refits BT-MLE. `--full-rejudge` to override.
- [x] Results repo structure: default config = leaderboard, `comparisons` = append-only, `metadata` = append-only run log.
- [ ] Re-run BPL judge with `--save-results` using new publication workflow
- [ ] Write README — "no single best model" as headline
- [ ] Choose a project name (captures "rankings depend on your documents")

### Known limitation — row alignment across PRs
`load_config_dataset()` merges configs by positional index — no alignment key. Safe if all model runs use the same `--seed`/`--max-samples` and the source dataset doesn't change. Future: add content hash column for validation.

### Phase 4: Blog + Visibility
- [ ] "There Is No Best OCR Model" blog post
- [ ] Deploy viewer as HF Space
- [ ] Cross-link repo, viewer, Hub datasets, blog

### Phase 5: Customization
- [ ] Judge prompt presets for GLAM document types
- [ ] Custom prompt and ignore list support
- [ ] Define leaderboard dataset schema
- [x] Adaptive stopping (`--adaptive` flag): run batches, compute BT-MLE + CIs, stop when adjacent-rank CIs don't overlap (ranking is statistically resolved). Avoids wasting judge calls when rankings are already clear.
- [ ] Judge comparison: run same dataset through different judges (e.g. Kimi K2.5 vs Qwen3.5-397B), compare BT-MLE ratings + CIs to see where judges agree/disagree. Overlapping CIs = single judge is fine; non-overlapping = jury mode adds value. Test on diverse document types — jury may only matter for ambiguous collections (e.g. index cards where everything ties).
- [ ] `--focus-pairs` for human validation: prioritize showing pairs with overlapping CIs in the vote UI, since those are the only ones where human input changes the ranking.

### If project gets traction
- [ ] Consolidate OCR model scripts into this repo + hub-sync
- [ ] CI/smoke tests
- [ ] Large-scale runs (Britannica, NLS index cards)
- [ ] CER/WER metrics alongside VLM judge

## Tooling

- **uv** for project management and running scripts
- **ruff** for linting and formatting

## Technical Reference

### Judge Models
- **Kimi K2.5 (`novita:moonshotai/Kimi-K2.5`)** — best human agreement, default
- **Qwen3-VL-30B-A3B (offline vLLM)** — best offline judge
- **7B/8B** — biased toward verbose output, not recommended as primary

### Core Benchmark Models
| Model | Size | Best on |
|-------|------|---------|
| DeepSeek-OCR | 4B | Most consistent across datasets |
| GLM-OCR | 0.9B | Card catalogs |
| LightOnOCR-2 | 1B | Manuscript cards |
| dots.ocr | 1.7B | Historical medical texts |

### Key Findings
1. **No single best OCR model** — rankings shuffle by document type
2. **DeepSeek-OCR most consistent** — #1 or #2 across all datasets
3. **Document type > model size** — 0.9B beats 4B on some collections
4. **Judge model size matters** — 170B closest to human rankings
5. **Jury mode works** — eliminates single-judge bias

### Results on Hub
- `davanstrien/bpl-ocr-bench-results` — BPL card catalog, 4 models (needs re-run)
- `davanstrien/ocr-bench-rubenstein-judge` — 50 samples, 300 comparisons
- `davanstrien/ocr-bench-ufo-judge-30b` — cross-validation on UFO-ColPali
- `davanstrien/ocr-bench-rubenstein-judge-kimi-k25` — Kimi K2.5 170B

### Test Datasets
- `davanstrien/ocr-bench-rubenstein` — 4 models, PRs, 50 samples (index cards, all tie)
- `davanstrien/ocr-bench-ufo` — 4 models, PRs, 50 samples (diverse docs, clear differentiation)
- `davanstrien/bpl-ocr-bench` — 4 models, PRs, 150 samples (BPL card catalog)

## Connections

- **uv-scripts/ocr on Hub**: OCR model scripts stay there for now
- **FineCorpus**: OCR quality = training data quality
- **NLS**: Index cards as flagship benchmark dataset
