# CLAUDE.md — ocr-bench

OCR model evaluation toolkit. VLM-as-judge with per-dataset leaderboards on Hugging Face Hub.

> Historical decisions, smoke tests, and completed phase details are in [ARCHIVE.md](ARCHIVE.md).
> PoC release plan and checklist in [POC-CHECKLIST.md](POC-CHECKLIST.md).

## What This Project Does

Lets anyone answer: **"Which OCR model works best for MY documents?"**

Rankings change by document type — manuscript cards, printed books, historical texts, tables all produce different winners. This tool creates per-collection leaderboards.

## Current State (2026-03-02)

**233 tests passing**, ruff clean. Full pipeline works with smart defaults:

```
ocr-bench run <input-ds> <output-repo> --max-samples 50
ocr-bench judge <output-repo>
ocr-bench view <output-repo>-results
```

Smart defaults: auto-detects PR + main branch configs, auto-derives results repo name, adaptive stopping on by default. Zero flags needed for the common case.

### What's built

| Module | What it does |
|--------|-------------|
| `elo.py` | Bradley-Terry MLE via scipy, bootstrap 95% CIs, ELO scale |
| `judge.py` | VLM-as-judge prompt, Comparison dataclass, structured output schema |
| `dataset.py` | Flat, config-per-model, PR-based dataset loading, OCR column discovery |
| `backends.py` | API backends: InferenceProvider + OpenAI-compatible, concurrent calls |
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
- [x] Re-run BPL judge with `--save-results` using new publication workflow
- [x] Smart defaults: zero-flag `judge` command (auto-detect configs, auto-derive results repo, adaptive on by default)
- [x] Concurrent judge calls (`--concurrency N`)
- [x] Britannica E2E: 5 models, 2 judges (Qwen3-VL-235B + Qwen3.5-35B-A3B)
- [x] Performance: Arrow-level dataset merge (no per-row image decode), JPEG encoding, text column pre-fetch
- [x] FireRed-OCR (2.1B) added to registry + benchmarked on Britannica (#3 behind GLM-OCR/LightOnOCR-2)
- [x] Judge prompt: markdown formatting markers now neutral (not penalised or rewarded)
- [x] Fix `_extract_model_id()`: take last inference_info entry (not first) so inherited metadata doesn't shadow actual model
- [ ] Write README — "no single best model" as headline
- [ ] Choose a project name (captures "rankings depend on your documents")
- [x] Deploy viewer as HF Space — https://huggingface.co/spaces/davanstrien/ocr-bench-britannica-results-qwen35-viewer
- [ ] PyPI publishing workflow (GitHub Actions trusted publishing — almost ready)
- [x] Switch default judge to Qwen3.5-35B-A3B (fastest, zero parse failures, same cluster rankings as 122B and 27B)

### Known limitation — row alignment across PRs
`load_config_dataset()` merges configs by positional index — no alignment key. Safe if all model runs use the same `--seed`/`--max-samples` and the source dataset doesn't change. Future: add content hash column for validation.

### Phase 4: Blog + Visibility
- [ ] "There Is No Best OCR Model" blog post
- [ ] Cross-link repo, viewer, Hub datasets, blog

### Phase 5: Customization
- [ ] Judge prompt presets for GLAM document types
- [ ] Custom prompt and ignore list support
- [ ] Define leaderboard dataset schema
- [x] Adaptive stopping (on by default, `--no-adaptive` to opt out): run batches, compute BT-MLE + CIs, stop when adjacent-rank CIs don't overlap (ranking is statistically resolved). Avoids wasting judge calls when rankings are already clear.
- [x] Judge prompt: hallucination penalized more than commentary. Markdown formatting neutral. Blank page descriptions no longer lose to nonsense output.
- [ ] Judge comparison: run same dataset through different judges (e.g. Kimi K2.5 vs Qwen3.5-397B), compare BT-MLE ratings + CIs to see where judges agree/disagree. Overlapping CIs = single judge is fine; non-overlapping = jury mode adds value. Test on diverse document types — jury may only matter for ambiguous collections (e.g. index cards where everything ties).
- [ ] `--focus-pairs` for human validation: prioritize showing pairs with overlapping CIs in the vote UI, since those are the only ones where human input changes the ranking.
- [ ] Blank page filtering: skip comparisons where neither model produced meaningful text.

### If project gets traction
- [ ] Consolidate OCR model scripts into this repo + hub-sync
- [ ] CI/smoke tests
- [x] Britannica E2E (50 samples, 4 models, 2 judges)
- [x] Large-scale run: full BPL card catalog (453K images) with GLM-OCR — 21.9 hrs, $39 on L40S
- [ ] Large-scale runs (full Britannica, NLS index cards)
- [ ] CER/WER metrics alongside VLM judge
- [ ] `bench` command: single `ocr-bench bench <input-dataset>` chains run → judge → view

## Tooling

- **uv** for project management and running scripts
- **ruff** for linting and formatting

## Technical Reference

### Judge Models
- **Qwen3.5-35B-A3B (`novita:Qwen/Qwen3.5-35B-A3B`)** — **default**. Zero parse failures, ~21 comps/min via Inference Providers (HF token only). Same cluster rankings as 122B and 27B. Best speed/quality tradeoff.
- **Qwen3.5-122B-A10B (`novita:Qwen/Qwen3.5-122B-A10B`)** — zero parse failures, ~19 comps/min. Slightly more separation between clusters. Good for authoritative runs.
- **Qwen3.5-27B (`novita:Qwen/Qwen3.5-27B`)** — zero parse failures, ~12 comps/min. Dense model, slower than MoE alternatives. Same clusters.
- **Kimi K2.5 (`novita:moonshotai/Kimi-K2.5`)** — best human agreement but 2-5% parse failures from degeneration. No longer default.
- **Qwen3-VL-235B (`novita:Qwen/Qwen3-VL-235B-A22B-Instruct`)** — zero parse failures, ~6 comps/min. Good but slower and Novita disconnects on long runs.
- **Qwen3-VL-30B-A3B (offline vLLM)** — best offline judge
- **7B/8B** — biased toward verbose output, not recommended as primary

### Core Benchmark Models
| Model | Size | Best on |
|-------|------|---------|
| DeepSeek-OCR | 4B | Most consistent across datasets |
| GLM-OCR | 0.9B | Card catalogs, Britannica (235B judge) |
| LightOnOCR-2 | 1B | BPL manuscript cards, Britannica (35B judge) |
| FireRed-OCR | 2.1B | Mid-pack on Britannica (#3), good on clean printed text, loses on degraded/stamps |
| dots.ocr | 1.7B | Worst on Britannica (1-2% win rate) |

### Key Findings
1. **No single best OCR model** — rankings shuffle by document type
2. **DeepSeek-OCR most consistent** — #1 on UFO (diverse docs), but #3 on BPL and Britannica
3. **LightOnOCR-2 best on BPL** — #1 (Kimi) or #2 (Qwen3-VL) on card catalogs; #1 on Britannica (35B judge), #2 (235B judge)
4. **GLM-OCR best on Britannica (235B)** — #1 (1779 ELO), but #2 with 35B judge. Strong on 18th century printed text.
5. **Document type > model size** — 0.9B beats 4B on some collections
6. **Judge model size matters** — 170B closest to human rankings
7. **Judges agree on clusters, swap within** — Kimi K2.5 and Qwen3-VL-235B produce same top-2/bottom-2 groupings on BPL but swap adjacent models. CIs overlap between judges, confirming fine ordering is noise. Same pattern on Britannica: 235B and 35B agree on clusters but swap #1/#2.
8. **Jury mode works** — eliminates single-judge bias
9. **dots.ocr struggles on historical printed text** — 1-2% win rate on Britannica vs 55% on BPL. Model-dataset fit matters enormously.
10. **FireRed-OCR mid-pack on Britannica** — #3 (1551 ELO, 35B judge). Loses to GLM/LightOn on degraded text (garbling) and gets penalised for aggressive markdown formatting (`# headings`, `**bold**` on everything). Beats DeepSeek-OCR (52% vs 39% win rate).
11. **Qwen3.5-35B-A3B is the best default judge** — assessed 35B-A3B, 27B, and 122B-A10B on identical 96-comparison Britannica benchmark. All three: zero parse failures, same cluster rankings (top-3: FireRed/LightOn/GLM, then DeepSeek, then dots). 35B-A3B fastest (4:35 vs 7:55 vs 4:57). Now the default.

### Results on Hub
- `davanstrien/bpl-ocr-bench-results` — BPL card catalog, 4 models, Kimi K2.5 judge (2026-02-24)
- `davanstrien/ocr-bench-britannica-results` — Britannica 1771, 5 models, Qwen3-VL-235B judge (2026-02-25, partial — Novita disconnects)
- `davanstrien/ocr-bench-britannica-results-qwen35` — Britannica 1771, 5 models, Qwen3.5-35B-A3B judge (2026-03-02)
- `davanstrien/ocr-bench-rubenstein-judge` — 50 samples, 300 comparisons
- `davanstrien/ocr-bench-ufo-judge-30b` — cross-validation on UFO-ColPali
- `davanstrien/ocr-bench-rubenstein-judge-kimi-k25` — Kimi K2.5 170B
- `davanstrien/ocr-bench-judge-eval-35b` — judge assessment: Qwen3.5-35B-A3B, 96 comps, 10 samples
- `davanstrien/ocr-bench-judge-eval-27b` — judge assessment: Qwen3.5-27B, 96 comps, 10 samples
- `davanstrien/ocr-bench-judge-eval-122b` — judge assessment: Qwen3.5-122B-A10B, 96 comps, 10 samples

### OCR Output Datasets
- `davanstrien/ocr-bench-rubenstein` — 4 models, PRs, 50 samples (index cards, all tie)
- `davanstrien/ocr-bench-ufo` — 4 models, PRs, 50 samples (diverse docs, clear differentiation)
- `davanstrien/bpl-ocr-bench` — 4 models, PRs, 150 samples (BPL card catalog)
- `davanstrien/ocr-bench-britannica` — 5 models (incl. FireRed-OCR), 4 PRs + 1 merged, 50 samples (Encyclopaedia Britannica 1771)
- `davanstrien/bpl-card-catalog-glm-ocr` — **full BPL run**, 453K cards, GLM-OCR, 21.9 hrs on L40S, $39 (2026-02-27)

## Connections

- **uv-scripts/ocr on Hub**: OCR model scripts stay there for now
- **FineCorpus**: OCR quality = training data quality
- **NLS**: Index cards as flagship benchmark dataset
