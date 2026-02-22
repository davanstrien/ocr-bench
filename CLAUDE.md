# CLAUDE.md — ocr-bench

OCR model evaluation toolkit. VLM-as-judge with per-dataset leaderboards on Hugging Face Hub.

## What This Project Does

Lets anyone answer: **"Which OCR model works best for MY documents?"**

OCR model rankings change depending on document type — manuscript cards, printed books, historical texts, tables all produce different winners. A single leaderboard is misleading. This tool lets you create a leaderboard for your specific collection.

## Desired Outcomes

1. **Evaluate OCR models on any dataset** — run models, then judge output quality using VLM-as-judge. Produces ELO rankings and per-dimension scores.
2. **Customizable evaluation criteria** — presets for common document types (index cards, manuscripts, tables, printed books), custom prompts for power users, configurable ignore lists.
3. **Per-dataset leaderboards on Hub** — every evaluation run publishes a leaderboard dataset to Hub. The judge prompt and metadata are stored for reproducibility.
4. **Leaderboard viewer on Hub** — reads any evaluation dataset matching the standard schema. Approach TBD.
5. **Blog post** — "There Is No Best OCR Model". Ties into FineCorpus narrative.

## Tooling

- **uv** for project management and running scripts
- **ruff** for linting and formatting

## Repo Architecture

### Now: Ship fast

GitHub repo with the eval tooling as a uv project. OCR model scripts stay on Hub in `uv-scripts/ocr/` where they already work with `hf jobs uv run`. No sync machinery — just cross-reference.

```
ocr-bench/                        ← GitHub repo
├── README.md                     ← polished from OCR-BENCHMARK.md
├── CLAUDE.md
├── pyproject.toml                ← uv project
└── src/ or ocr_bench/            ← eval tooling (judge, publishing, presets)
```

OCR model scripts stay at `davanstrien/uv-scripts` on Hub (14 scripts, all reviewed).

### Later (if the project gets traction): Consolidate

Move OCR model scripts into this repo under a `uv-scripts/` subdirectory. Use [hub-sync](https://github.com/huggingface/hub-sync) GitHub Action to sync that subdirectory to Hub so `hf jobs uv run` keeps working. One source of truth, automatic sync. Only worth the effort if people are actually using the benchmark.

## What Already Exists

All current code is in `~/Documents/projects/uv-scripts/ocr/`. Key files to port:

| Script | What it does |
|--------|-------------|
| `ocr-vllm-judge.py` | Offline VLM judge. Jury mode (multiple judges vote). Structured output via xgrammar — 0 parse failures over 300+ comparisons. ELO. |
| `ocr-jury-bench.py` | API-based judge via HF Inference Providers. Default: Kimi K2.5 via Novita. |
| `ocr-human-eval.py` | Blind A/B Gradio app for human validation of judge quality. |
| `ocr-bench-run.py` | Orchestrator: launches N OCR models via HF Jobs, auto-merges PRs. |
| `OCR-BENCHMARK.md` | 31KB design doc — methodology, results, findings. Becomes README. |
| `SHIPPING.md` | Architecture decisions. |
| `REVIEW.md` | Review notes for all 14 OCR model scripts. |
| 14 OCR model scripts | All reviewed + fixed 2026-02-16. Core 4: DeepSeek-OCR, GLM-OCR, LightOnOCR-2, DoTS.ocr. |

### Results on Hub
- `davanstrien/ocr-bench-rubenstein-judge` — 50 samples, 300 comparisons, jury of 2
- `davanstrien/ocr-bench-ufo-judge-30b` — cross-validation on UFO-ColPali
- `davanstrien/ocr-bench-rubenstein-judge-kimi-k25` — Kimi K2.5 170B
- Human validation: 30 blind comparisons, Kimi K2.5 matches human rankings best

## Validated Findings (for README/blog)

1. **No single best OCR model** — rankings shuffle by document type
2. **DeepSeek-OCR most consistent** — #1 or #2 across all datasets
3. **Document type > model size** — 0.9B beats 4B on some collections
4. **Judge model size matters** — small judges biased toward verbose output; 170B closest to human
5. **Jury mode works** — eliminates single-judge bias
6. **Structured output reliable** — 0 parse failures at scale via xgrammar

## Prior Art

### DoTS.ocr
`https://github.com/rednote-hilab/dots.ocr/blob/master/tools/elo_score_prompt.py`
Pairwise VLM judge (Gemini), fixed criteria, generous ties, ignores formatting/layout. No jury, no human validation, no customization. Their explicit ignore list is worth adopting as a configurable option.

### Inspect AI (UK AISI)
`https://inspect.aisi.org.uk/`
Open-source eval framework. **Decision (2026-02-20): NO-GO.** Inspect evaluates one model at a time (input → model → score against target). Our pipeline is fundamentally different: pre-computed outputs from N models, pairwise VLM judge, ELO ranking. Adopting Inspect would mean no-op solvers, reimplementing jury logic inside a custom scorer, and computing ELO outside the framework. Not worth the impedance mismatch. Build standalone.

### HF Eval Results System
`https://huggingface.co/docs/hub/en/eval-results`
Decentralized eval results on Hub: benchmark datasets define `eval.yaml`, model repos store scores in `.eval_results/*.yaml`, results appear on model cards + benchmark leaderboards. Useful as a **publishing/discovery layer** — after our judge produces rankings, publish into this system. Not a replacement for our eval tooling. The `evaluation_framework` enum would need an `ocr-bench` entry added.

## Sprint Plan

### Phase 1: Foundation
- [x] Investigate Inspect AI — **NO-GO** (2026-02-20). Build standalone.
- [x] Init uv project with `src/ocr_bench/` layout (2026-02-20)
- [x] Port `elo.py` (Bradley-Terry, K=32) + tests (2026-02-20)
- [x] Port `judge.py` (prompt template, Comparison, structured output schema, image utils) + tests (2026-02-20)
- [x] Port `dataset.py` (flat, config-per-model, PR-based, OCR column discovery) + tests (2026-02-20)
- [x] Port `backends.py` (API only: InferenceProvider + OpenAI-compatible) + tests (2026-02-20)
- [x] Port `publish.py` (comparisons, leaderboard, metadata configs) + tests (2026-02-20)
- [x] CLI entrypoint: `ocr-bench judge` with auto jury mode (2026-02-20)
- [x] **86 tests passing**, ruff clean, ty clean (2026-02-20)
- [ ] Write README — key finding ("no single best model") as the headline
- [ ] Adapt `OCR-BENCHMARK.md` methodology section for README

#### Design decisions made in Phase 1
- **API-only backends** for Phase 1. vLLM offline stays as standalone UV script for HF Jobs.
- **Single `ocr-bench judge` command**. Auto-detects jury mode when multiple `--model` flags given.
- **Default judge**: `novita:moonshotai/Kimi-K2.5` (was `Kimi-K2.5-Instruct`, model ID changed on Hub).
- **stamina** for retry (exponential backoff + jitter), **structlog** for structured logging.
- **`DatasetError`** instead of `sys.exit()` — CLI catches and prints.

#### Smoke test results (2026-02-20)

**Rubenstein manuscript cards** (`davanstrien/ocr-bench-rubenstein --from-prs --max-samples 5`):
- 4 models, 30 comparisons, 30/30 valid, **all ties**
- Correct result: these are short index cards, all 4 models produce near-identical output
- Validates the prompt's "only pick a winner when there is a clear quality difference" instruction

**UFO-ColPali diverse docs** (`davanstrien/ocr-bench-ufo --from-prs --max-samples 5`):
- 4 models, 30 comparisons, 28/30 valid (2 truncated by max_tokens=300)
- Clear ranking differentiation:

| Rank | Model | ELO | Win% |
|------|-------|-----|------|
| 1 | DeepSeek-OCR | 1539 | 64% |
| 2 | LightOnOCR-2-1B | 1530 | 57% |
| 3 | dots.ocr | 1481 | 43% |
| 4 | GLM-OCR | 1449 | 36% |

Confirms core finding: **rankings change by document type** (all-tie on cards, clear winners on diverse docs).

#### Issues found in smoke test
- **Judge response truncation**: `max_tokens=300` too low for some comparisons — 2/30 parse failures on UFO. Should bump default or make configurable.
- **BPL dataset** (`davanstrien/bpl-card-catalog-glm-ocr-bench`) has 500 samples but only 2 unique models (GLM-OCR + DeepSeek-OCR-2 x2 columns). No PRs. Works with flat mode but less interesting for benchmarking.

#### Available test datasets

| Dataset | Models | Shape | Samples | Notes |
|---------|--------|-------|---------|-------|
| `davanstrien/ocr-bench-rubenstein` | 4 (GLM, DeepSeek, dots, LightOn) | PRs | 50 | Index cards, all models tie |
| `davanstrien/ocr-bench-ufo` | 4 (same) | PRs | 50 | Diverse docs, clear differentiation |
| `davanstrien/bpl-card-catalog-glm-ocr-bench` | 2 unique (GLM, DeepSeek-OCR-2) | Flat | 500 | Card catalog, only 2 models |

### Phase 2: Customization + Polish
- [x] Fix `max_tokens` truncation — bumped default to 512, added `--max-tokens` CLI flag (2026-02-20)
- [x] Enrich comparison data with OCR texts — `text_a`, `text_b`, `col_a`, `col_b` in published comparisons (2026-02-22)
- [x] Gradio results viewer — `ocr-bench browse <repo_id>` with leaderboard + comparison browser tabs (2026-02-22)
- [x] `ocr-bench run` — orchestrator to launch N OCR models via HF Jobs (2026-02-22)
- [x] Viewer: fix empty col_a/col_b display (no empty parens), add model-pair win/loss summary (2026-02-22)
- [x] `ocr-bench validate` — blind human A/B validation with judge agreement tracking (2026-02-22)
- [x] **162 tests passing**, ruff clean (2026-02-22)
- [ ] Judge prompt presets for GLAM document types
- [ ] Custom prompt and ignore list support
- [ ] Define leaderboard dataset schema (publishing already works via `--save-results`)

#### Design decisions made in Phase 2
- **Gradio as optional dep** — `pip install ocr-bench[viewer]`. Core judge pipeline stays lightweight.
- **Viewer reads from Hub** — no local data needed, any published results dataset works.
- **Text-only for now** — image display deferred (requires join back to source dataset by sample_idx).
- **`run.py` references scripts on Hub** — no local copies. MODEL_REGISTRY maps slugs to script URLs + GPU flavors.
- **`validate.py` reads enriched comparisons** — uses `text_a`/`text_b` from published results, no need to join back to source dataset.
- **AgreementStats** tracks agree/soft-disagree/hard-disagree. Hard disagree rate > 25% flags judge as miscalibrated.

### Phase 3: Results Visibility
Three related pieces that make results credible and shareable:

**Leaderboard viewer** — DONE (2026-02-22)
- `ocr-bench browse` reads any result dataset matching our schema
- Two tabs: leaderboard table + comparison browser with filter by winner/model
- Tested against `davanstrien/ocr-bench-rubenstein-judge` — works with old data (texts empty), will show full OCR when re-published

**What's next for the viewer**
- [ ] Re-run judge with `--save-results` to publish enriched data (texts + column names)
- [ ] Add document image display (join comparisons back to source dataset by sample_idx)
- [ ] Deploy as HF Space for public access
- [x] Clean up model name display when col_a/col_b are empty (hide empty parens) (2026-02-22)
- [x] Model-pair win/loss summary above comparison slider (2026-02-22)

**Human validation integration** — DONE (2026-02-22)
- `ocr-bench validate <results-repo>` — blind A/B Gradio app ported from `ocr-human-eval.py`
- Reads enriched comparisons from Hub (text_a, text_b already available)
- Split-jury cases shown first (most informative for validation)
- AgreementStats: agree/soft-disagree/hard-disagree with confidence thresholds
- Live agreement banner during annotation, Results tab with summary
- JSON persistence with atomic writes, supports resume
- Prior data: 30 blind comparisons showed Kimi K2.5 matches human rankings best

**Side-by-side comparison browser** — DONE (2026-02-22, part of `ocr-bench browse`)

These all feed into each other — the README/blog become much more powerful with a live leaderboard, human validation scores, and browsable comparisons.

### Before Shipping
- [ ] Choose a project name — "ocr-bench" is placeholder. Want something that captures "rankings depend on your documents"

### Phase 4: Blog + Visibility
- [ ] "There Is No Best OCR Model" blog post
- [ ] Cross-link repo ↔ viewer ↔ Hub datasets ↔ blog

### Phase 5: Flagship Run (E2E pipeline ready)

Full pipeline now works:
```
ocr-bench run <input-ds> <output-repo> --max-samples 50
ocr-bench judge <output-repo> --from-prs --save-results <results-repo>
ocr-bench browse <results-repo>
ocr-bench validate <results-repo> --n 30
```

- [ ] BPL card catalog run (50 samples, 4 models) — first real E2E test
- [ ] Large-scale run on Britannica, NLS index cards, or BPL catalog

### If project gets traction
- [ ] Consolidate OCR model scripts into this repo + hub-sync to Hub
- [ ] CI/smoke tests
- [ ] Scheduled benchmarking
- [ ] CER/WER metrics alongside VLM judge

## Technical Reference

### ELO
Bradley-Terry, K=32, initial 1500. Position-bias randomization.

### Judge Models
- **Kimi K2.5 (`novita:moonshotai/Kimi-K2.5`)** — best human agreement, default for API
- **Qwen3-VL-30B-A3B (offline vLLM)** — best offline judge
- **7B/8B** — biased toward verbose output, not recommended as primary

### Core Benchmark Models
| Model | Size | Best on |
|-------|------|---------|
| DeepSeek-OCR | 4B | Most consistent across datasets |
| GLM-OCR | 0.9B | Card catalogs |
| LightOnOCR-2 | 1B | Manuscript cards |
| dots.ocr | 1.7B | Historical medical texts |

## Connections

- **uv-scripts/ocr on Hub**: OCR model scripts stay there for now
- **FineCorpus**: OCR quality = training data quality
- **NLS**: Index cards as flagship benchmark dataset
- **Obsidian**: `~/Documents/obsidian/Work/Projects/ocr-benchmark.md`
