# ARCHIVE.md — ocr-bench development history

Historical decisions, smoke tests, and completed phase details. Moved from CLAUDE.md to keep the main file focused on current state and next steps.

## What Already Existed (before porting)

All original code was in `~/Documents/projects/uv-scripts/ocr/`. Key files ported:

| Script | What it does |
|--------|-------------|
| `ocr-vllm-judge.py` | Offline VLM judge. Jury mode (multiple judges vote). Structured output via xgrammar — 0 parse failures over 300+ comparisons. ELO. |
| `ocr-jury-bench.py` | API-based judge via HF Inference Providers. Default: Kimi K2.5 via Novita. |
| `ocr-human-eval.py` | Blind A/B app for human validation of judge quality. (Now `ocr-bench view`) |
| `ocr-bench-run.py` | Orchestrator: launches N OCR models via HF Jobs, auto-merges PRs. |
| `OCR-BENCHMARK.md` | 31KB design doc — methodology, results, findings. Becomes README. |
| `SHIPPING.md` | Architecture decisions. |
| `REVIEW.md` | Review notes for all 14 OCR model scripts. |
| 14 OCR model scripts | All reviewed + fixed 2026-02-16. Core 4: DeepSeek-OCR, GLM-OCR, LightOnOCR-2, DoTS.ocr. |

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

## Phase 1: Foundation (completed 2026-02-20)

- [x] Investigate Inspect AI — **NO-GO**. Build standalone.
- [x] Init uv project with `src/ocr_bench/` layout
- [x] Port `elo.py` (Bradley-Terry, K=32) + tests
- [x] Port `judge.py` (prompt template, Comparison, structured output schema, image utils) + tests
- [x] Port `dataset.py` (flat, config-per-model, PR-based, OCR column discovery) + tests
- [x] Port `backends.py` (API only: InferenceProvider + OpenAI-compatible) + tests
- [x] Port `publish.py` (comparisons, leaderboard, metadata configs) + tests
- [x] CLI entrypoint: `ocr-bench judge` with auto jury mode
- [x] **86 tests passing**, ruff clean, ty clean

### Design decisions
- **API-only backends** for Phase 1. vLLM offline stays as standalone UV script for HF Jobs.
- **Single `ocr-bench judge` command**. Auto-detects jury mode when multiple `--model` flags given.
- **Default judge**: `novita:moonshotai/Kimi-K2.5` (was `Kimi-K2.5-Instruct`, model ID changed on Hub).
- **stamina** for retry (exponential backoff + jitter), **structlog** for structured logging.
- **`DatasetError`** instead of `sys.exit()` — CLI catches and prints.

### Smoke test results

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

### Issues found in smoke test
- **Judge response truncation**: `max_tokens=300` too low for some comparisons — 2/30 parse failures on UFO. Bumped default to 512 in Phase 2.
- **BPL dataset** (`davanstrien/bpl-card-catalog-glm-ocr-bench`) has 500 samples but only 2 unique models (GLM-OCR + DeepSeek-OCR-2 x2 columns). No PRs. Works with flat mode but less interesting for benchmarking.

## Phase 2: Customization + Polish (completed 2026-02-22)

- [x] Fix `max_tokens` truncation — bumped default to 512, added `--max-tokens` CLI flag
- [x] Enrich comparison data with OCR texts — `text_a`, `text_b`, `col_a`, `col_b` in published comparisons
- [x] ~~Gradio results viewer~~ — superseded by FastAPI + HTMX viewer in Phase 3
- [x] `ocr-bench run` — orchestrator to launch N OCR models via HF Jobs
- [x] `ocr-bench validate` — blind human A/B validation with judge agreement tracking
- [x] **162 tests passing**, ruff clean

### Design decisions
- **Viewer as optional dep** — `pip install ocr-bench[viewer]`. Core judge pipeline stays lightweight.
- **Viewer reads from Hub** — no local data needed, any published results dataset works.
- **`run.py` references scripts on Hub** — no local copies. MODEL_REGISTRY maps slugs to script URLs + GPU flavors.
- **`validate.py` reads enriched comparisons** — uses `text_a`/`text_b` from published results, no need to join back to source dataset.
- **AgreementStats** tracks agree/soft-disagree/hard-disagree. Hard disagree rate > 25% flags judge as miscalibrated.

## Phase 3: Results Visibility (completed 2026-02-23)

**Unified FastAPI + HTMX viewer** replaced separate Gradio `browse` and `validate` commands:
- `ocr-bench view <results-repo>` — browsing IS validation
- Keyboard-first: `←`/`→` navigate, `a`/`b`/`t` vote, `r` reveal
- Leaderboard with judge ELO + human ELO side-by-side
- Head-to-head pair summary table, winner/model filter dropdowns
- Document images lazy-loaded from source dataset
- HTMX partial page updates, stats auto-refresh on vote
- JSON annotation persistence with atomic writes, resume support

### Design decisions
- **FastAPI + HTMX over Gradio** — faster, keyboard-driven, no framework overhead. Jinja2 templates for server-rendered HTML. HTMX for partial updates without JS framework.
- **One app, voting always optional** — no separate browse/validate modes. Vote buttons always present; voting reveals the judge verdict. Votes accumulate into human ELO on the leaderboard.
- **Tufte-inspired design** — high data-ink ratio. No CSS framework. system-ui for UI, monospace for OCR text. Color used sparingly (only agreement feedback). Tables without vertical lines.
- **`viewer` extra = FastAPI stack** — `fastapi`, `uvicorn[standard]`, `jinja2`, `python-multipart`. Gradio moved to separate `gradio` extra for backward compat.
- **Data layer unchanged** — `web.py` imports pure functions from `viewer.py` and `validate.py`. No modifications to existing data code.

## BPL E2E Run (2026-02-22)

- 4 jobs launched, all completed, 4 PRs on `davanstrien/bpl-ocr-bench`
- 150/150 valid comparisons, 0 parse failures
- All ties (expected for short index cards)
- Published to `davanstrien/bpl-ocr-bench-results`
- **Bug**: `_find_text_column()` picked wrong column (`text` instead of `markdown`) — all comparisons used Tesseract baseline, not model OCR output. Need to re-run after fix.

### Bugs found & fixed
- `run.py`: removed unsupported `labels=` kwarg from `run_uv_job()`
- `run.py`: changed `inspect_job(job_id)` → `inspect_job(job_id=job_id)` (keyword-only)
- `dataset.py`: `_find_text_column()` wrong column priority — fixed (2026-02-23)

## Available test datasets

| Dataset | Models | Shape | Samples | Notes |
|---------|--------|-------|---------|-------|
| `davanstrien/ocr-bench-rubenstein` | 4 (GLM, DeepSeek, dots, LightOn) | PRs | 50 | Index cards, all models tie |
| `davanstrien/ocr-bench-ufo` | 4 (same) | PRs | 50 | Diverse docs, clear differentiation |
| `davanstrien/bpl-card-catalog-glm-ocr-bench` | 2 unique (GLM, DeepSeek-OCR-2) | Flat | 500 | Card catalog, only 2 models |
| `davanstrien/bpl-ocr-bench` | 4 (GLM, DeepSeek, dots, LightOn) | PRs | 150 | BPL card catalog, E2E test |

## Results on Hub
- `davanstrien/ocr-bench-rubenstein-judge` — 50 samples, 300 comparisons, jury of 2
- `davanstrien/ocr-bench-ufo-judge-30b` — cross-validation on UFO-ColPali
- `davanstrien/ocr-bench-rubenstein-judge-kimi-k25` — Kimi K2.5 170B
- `davanstrien/bpl-ocr-bench-results` — BPL card catalog, 4 models (needs re-run after text column fix)
- Human validation: 30 blind comparisons showed Kimi K2.5 matches human rankings best
