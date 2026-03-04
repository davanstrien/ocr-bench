# CLAUDE.md — ocr-bench

OCR model evaluation toolkit. Answers: **"Which OCR model works best for MY documents?"**

Rankings change by document type — the best model for manuscript cards is different from the best for printed books or historical texts. This tool creates per-collection leaderboards using pairwise VLM-as-judge comparisons, so users can find what works for their specific documents.

**Pipeline**: `run` (launch OCR models via HF Jobs) → `judge` (pairwise VLM comparison → Bradley-Terry ELO) → `view` (leaderboard + human validation). Everything lives on the Hugging Face Hub — no local GPU needed.

## Architecture

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

## Tooling

- **uv** for project management and running scripts
- **ruff** for linting and formatting
- Release process documented in [RELEASING.md](RELEASING.md)

## Development

```bash
uv sync --dev --extra viewer
uv run ruff check src/ tests/
uv run pytest tests/ -x -q
```

Branch protection is on — all changes go through PRs with CI checks.

## Key design decisions

- **Smart defaults**: `ocr-bench judge <repo>` needs zero flags (auto-detect configs, auto-derive results repo, adaptive stopping on)
- **Arrow-level merges**: dataset loading uses Arrow column ops to avoid per-row image decode
- **Don't merge PRs**: load OCR outputs via `revision=` to avoid README merge conflicts on Hub datasets
- **Default judge**: Qwen3.5-35B-A3B via HF Inference Providers (zero parse failures, fastest, only needs HF token)

## Known limitations

- **Row alignment across configs is positional only** — `load_config_dataset()` merges by index. Safe if all model runs use the same `--seed`/`--max-samples` and source dataset doesn't change. Future: add content hash column.
- **Blank page filtering** not yet implemented — wastes judge calls when neither model produced meaningful text.

## Roadmap

- Blog post: "There Is No Best OCR Model"
- Judge prompt presets for GLAM document types
- Custom prompt and ignore list support
- Judge comparison across different judge models
- `--focus-pairs`: prioritize overlapping-CI pairs in validation UI
- CER/WER metrics alongside VLM judge
- `bench` command: single `ocr-bench bench <input-dataset>` chains run → judge → view
