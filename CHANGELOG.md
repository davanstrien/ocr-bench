# Changelog

Notable changes to ocr-bench. Each release has a section headed `## [X.Y.Z] - YYYY-MM-DD`.
The release workflow uses that section as the GitHub release notes; see [RELEASING.md](RELEASING.md).

ocr-bench is a proof of concept. Before 0.1.0, any release can change the CLI, the defaults
or the results format. Pin an exact version (for example `ocr-bench==0.0.5`) if you depend on
current behaviour.

## [Unreleased]

### Other

- The README example boards are re-judged with 0.0.5: the default judge, page-level intervals and `--max-ocr-text-len 8000`. dots.ocr's Britannica row comes from a working re-run.

## [0.0.5] - 2026-11-03

### Upgrading from 0.0.4

- `uv pip install -U ocr-bench`. This needs `huggingface-hub>=1.0`.
- **Results repos from 0.0.4 cannot be resumed.** Their comparisons carry no resume provenance (#57), and the new default judge (#68) and text normalisation (#48) change it anyway. Run `ocr-bench judge <dataset> --full-rejudge`, which costs a full judge run again, or write to a new results repo with `--save-results`. `--full-rejudge` is a `judge` option; `bench` has none.
- **New default judge: `novita:Qwen/Qwen3.8-27B`** (#68). It's an estimated 1.6× the cost per comparison at list prices. It was chosen for consistency, with lower order bias and correct ties on formatting-only differences. This is not a claim of better accuracy. Boards judged with the old and new defaults are not comparable. A provenance-stamped board judged with the old default can resume with `--model novita:Qwen/Qwen3.5-35B-A3B` (`--judge-model` with `bench`) if all other settings match; the error names the earlier judge.
- **Private or gated sources stay private** (#67). New output and results repos are created private. `run`, `judge` and `bench` stop before writing to an existing public repo unless you pass `--public`, and `--private` requires private repos for any source. Other new repos keep the account or organisation default. `publish` creates a private viewer Space for private results, which needs an `HF_TOKEN` secret with read access to them.

### Behaviour changes

- **`run` applies each recipe's `[tool.hf-jobs]` header** (Docker image, Python, env), as `hf jobs uv run` does (#69). Many uv-scripts OCR recipes moved vLLM into that header in September, so `run` had been starting them without vLLM. Of the default models:
  - glm-ocr and dots-ocr are smoke-tested with the current recipes, which include the uv-scripts#167 fixes;
  - deepseek-ocr still uses the v0.29.0 image, which has a known import error;
  - lighton-ocr-2 is likely affected the same way (unverified);
  - firered-ocr is marked legacy upstream and untested here.
- **Resumes fail closed** when the source, configs, judge, prompt or preprocessing changed (#57).
- **Invalid judge verdicts fail** instead of being counted (#61).
- **Error-placeholder outputs** such as `[OCR ERROR]` are excluded from judging. An all-placeholder run is marked `FAILED` with no rank (#49).
- **HTML is normalised before the per-output text cap** (#48). Both caps are configurable with `--max-ocr-text-len` and `--judge-image-dim`, and `--judge-text-mode raw` skips normalisation.
- **Confidence intervals resample whole pages** (#62). A model's rank counts as statistically resolved only once it rests on at least 10 distinct pages.
- **Pairs where both outputs are shorter than `--min-chars`** (default 20; `0` disables) are skipped. Identical outputs auto-tie without a judge call (#42).
- **The viewer reports exact human agreement** and no longer claims judge reliability (#63).

### New

- `ocr-bench bench`: run, judge and view in one command (#42).
- `ocr-bench audit`: a read-only check of OCR outputs before judging. It fails on configs that can't be judged (#49, #64).
- `--criteria` judge prompt profiles and `--criteria-file` custom prompts, with the prompt hash recorded (#47).
- `--max-comparisons` budget cap and checkpoint publishing (#41).
- `judge --from-prs --configs a b` judges only the listed configs, each from its newest open PR or from main. Results metadata records which revision each config came from, and the viewer loads page images from those configs. When several open PRs carry the same config, the newest is used and the others are named (#71).
- Opt-in `--adaptive-strategy targeted` and `--size-tie-ratio` (#53, #54).
- New models: tesseract, PP-OCRv6 and three VLMs (#34); Falcon OCR and OvisOCR2 (#52).
- `ocr_bench.__version__` comes from the package metadata (#70).

### Fixes

- Source splits are honoured throughout (#56).
- Inference metadata and OCR columns stay in sync (#55).
- Row alignment across configs is checked by key, or by image bytes when keys can't decide. It stays explicitly unverified when neither can decide (#49, #65).
- Provider concurrency is honoured. Judge URLs on HF Jobs authenticate with the HF token, and the token is forwarded only to HTTPS HF Jobs and Inference Endpoints hosts (#61).
- Stale viewer Spaces are fixed, and dataset cross-links added (#40).
- README example boards are corrected (#66).

### Other

- Merging a version-bump PR now releases: the workflow tests, publishes to PyPI, tags the commit and creates the GitHub release from this file (#70).
- A reproducible Kraken olmOCR-bench experiment and its results, under `experiments/` (#59).

## [0.0.4] and earlier

No changelog entries. The releases are tagged `v0.0.1` to `v0.0.4`.
