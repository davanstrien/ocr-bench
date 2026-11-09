# Changelog

Notable changes to ocr-bench. Each release has a section headed `## [X.Y.Z] - YYYY-MM-DD`.
The release workflow uses that section as the GitHub release notes; see [RELEASING.md](RELEASING.md).

ocr-bench is a proof of concept. Before 0.1.0, any release can change the CLI, the defaults
or the results format. Pin an exact version (for example `ocr-bench==0.0.5`) if you depend on
current behaviour.

## [Unreleased]

### Behaviour changes

- `judge` visits the judged sample in a seeded random order. Before, without `--max-samples` below the dataset length, it visited the rows in dataset order, so a `--max-comparisons` budget or adaptive stopping on a sorted dataset (by collection, decade or document) judged only its first rows. When a budget or adaptive stopping ends a run early, the judged rows are now a random subset of the sample. The sample itself does not change, so the page fingerprints, resume and `share` work as before on existing results repos.
  - A results repo whose earlier run judged only the first rows keeps those comparisons. The random order applies to the rows judged from now on.
  - The order uses its own random stream, derived from `--seed`. Reusing the stream that draws the sample would bias which pages come first.
  - The sample uses a private random generator. It used to seed Python's global random generator; the sample is the same.
  - Which output is shown to the judge as A and which as B is still drawn at random, but in the new visiting order, so for a given seed it differs from earlier runs.

- `audit --screen` also counts a counting list (`- 1`, `- 2`, `- 3`, …) as a loop: the last 2,000 characters of an output of 500 or more characters hold 50 or more numbers in a row that each count up by one, with the same text between each number and the next. Such a list does not compress, because each line has a new number, so the compression test missed it. In `--json`, these rows are listed under `loop.counting_rows`, and the thresholds include `counting_run`. Peer agreement leaves out these loops, as it does the others.
  - The same-text condition keeps real numbered lists with different entries (a membership roll, a coin catalogue, numbered verse, a list of plates) from being flagged. A real list of 50 or more numbered entries with the same text in each (`1 Ditto`, `2 Ditto`, …, or folio numbers with the same word between them) is still flagged; check such rows in `loop.counting_rows`.
  - On the public Britannica board (6 models, 50 pages), the screen now finds 21 loops: the 20 it found before, and DeepSeek-OCR's 10,490-character list on row 0, which only `>5x length` flagged before.
  - On 2,230 outputs from seven public ocr-bench collections, the test flags 15 more outputs. All 15 are counting lists. No real transcription had a run longer than 23 numbers.
  - Replacing every digit with 0 before the compression test also finds these lists, but it flagged 16 numeric tables on the same outputs, so it is not used.

### New

- `share` publishes a scores-only board from a results repo to a separate dataset repo: `ocr-bench share <results-repo> <board-repo> --alias <name>`. The board has the overall ranking, one board per slice with `--slice-column`, and the judge settings. It has no page text, OCR outputs, judge reasons, page ids or repo ids. Before any Hub call, `share` checks the final files: allowed columns and values only, a README rendered from them, no repo ids, and no 32-character chunk of a compared output or judge reason. If a check fails, nothing is written. Slice labels are read only after the page fingerprint of the judged source config matches. A missing destination is created private, and `share` prints how to publish it (the repo's Settings page, or `hf repos settings ... --public` with the `hf` CLI 2.0 or later); `--public` creates it public in one step. An existing destination keeps its visibility. `--dry-run` writes the files locally. See the README.
- `audit --columns` reads the named text columns of a flat dataset (one column per model, named after the model), as `judge --columns` does. Before, `audit` found flat columns only from an `inference_info` column or from column names that contain `markdown`, `ocr` or `text`, so it could miss bring-your-own outputs. `--configs` takes precedence, as in `judge`.
- `judge` and `bench` report on how many pages each model is ranked. The leaderboard has a Pages column, N/M: M is the number of pages in the judged sample (`--max-samples`, or every row), and N is the number of those pages where the model took part in at least one judged or auto-tied comparison. A footnote names each model with error sentinels (placeholders such as `[OCR ERROR]`) in the sample. The results card has a Coverage section, and each new `metadata` row has `model_coverage` (JSON: `sampled_pages`, `ranked_pages` and `sentinel_pages` per model).
- `judge` prints a note when a model has error sentinels at a rate of 10% or less: they are excluded from judging, so its rank leaves out those pages. Before, only rates over 10% were reported. The threshold and the exclusion do not change.
- Before judging, `judge` and `bench` warn when `--max-ocr-text-len` truncates more than half of the outputs in the judged sample, overall or for any one model. The warning suggests a cap from the `audit --screen` checks (a cap that fits every output that no check flags), when 3 or more models have output. The default cap does not change.

These are reports only. Prompt hashes, comparison fingerprints and resume provenance do not change, so results repos from 0.0.6 can resume.

### Fixes

- An OCR output cell that is not a string (a number, a NaN, a list) is read as an empty output by `judge`, `audit` and `audit --screen`. Before, such a cell crashed them. The cell is not converted with `str()`, so the judge never sees text such as `nan`. In the audit, the cell counts as empty, and a column with no string values has no judgeable rows.
- A flat dataset with no rows stops with an error that says so. Before, it raised `IndexError`.
- `audit` and `judge` check that the dataset repo exists before they look for configs. For a missing repo, or a private repo that the token cannot read, the error is now "Dataset repo '<id>' was not found on the Hub (repo type: dataset)". Before, the error named the discussions endpoint. `audit` still exits with 2.
- The "no HF token" error of `run` says `hf auth login`, not `hf login`.

### Removed

ocr-bench is experimental, so these go without a transition release.

- `judge` does not read old metadata rows to explain a criteria or preprocessing change any more. If a comparison it would reuse (one between two models of this run) has no resume provenance hash, the run stops with the "lack resume provenance" error. Such rows come from older ocr-bench versions or an incomplete old checkpoint. Run `--full-rejudge` once. A change of criteria, criteria file, text cap, text mode or image size on stamped comparisons still stops with "resume provenance mismatch".
- `judge` does not drop old comparisons that hold an OCR error placeholder any more. The judge never compares a placeholder, so only rows from older versions can hold one, and those rows have no provenance hash.
- Results repos do not get a duplicate `leaderboard` config any more, and the README does not declare one. Load the default config: `load_dataset("<repo>")`. Existing repos keep their old `leaderboard/` folder, which stops updating. The viewer reads the default config only.
- The `PAIRWISE_PROMPT` alias and three unused viewer helpers are gone. Use `CRITERIA_PROFILES[DEFAULT_CRITERIA]`.

### Other

- Code that pairs lists that must have the same length now uses `zip(..., strict=True)`, so a mismatch raises an error instead of dropping rows silently. This covers judge results and comparisons, jury votes, fingerprints and alignment columns.
- Development: `ty` is pinned to 0.0.97 and ruff to 0.15.4. ruff also checks the `B` rules. `src/` and `tests/` are formatted with `ruff format`. CI checks the format. CI also checks that `uv.lock` resolves from https://pypi.org/simple only and records no `exclude-newer` setting.

## [0.0.6] - 2026-11-04

### Upgrading from 0.0.5

- `uv pip install -U ocr-bench`. Results repos from 0.0.5 can resume.
- A resume stops with a provenance mismatch if `judge` now loads a config from `main` instead of the old open PR it used before (see below). Those outputs changed, so the earlier verdicts do not apply. Use `--full-rejudge`.

### Behaviour changes

- `judge --concurrency` defaults to 8, not 1, so a run makes up to 8 judge calls at once. Pass `--concurrency 1` for sequential calls. A value below 1 is now an error. `bench` has a new `--concurrency` option with the same default.
- With PR discovery, a config that changed on `main` after its newest open PR is loaded from `main`. Before, an old open PR (for example a failed run left open after a re-run was merged) replaced the newer outputs on `main`. With `--merge`, such a PR is not merged. `judge`, `bench` and `audit` print the PRs they passed over; to use such a PR, merge it. The check compares the dates of the last commit to the `<config>/` folder on each branch. A merge commit has the merge date, so a PR merged later counts as newer even if its run is older. When a date is missing, the PR is used.

### New

- `--criteria body-text`: the default criteria, except for completeness. Completeness covers the main text, headings, footnotes and captions. Running headers, page numbers, signature marks and catchwords are optional, and the prompt tells the judge not to penalise their absence or reward their presence. Only the completeness criterion changes, so the accuracy and tie rules still see all the text. The prompt does not name other text, such as marginalia or stamps, so the judge decides. Use it when the text will be read or searched, or to compare a board with a ground truth that leaves page furniture out.
- `audit --screen`: free checks to run before a judge run. Each model's output on a page is compared with the other models' outputs on that page. The screen counts:
  - loops: the last 2,000 characters compress to less than 15% of their size, or one non-space character repeats 200 or more times;
  - outputs more than 5× the median length of the others;
  - text (100 or more characters) on blank pages, where most outputs have under 20 characters;
  - empty output (under 20 characters) on text pages, where most outputs have 200 or more.

  It also gives each model a peer-agreement score: the mean, over pages, of the median normalised Levenshtein similarity to the others, with loops left out. It prints the length distribution of the outputs that no check flagged, and suggests a `--max-ocr-text-len` that fits them all. The screen needs 3 or more models whose outputs are not all empty or error placeholders. It does not change the exit code.
- `audit --configs` and `audit --from-prs` select configs as `judge` does.
- `audit --json` prints the report as JSON: the ocr-bench version, the screen thresholds, and each flagged output with its row, checks and lengths. The exit codes are unchanged. An audit that cannot complete (a Hub, network or dataset-load error) prints `{"error": ...}` and exits with 2.

### Other

- New dependency: `rapidfuzz`, for the peer-agreement score.
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
