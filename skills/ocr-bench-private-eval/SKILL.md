---
name: ocr-bench-private-eval
description: Build a private OCR leaderboard with ocr-bench on HF Jobs - a user's own pages in a private dataset, OCR outputs from model recipes or brought by the user, a VLM judge run as a Job, and an optional scores-only public board. Use when a user wants to know which OCR model works best on their documents without publishing the documents.
---

# Private OCR eval with ocr-bench

The user wants to rank OCR models on their own pages. The pages, the OCR outputs and the judge's reasons stay in private Hub repos. Only `ocr-bench share` publishes anything, and it publishes scores and judge settings only.

The full human guide is [docs/private-eval.md](../../docs/private-eval.md). Read it once. This file gives the order, the checks and the points where you must stop and ask.

## Stop and ask the user before

1. **Each paid step.** OCR Jobs (`run`) and the judge Job. Give the estimate first:
   - OCR: `ocr-bench run ... --dry-run` shows each model's flavor; price from `hf jobs hardware`.
   - Judge: about $0.0014 per comparison (default judge, 8,000-character cap). Upper bound: pages × models × (models − 1) / 2; adaptive stopping usually uses fewer. Use `--max-comparisons` as a hard limit; a capped run continues later with the same command.
2. **Anything public.** `share` creates a missing board repo private; making it public (the repo's Settings page, `hf repos settings <board-repo> --repo-type dataset --public` with `hf` 2.0 or later, or `share --public`) needs the user's OK. Show the `--dry-run` files first.
3. **Where the pages go.** The default judge (`novita:Qwen/Qwen3.8-27B`) sends each page image to Novita through HF Inference Providers. Say so, and ask whether that is acceptable. If not, the user chooses an endpoint that meets their needs and you pass it with `--model`: an HF Inference Endpoint or a Jobs-served model (the HF token is sent), or an OpenAI-compatible server that needs no API key.

Never change the visibility of a source, OCR output or results repo. Never publish the OCR output repo or the results repo: they contain page text and judge reasons that quote the page.

## Steps

1. **Pages.** A private dataset, one row per page: `image` (required), `page_id` (recommended), and optional slice columns (`collection`, `decade`, `doc_type`). The slice column name and its labels are published by `share` as they are: check with the user that they may be public. At most 50 labels per slice column. Aim for 300–500 pages, one or a few per document, more than 10 pages per slice the user cares about (a model needs 10 judged pages in a slice to be ranked there), and a few labelled blank pages.
2. **OCR outputs.** Either:
   - **Recipes on Jobs:** `ocr-bench run <source> <ocr-repo> --models ... --dry-run`, then the same with `--no-wait`. Models: `ocr-bench run --list-models`. Each model becomes a PR config on `<ocr-repo>`. Wait until every Job has finished (`hf jobs ps`, `hf jobs logs <id>`) before the audit. Use `--from-prs` in later steps.
   - **Bring your own:** one dataset with `image` and one text column per model. The column name is the model's name, and `share` publishes it: check it may be public. `"[OCR ERROR]"` marks a failed page. Use `--columns <names>` in later steps.
3. **Audit (free).** `ocr-bench audit <ocr-repo> --from-prs --screen --json` (or `--columns ...`). Report to the user: models with many empty outputs, error placeholders, loops, text on blank pages, and the suggested text cap. Exit code 1 means the outputs would break a judge run; 2 means the audit could not run.
4. **Judge, as a Job** (after the cost OK):

   ```bash
   hf jobs run --detach --timeout 3h --flavor cpu-upgrade -s HF_TOKEN \
     ghcr.io/astral-sh/uv:python3.12-bookworm \
     uvx "ocr-bench==<version>" judge <ocr-repo> --from-prs \
       --criteria body-text --max-ocr-text-len <cap from audit> \
       --max-comparisons <budget> --save-results <results-repo> --private
   ```

   - The user must be logged in (`hf auth login` or `HF_TOKEN`); `-s HF_TOKEN` passes that token to the Job.
   - Always set `--timeout`: a Job stops after 30 minutes by default, and judging runs at about 85 comparisons a minute with `--concurrency 12`.
   - Follow with `hf jobs logs <id>`. Check the end state with `hf jobs inspect <id>`: a timeout shows as `ERROR`, `Job timeout`.
   - If it stopped early, run the same command again: it reuses the checkpoints (every 500 comparisons). The leaderboard is printed at the end.
   - To add a model later, get its outputs and run the same command: only the new pairs are judged.
   - If `judge` says the inputs changed (for example, one model was re-run), ask the user before `--full-rejudge` or a new results repo: either costs a full run.
5. **Report.** The leaderboard with confidence intervals, the Pages column (a model ranked on fewer pages has failed pages), and which models overlap. Do not call models with overlapping intervals different. Suggest `ocr-bench view <results-repo>` (local, needs `ocr-bench[viewer]` and a login that can read the results and the source) to read comparisons and vote.
6. **Share (only if the user wants a public board).** `ocr-bench share <results-repo> <board-repo> --alias <public name> [--slice-column <col>] --dry-run`. Show the user the files in `./ocr-bench-share-<alias>`. (`share` reads the results repo, and with `--slice-column` the source dataset too, so the token needs read access to both. It refuses a board whose models cannot all be ranked against each other, and a results repo whose last judge run did not finish; slice boards need page images in the source.) Then run the same command without `--dry-run`: the board repo is created private. Give the user its URL to check. Make it public only after they agree, as `share` prints: the repo's Settings page, or `hf repos settings ... --public` (`hf` 2.0 or later; check with `hf version`). Use a new or empty board repo.

## Facts to keep straight

- Private source → `run` and `judge` create private output and results repos, and stop before writing to an existing public repo.
- `judge` visits the sample in a random order, so a budget stop judges a random subset of pages.
- Boards with different judges, or the same judge model on different providers, are not comparable.
- ocr-bench measures transcription quality on these pages. Do not report speed or throughput rankings from it.
- ocr-bench is a proof of concept (before 0.1.0). If a flag in this file fails, check `ocr-bench <command> --help`.
