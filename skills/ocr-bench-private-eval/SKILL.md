---
name: ocr-bench-private-eval
description: Build a private OCR leaderboard with ocr-bench on HF Jobs - a user's own pages in a private dataset, OCR outputs from model recipes or brought by the user, a VLM judge run as a Job, and an optional scores-only public board. Use when a user wants to know which OCR model works best on their documents without publishing the documents.
---

# Private OCR eval with ocr-bench

The user wants to rank OCR models on their own pages. The pages, the OCR outputs and the judge's reasons stay in private Hub repos. Only `ocr-bench share` publishes anything, and it publishes scores only.

The full human guide is [docs/private-eval.md](../../docs/private-eval.md). Read it once. This file gives the order, the checks and the points where you must stop and ask.

## Stop and ask the user before

1. **Each paid step.** OCR Jobs (`run`) and the judge Job. Give the estimate first:
   - OCR: `ocr-bench run ... --dry-run` shows each model's flavor; price from `hf jobs hardware`.
   - Judge: about $0.0013 per comparison (default judge, 8,000-character cap). Upper bound without adaptive stopping: pages × models × (models − 1) / 2. Use `--max-comparisons` as a hard limit.
2. **Anything public.** `share` without `--private` creates a public repo. Show the `--dry-run` files first.
3. **Where the pages go.** The default judge (`novita:Qwen/Qwen3.8-27B`) sends each page image to Novita through HF Inference Providers. Say so, and ask whether that is acceptable. If not, the user chooses an endpoint that meets their needs and you pass it with `--model`.

Never change the visibility of a source, OCR output or results repo. Never publish the OCR output repo or the results repo: they contain page text and judge reasons that quote the page.

## Steps

1. **Pages.** A private dataset, one row per page: `image` (required), `page_id` (recommended), and optional slice columns (`collection`, `decade`, `doc_type`). Slice labels are published by `share` as they are: check with the user that they may be public. Aim for 300–500 pages, one or a few per document, 10+ pages per slice the user cares about, and a few labelled blank pages.
2. **OCR outputs.** Either:
   - **Recipes on Jobs:** `ocr-bench run <source> <ocr-repo> --models ... --dry-run`, then the same with `--no-wait`. Models: `ocr-bench run --list-models`. Each model becomes a PR config on `<ocr-repo>`. Use `--from-prs` in later steps.
   - **Bring your own:** one dataset with `image` and one text column per model (the column name is the model's name). `"[OCR ERROR]"` marks a failed page. Use `--columns <names>` in later steps.
3. **Audit (free).** `ocr-bench audit <ocr-repo> --from-prs --screen --json` (or `--columns ...`). Report to the user: models with many empty outputs, error placeholders, loops, text on blank pages, and the suggested text cap. Exit code 1 means the outputs would break a judge run; 2 means the audit could not run.
4. **Judge, as a Job** (after the cost OK):

   ```bash
   hf jobs run --detach --flavor cpu-upgrade -s HF_TOKEN \
     ghcr.io/astral-sh/uv:python3.12-bookworm \
     uvx "ocr-bench==<version>" judge <ocr-repo> --from-prs \
       --criteria body-text --max-ocr-text-len <cap from audit> \
       --max-comparisons <budget> --save-results <results-repo> --private
   ```

   Follow with `hf jobs logs <id>`. The leaderboard is printed at the end. To add a model later, get its outputs and run the same command: only the new pairs are judged. If `judge` says the inputs changed, ask the user before `--full-rejudge` (it costs a full run).
5. **Report.** The leaderboard with confidence intervals, the Pages column (a model ranked on fewer pages has failed pages), and which models overlap. Do not call models with overlapping intervals different. Suggest `ocr-bench view <results-repo>` (local, needs `ocr-bench[viewer]`) to read comparisons and vote.
6. **Share (only if the user wants a public board).** `ocr-bench share <results-repo> <board-repo> --alias <public name> [--slice-column <col>] --dry-run`. Show the user the files in `./ocr-bench-share-<alias>`. Publish only after they agree, with the same command without `--dry-run`.

## Facts to keep straight

- Private source → `run` and `judge` create private output and results repos, and stop before writing to an existing public repo.
- `judge` visits the sample in a random order, so a budget stop judges a random subset of pages.
- Boards with different judges, or the same judge model on different providers, are not comparable.
- ocr-bench measures transcription quality on these pages. Do not report speed or throughput rankings from it.
- ocr-bench is a proof of concept (before 0.1.0). If a flag in this file fails, check `ocr-bench <command> --help`.
