# Build a private OCR eval

This guide shows how to rank OCR models on your own pages without publishing the pages. The pages, the OCR outputs and the judge's reasons stay in private Hub repos. When you want to show the result, `ocr-bench share` publishes the scores and the judge settings only.

Everything runs on the Hugging Face Hub: the OCR models and the judge run as [Jobs](https://huggingface.co/docs/hub/jobs-overview), and the data lives in private dataset repos. You need an [HF token](https://huggingface.co/settings/tokens) with write access, and a plan that includes Jobs (PRO, Team or Enterprise).

> [!NOTE]
> ocr-bench is a proof of concept. Before 0.1.0, the CLI, the defaults and the results format can change. The way ocr-bench gets OCR outputs (step 2) is the part most likely to change. `share` needs ocr-bench 0.0.7 or later.

## Before you start

- Install the CLI with `uv tool install ocr-bench` (or run a single command with `uvx ocr-bench ...`). Use the latest release from [PyPI](https://pypi.org/project/ocr-bench/), and the same version in the Job commands below.
- Jobs guides: [overview](https://huggingface.co/docs/hub/jobs-overview), [configuration](https://huggingface.co/docs/hub/jobs-configuration) (hardware flavors, timeout), [managing Jobs](https://huggingface.co/docs/hub/jobs-manage) (logs, cancel) and [pricing](https://huggingface.co/docs/hub/jobs-pricing). `hf jobs --help` lists the CLI commands; the [`hf` CLI guide](https://huggingface.co/docs/huggingface_hub/guides/cli) covers the rest.
- Use full repo ids (`<you>/my-pages`, never a bare `my-pages`) in every command.
- Working with an AI agent? `hf skills add` installs a skill that teaches it the `hf` CLI, and this repo's [agent skill](../skills/ocr-bench-private-eval/SKILL.md) gives it these steps with stop points.

## The steps

| Step                                   | Command                            | Cost             |
| -------------------------------------- | ---------------------------------- | ---------------- |
| 1. Put your pages in a private dataset | your own script (see below)        | free             |
| 2. Get OCR outputs                     | `ocr-bench run`, or bring your own | GPU time on Jobs |
| 3. Check the outputs                   | `ocr-bench audit --screen`         | free             |
| 4. Judge                               | `ocr-bench judge`, as a Job        | judge API calls  |
| 5. Look at the results                 | `ocr-bench view`                   | free             |
| 6. Publish the scores                  | `ocr-bench share`                  | free             |

In the 425-page proof of concept, 8 OCR models ran and 7 were judged. The OCR and judge Jobs used about $4.50 of Jobs time, and the judge calls (about 6,400) cost about $9.30.

Log in first (`hf auth login`, or export `HF_TOKEN`): `-s HF_TOKEN` in the Job commands below passes that token to the Job.

## Who sees your pages

Decide this first.

| Where                       | What it sees                                               | Visibility                         |
| --------------------------- | ---------------------------------------------------------- | ---------------------------------- |
| Your source dataset         | page images and your metadata                              | private, if you create it private  |
| The OCR output repo (`run`) | images and OCR text                                        | private when the source is private |
| The results repo (`judge`)  | OCR text and the judge's reasons, which can quote the page | private when the source is private |
| HF Jobs                     | everything the Job reads                                   | runs under your account            |
| The judge model             | each page image and two transcriptions                     | depends on the judge you choose    |
| The board repo (`share`)    | ratings, settings and a fingerprint of the page set        | private until you publish it       |

The default judge is `novita:Qwen/Qwen3.8-27B` through [HF Inference Providers](https://huggingface.co/docs/inference-providers/index), so Novita processes the page images.

**If you need to control where your pages are sent, pass `--model` with an endpoint that meets your privacy needs.** It can be a [Hugging Face Inference Endpoint](https://huggingface.co/docs/inference-endpoints) URL, or a model you serve yourself on a Job ([Serve Models on Jobs](https://huggingface.co/docs/hub/jobs-serving)), passed as `--model https://<job-id>--8000.hf.jobs/v1/:<org>/<model>`; ocr-bench sends your HF token to both. Only the default judge has been checked against ground truth, so look at some of another judge's verdicts in `view` before you trust its board. It can also be an OpenAI-compatible server that needs no API key (`https://host/v1/:org/model`).

Boards judged by different judges, or by the same model on different providers, are not comparable.

`run`, `judge` and `bench` never make a private repo public, and they stop before writing to an existing public repo for a private source. Do not change the visibility of an output or results repo yourself: they contain page text.

## 1. Put your pages in a private dataset

ocr-bench reads one Hub dataset with one row per page:

| Column        | Required    | What it is                                                           |
| ------------- | ----------- | -------------------------------------------------------------------- |
| `image`       | yes         | the page image                                                       |
| `page_id`     | recommended | a stable, unique id for the page, for example `collection/item:page` |
| slice columns | optional    | labels for reporting, for example `collection`, `decade`, `doc_type` |

Slice columns let `share` publish a board per label. **`share` publishes the column name and its labels as they are**, so use a column name and labels that may be public: `doc_type` with `typescript` or `1950s`, not a person's name, a client's name or a page's title. A slice column can have at most 50 different labels on the judged pages, so do not use a per-page or per-document column.

A simple way to build it: put the images in a folder with a `metadata.csv` and push it as a private dataset. The CSV needs a `file_name` column with each image's file name (relative to the folder), then your own columns, for example `file_name,page_id,collection,decade`. TIFF, PNG and JPEG all work; split multi-page TIFFs or PDFs into one image per page first.

```python
from datasets import load_dataset

ds = load_dataset("imagefolder", data_dir="my-pages", split="train")
print(ds)  # check: an `image` column, your columns, one row per page
ds.push_to_hub("<you>/my-pages", private=True)
```

### Choosing pages

- **Size.** 300–500 pages is enough for a first board. In the proof of concept, with 278 judged text pages, the top model's interval did not overlap the next model's.
- **One page per document**, or a few. Many pages from one document make the board about that document.
- **Slices.** `share` ranks a model in a slice only when the model has 10 or more judged pages in it. A failed page (`[OCR ERROR]`) does not count, so plan more than 10 pages for each slice you care about.
- **Blank pages.** Add some known-blank pages on purpose, and label them (for example `page_type = blank`). Some models write text on blank pages: a description of the image, a stock phrase or another page's text. `audit --screen` counts this.
- **Order does not matter.** `judge` visits the sample in a random order, so a sorted dataset is fine.
- **Choose the pages up front.** Adding or replacing pages later means running every model again and a full re-judge: the judged pages are fingerprinted, and each model's output is matched to the pages by row. Adding a model later is cheap (see step 4).

## 2. Get OCR outputs

There are two ways. Use the first to try known models, and the second for anything else.

### A. Run model recipes on Jobs

`ocr-bench run` launches OCR models from [uv-scripts/ocr](https://huggingface.co/datasets/uv-scripts/ocr) on HF Jobs. Each model writes its output as a pull request on an OCR output repo, one config per model.

```bash
ocr-bench run --list-models
ocr-bench run <you>/my-pages <you>/my-pages-ocr --models glm-ocr lighton-ocr-2 dots-ocr --dry-run
ocr-bench run <you>/my-pages <you>/my-pages-ocr --models glm-ocr lighton-ocr-2 dots-ocr --no-wait
```

Without `--models`, `run` uses the default set listed in the [README](../README.md#available-models). With `--no-wait`, `run` prints one Job per model and exits. In the proof of concept (425 pages), each model took between about 5 and 40 minutes. Check them with `hf jobs ps` and `hf jobs logs <job-id>`. Each finished model appears as a pull request on the OCR output repo. Wait for all of them before step 3.

`--dry-run` shows the GPU flavor and the arguments for each model. The cost is the flavor's price (`hf jobs hardware`) times the run time. The output repo is created private when the source is private.

> [!NOTE]
> The list of models in `run` is changing: it will follow the maintained recipes in uv-scripts and focus on newer models. Bring-your-own outputs (route B) will stay.

### B. Bring your own outputs

If you already have transcriptions (from a vendor, another tool, a model that `run` does not have, or the text that came with your scans), put them in one dataset: the `image` column, your other columns, and one text column per model. The column name is the model's name on the board, and `share` publishes it: do not use a contract name or a project code name.

```python
ds = ds.add_column("vendor-ocr", vendor_texts)        # same row order as ds
ds = ds.add_column("my-finetune", finetune_texts)
ds.push_to_hub("<you>/my-pages-byo", private=True)
```

Use `"[OCR ERROR]"` for a page where a model failed. ocr-bench leaves such pages out of that model's comparisons and reports them as coverage. An empty string counts as "this model found no text".

Then pass the columns to `audit` and `judge` with `--columns`, in place of `--from-prs`.

## 3. Check the outputs (free)

```bash
ocr-bench audit <you>/my-pages-ocr --from-prs --screen
```

`audit` reads the outputs and makes no judge calls. It reports empty outputs, error placeholders and output lengths. With `--screen` and 3 or more models, it also finds repetition loops, outputs much longer than the other models' outputs, text on pages that most models leave blank, and empty output on pages where most models have text. It suggests a text cap (`--max-ocr-text-len`) for the judge. Add `--json` for a machine-readable report.

Fix or drop a model with many failures before you pay for the judge.

## 4. Judge, as a Job

The judge compares two transcriptions of the same page at a time and picks the better one. A board of 8 models on 400 pages is several thousand comparisons, so run it as a Job, not on your laptop:

```bash
hf jobs run --detach --timeout 3h --flavor cpu-upgrade -s HF_TOKEN \
  ghcr.io/astral-sh/uv:python3.12-bookworm \
  uvx "ocr-bench==0.0.7" judge <you>/my-pages-ocr --from-prs \
    --criteria body-text --max-ocr-text-len 8000 \
    --save-results <you>/my-pages-results --private
```

For a private source, the results repo is private automatically. Pass `--private` when the source is public and the results must stay private.

**Set `--timeout`.** A Job stops after 30 minutes by default ([Job timeout](https://huggingface.co/docs/hub/jobs-configuration#timeout)), and a few thousand comparisons take longer: the proof of concept judged about 85 a minute with `--concurrency 12` (the default is 8). `judge` pushes a checkpoint every 500 comparisons (not under `--full-rejudge`), so if a Job stops early, run the same command again: it reuses the checkpointed comparisons and judges the rest. Comparisons made after the last checkpoint are lost and paid for again.

- **Cost.** About $0.0014 per comparison with the default judge, an 8,000-character cap and 1,024 px images (billed in the proof of concept). A run is at most pages × models × (models − 1) / 2 comparisons; adaptive stopping usually uses fewer. `--max-comparisons N` sets a hard limit: your budget divided by $0.0014, for example about 7,000 comparisons for $10. A run stopped by this limit can be continued later with the same command and a higher limit.
- **Text cap.** `--max-ocr-text-len` is 2,500 characters by default; longer outputs are cut before the judge sees them. Use the cap that `audit --screen` suggests (the proof of concept used 8,000 for dense pages). Choose it before the first run: changing it later needs `--full-rejudge`.
- **`--criteria body-text`** asks the judge to compare the body text and to ignore page furniture (running heads, page numbers). For table-dense material, `--criteria table-fidelity` adds a row-and-column check. One board uses one criterion.
- **`--max-samples N`** judges a random sample of N pages.
- **Adding a model later:** run the OCR for the new model, then run the same `judge` command again. Only the new pairs are judged. If an input changed (a re-run model, other pages, another judge setting), `judge` stops and tells you to pass `--full-rejudge` or use a new results repo. A re-run of one model counts as a change, so the whole board is judged again.
- **Logs:** `hf jobs logs <job-id>`. The leaderboard is printed at the end and written to the results repo.

## 5. Look at the results

```bash
uv tool install "ocr-bench[viewer]"
ocr-bench view <you>/my-pages-results
```

The viewer runs on your machine. It reads the private results repo and the private source images with the token you are logged in with (`hf auth login` or `HF_TOKEN`), so that token needs read access to both. It shows the leaderboard and each comparison: the page, both transcriptions and the judge's reason. Vote on some comparisons to check that the judge agrees with you.

## 6. Publish the scores

```bash
ocr-bench share <you>/my-pages-results <you>/my-ocr-board \
  --alias "gov-typescripts-v1" --slice-column decade --dry-run
```

`--dry-run` writes the board to a local folder (`./ocr-bench-share-<alias>`) and pushes nothing. Read the files. When they are right, run the same command without `--dry-run`. This creates the board repo private; look at it on the Hub, then publish it (below). `share` reads the results repo, and with `--slice-column` also the source dataset (for the slice labels and the page fingerprint), so your token needs read access to both.

- The board has the overall ranking, one board per slice label, the judge settings and a fingerprint of the page set. It has no page text, OCR outputs, judge reasons, page ids, or the names of your repos. `--alias` is the public name of your page set.
- Before it writes anything, `share` checks the final files: only allowed columns and values, and no copied fragment of 63 or more characters from any compared output or judge reason. Shorter fragments, text with little variety (a 32-character chunk with fewer than 4 different letters or digits is skipped), and the alias, slice labels and model names are not checked: read them in the dry run. If a check fails, nothing is written.
- Every model on a board must be rankable against every other. If two groups of models were never compared, or a model won (or lost) every comparison, the ratings would be meaningless: `share` refuses such an overall board and leaves such a slice out.
- Slice boards need page images in the source: `share` checks that the labels are on the judged pages by their image bytes.
- If the last `judge` run did not finish publishing (for example, it stopped after a checkpoint), `share` refuses until you run `judge` again.
- Use a new or empty repo for the board. `share` refuses a repo with files that it did not write, and a repo with the board of another alias unless you pass `--replace`.
- A missing board repo is created private, so you can look at the board on the Hub first. To publish it, use **Settings → Change visibility** on the repo page, or, with the `hf` CLI 2.0 or later:

  ```bash
  hf repos settings <you>/my-ocr-board --repo-type dataset --public
  ```

  Pass `--public` to `share` to create it public in one step. An existing board repo keeps its visibility.
- Never publish the results repo or the OCR output repo instead. They contain page text.

## Troubleshooting

- **`judge` says the inputs changed.** A model was re-run, the pages changed, or a judge setting changed. Pass `--full-rejudge` (a full judge run again) or `--save-results` with a new repo.
- **A model ranks on fewer pages than the others.** Its failed pages (`[OCR ERROR]`) are left out. The leaderboard's Pages column and the results card's Coverage section show how many.
- **Many outputs are truncated.** `judge` warns when the text cap cuts more than half of the outputs. Use the cap that `audit --screen` suggests.
- **`share` says the refit does not match the leaderboard.** The last judge run stopped after a checkpoint. Run `judge` again to finish it.
