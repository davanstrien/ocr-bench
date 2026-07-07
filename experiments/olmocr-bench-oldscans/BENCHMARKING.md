# olmOCR-bench old_scans — multi-model comparison

Extends the PaddleOCR-VL experiment ([README.md](README.md)) to a **10-model
comparison**, all scored through the same harness — for my own benchmarking. Same
subset (`old_scans`, 98 Library-of-Congress scans), same scorer (stock
`olmocr.bench.benchmark`), greedy decoding, no tuning. Core models run 2026-06-27;
olmOCR-2 added 2026-07-07.

## Results

| Model | params | **old_scans** | present | absent | order | baseline |
|---|---|---|---|---|---|---|
| olmOCR-2 | 7B | **46.8** | 45.5 | 91.4 | 31.1 | 100.0 |
| LightOnOCR-2 | 1B | **42.2** | **45.9** | 47.1 | **34.5** | 100.0 |
| dots.ocr | 1.7B | **41.6** | 39.1 | 81.4 | 29.9 | 96.9 |
| GLM-OCR | 0.9B | **40.5** | 36.9 | 88.6 | 27.1 | 96.9 |
| PaddleOCR-VL v1.6 | 0.9B | **38.6** | 31.2 | 95.7 | 27.7 | 84.7 |
| PaddleOCR-VL v1 | 0.9B | **38.2** | 32.3 | 95.7 | 24.9 | 88.8 |
| NuExtract3 | 4.5B | **37.8** | 41.6 | 41.4 | 30.5 | 100.0 |
| DeepSeek-OCR | ~3B | **34.6** | 27.2 | 92.9 | 23.2 | 100.0 |
| FireRed-OCR | 2.1B | **33.3** | 30.8 | 62.9 | 25.4 | 77.6 |
| Unlimited-OCR | 3.3B | **30.6** | 29.0 | 50.0 | 25.4 | 89.8 |

`old_scans` = present + absent + order (the leaderboard's "Old scans" column).
`baseline` = auto no-hallucination check over these 98 scans (NOT the leaderboard "Base" column).
Param counts vendor-reported/approx.

**Read across the columns, not just the headline.** olmOCR-2 tops it — expected, it's
the model olmOCR-bench was built around. But rank by pure transcription (`present`) and
a **1B model, LightOnOCR-2 (45.9), leads the field**, edging olmOCR-2 (45.5). And the
headline and `present` orders disagree: PaddleOCR-VL v1.6 beats NuExtract3 on the
headline (38.6 vs 37.8) but reads far less of the page (`present` 31.2 vs 41.6). The
sub-score section below unpacks why.

## ⚠️ Read the sub-scores, not just the headline

The single `old_scans` number **conflates two different abilities**, and is
misleading on its own:

- **`present`** — did the model transcribe the body text? *(transcription quality)*
- **`absent`** — did it *exclude* the boilerplate the bench wants dropped:
  letterheads (`LUCIEN BECKNER`, `TELEPHONE 478`), archival stamps (`ack 5/27/14`),
  page numbers (`31`, `2590`)? *(boilerplate exclusion)*

These pull in opposite directions **by architecture**:

- **NuExtract3 is the best *transcriber*** — `present` 41.6, well above paddle's
  31.2, and it never hallucinates CJK (`baseline` 100%). It scores low on `absent`
  (41.4) only because markdown-mode transcribes the letterhead/stamps. Inspected:
  those appear as **unmarked plain body text** at the top of the page, *not* in
  `<figure>`/HTML you could filter on — so the `absent` failures are real, not a
  formatting artifact you could strip away.
- **PaddleOCR-VL is the best *boilerplate-excluder*** — `absent` 95.7, because its
  layout pipeline drops running headers/footers. But it reads less of the body
  (`present` 31.2) and hallucinates CJK glyphs on the hardest scans (`baseline`
  84.7).

So **NuExtract "losing" on `old_scans` (37.8 vs 38.6) is not a transcription
deficit** — it reads *better*; it just doesn't do boilerplate exclusion, which a
layout pipeline gets for free. Pick by use case:

- want the most faithful text → **NuExtract3** (`present` 41.6).
- want clean body-only markdown without boilerplate → **PaddleOCR-VL** (`absent` 95.7).

*(Deliberately not done: a heuristic "drop the leading letterhead + standalone page
numbers" pass that would lift NuExtract's `absent`. That's gaming the very
boilerplate-exclusion criterion the bench is testing the model to do natively, so
it's left out — the honest move is to report the sub-scores.)*

## Why it matters

olmOCR-bench encodes olmOCR's own goal: clean, linearised **reading-order body
text** for LLM training/RAG, where headers, letterheads and page numbers are noise
to drop. So the headline rewards leaving them out — reasonable if that's what you
want. For **faithful OCR of the whole page** (archives, where the letterhead /
stamp / marginal note *is* the record), the same number ranks the model you'd
prefer *lower*. A benchmark score measures fitness for one purpose — check it's
yours before trusting the ranking.

## olmOCR-2 — the model the benchmark was built for

olmOCR-2-7B (`allenai/olmOCR-2-7B-1025`) tops the table (46.8). No surprise: olmOCR-bench
encodes olmOCR's own goal, and this model is trained for exactly it. Even so it fails
53% of the tests, and it's the largest model here — the 1B LightOnOCR-2 matches its
transcription (`present` 45.9 vs 45.5).

Run faithfully — its own prompt (`build_no_anchoring_v4_yaml_prompt`), its 1288px
longest-side render, YAML front matter stripped to body text — served on a Job with an
exposed port and hit by `olmocr2.py`:

```bash
# 1. serve the model on a Job (exposed port); note the endpoint URL it prints
hf jobs run --detach --expose 8000 --flavor a10g-small --timeout 45m -s HF_TOKEN \
    vllm/vllm-openai \
    vllm serve allenai/olmOCR-2-7B-1025 --max-model-len 16384

# 2. convert against that endpoint (writes the olmocr2 candidate into the bucket)
B=hf://buckets/davanstrien/paddleocr-vl16-oldscans
hf jobs uv run --flavor cpu-upgrade -s HF_TOKEN -v $B:/bucket \
    -e ENDPOINT=https://<jobid>--8000.hf.jobs/v1 olmocr2.py

# 3. cancel the serve job once the convert finishes (exposed ports bill per minute)
hf jobs cancel <serve-jobid>
```

`olmocr2.py` polls the endpoint until it's ready, so step 2 can be launched while the
server warms up. Pattern: [Serve Models on Jobs](https://huggingface.co/docs/hub/jobs-serving).

## Provenance / reproducing each model

Committed runners in this folder produce five of the candidates: `convert.py`
(PaddleOCR-VL v1.6 + v1), `nuextract3.py`, `unlimited_ocr.py`, and `olmocr2.py`. The
other five — `dots.ocr`, `GLM-OCR`, `DeepSeek-OCR`, `FireRed-OCR`, `LightOnOCR-2` — were
produced by the corresponding [uv-scripts/ocr](https://huggingface.co/datasets/uv-scripts/ocr)
runners writing into the same `{candidate}/old_scans/{basename}_pg{page}_repeat1.md`
bucket layout, so `score.py` ranks all ten together. Their exact `hf jobs uv run`
invocations aren't captured in this folder yet — a documentation follow-up. `score.py`
reports scores over whatever candidate folders are present in the bucket, so re-running
it after adding a model re-ranks the whole set (that's how olmOCR-2 was folded in).

## Fairness / processing notes

- **DPI**: each model's recommended PDF→PNG render DPI — NuExtract **170**,
  Unlimited-OCR **300**; paddle rasterizes internally. Different DPI is a confound;
  using each model's own default is the per-model-fair choice (footnoted here).
- **Unlimited-OCR** emits DeepSeek-OCR-style `<|det|>category [bbox]<|/det|>`
  grounding before each span; we **strip it** to recover plain text comparable to
  the others' clean markdown.
- **NuExtract3**: `mode="markdown"`, **non-thinking + greedy** — the decoding-
  comparable setting (thinking mode is temperature 0.6, non-deterministic).
- **Decoding**: greedy for all four.
- These two models take **images**, so the convert scripts render PDF→PNG (the
  paddle pipeline rasterizes internally). They run as `hf jobs uv run` (root), so
  they write the bucket mount **directly** — no `sync_bucket`, because
  `transformers==4.57.1` pins an older `huggingface_hub` that lacks it.
- **Not size-matched**: NuExtract (4.5B) and Unlimited-OCR (3.3B) are 3–5× larger
  than PaddleOCR-VL (0.9B). Fine for "what's the best number," but note it.

## Run

```bash
B=hf://buckets/davanstrien/paddleocr-vl16-oldscans
# convert (each writes its own candidate folder; LIMIT=3 for a smoke)
hf jobs uv run --flavor l4x1 --timeout 1h -s HF_TOKEN -v $B:/bucket unlimited_ocr.py
hf jobs uv run --flavor l4x1 --timeout 1h -s HF_TOKEN -v $B:/bucket nuextract3.py
# olmocr2.py additionally needs a served endpoint — see the olmOCR-2 section above
# score every candidate folder in the bucket together
hf jobs uv run --flavor cpu-upgrade -s HF_TOKEN -v $B:/bucket:ro score.py
```

`NUM_SHARDS`/`SHARD` on either convert for data-parallel fan-out (default 1).
