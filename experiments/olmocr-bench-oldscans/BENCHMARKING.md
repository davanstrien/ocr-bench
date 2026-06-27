# olmOCR-bench old_scans — multi-model comparison

Extends the PaddleOCR-VL experiment ([README.md](README.md)) to two more open
document models, scored through the same harness — for my own benchmarking. Same
subset (`old_scans`, 98 Library-of-Congress scans), same scorer (stock
`olmocr.bench.benchmark`), greedy decoding, no tuning. Run 2026-06-27.

## Results

| Model | params | **old_scans** | present | absent | order | baseline |
|---|---|---|---|---|---|---|
| PaddleOCR-VL v1.6 | 0.9B | **38.6** | 31.2 | 95.7 | 27.7 | 84.7 |
| PaddleOCR-VL v1 | 0.9B | **38.2** | 32.3 | 95.7 | 24.9 | 88.8 |
| NuExtract3 | 4.5B | **37.8** | **41.6** | 41.4 | 30.5 | **100.0** |
| Unlimited-OCR | 3.3B | **30.6** | 29.0 | 50.0 | 25.4 | 89.8 |

`old_scans` = present + absent + order (the leaderboard's "Old scans" column).

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
# score every candidate folder in the bucket together
hf jobs uv run --flavor cpu-upgrade -s HF_TOKEN -v $B:/bucket:ro score.py
```

`NUM_SHARDS`/`SHARD` on either convert for data-parallel fan-out (default 1).
