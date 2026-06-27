# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "transformers",
#     "torch",
#     "torchvision",
#     "accelerate",
#     "pillow",
#     "pymupdf",
#     "einops",
#     "huggingface-hub>=0.34",
# ]
# ///
"""
Convert the olmOCR-bench `old_scans` subset with numind/NuExtract3 (markdown mode)
and write candidate markdown in the layout `olmocr.bench.benchmark` expects -- the
same convention as convert.py / unlimited_ocr.py, so the score job ranks them all
together.

NuExtract3 is extraction-first but has a native image->Markdown mode; we use
`mode="markdown"`, **non-thinking + greedy** (`enable_thinking=False`,
`do_sample=False`) -- the decoding-comparable setting vs the other models (thinking
mode would be temperature 0.6, non-deterministic). It takes IMAGES, so we render
each PDF page to PNG at the card's recommended DPI (170).

Runs as a uv script (`hf jobs uv run`), root, writing the bucket mount directly:

    hf jobs uv run --flavor l4x1 -s HF_TOKEN \
        -v hf://buckets/davanstrien/paddleocr-vl16-oldscans:/bucket \
        nuextract3.py

Env:
  CANDIDATE    output subfolder + label (default nuextract3)
  DPI          PDF->PNG render DPI (default 170, the card's recommendation)
  OUT_ROOT     where to write (default /bucket, i.e. the bucket mount)
  NUM_SHARDS   data-parallel fan-out (default 1); SHARD = 0..N-1
  SHARD        which shard this job handles (default 0)
  LIMIT        cap number of PDFs (plumbing smoke test; 0 = all)
"""
import json
import os
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

import fitz  # PyMuPDF
from huggingface_hub import hf_hub_download

BENCH_REPO = "allenai/olmOCR-bench"
JSONL_PATH = "bench_data/old_scans.jsonl"
MODEL_ID = "numind/NuExtract3"
CANDIDATE = os.environ.get("CANDIDATE", "nuextract3")
DPI = int(os.environ.get("DPI", "170"))
OUT_ROOT = Path(os.environ.get("OUT_ROOT", "/bucket"))
NUM_SHARDS = int(os.environ.get("NUM_SHARDS", "1"))
SHARD = int(os.environ.get("SHARD", "0"))
LIMIT = int(os.environ.get("LIMIT", "0"))

# ---- test manifest ----------------------------------------------------------
jsonl_local = hf_hub_download(BENCH_REPO, JSONL_PATH, repo_type="dataset")
tests = [json.loads(ln) for ln in Path(jsonl_local).read_text().splitlines() if ln.strip()]

pages_by_pdf = defaultdict(set)
for t in tests:
    pages_by_pdf[t["pdf"]].add(int(t.get("page", 1)))
print(f"{len(tests)} tests across {len(pages_by_pdf)} PDFs  candidate={CANDIDATE} dpi={DPI}", flush=True)

# ---- model ------------------------------------------------------------------
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import AutoModelForImageTextToText, AutoProcessor  # noqa: E402

print(f"cuda available: {torch.cuda.is_available()}", flush=True)
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
).eval()


def resolve_pdf(pdf_field):
    mounted = Path("/bucket/pdfs") / pdf_field
    if mounted.is_file():
        return str(mounted)
    for cand in (f"bench_data/{pdf_field}", f"bench_data/pdfs/{pdf_field}", pdf_field):
        try:
            local = hf_hub_download(BENCH_REPO, cand, repo_type="dataset")
        except Exception:
            continue
        dest = OUT_ROOT / "pdfs" / pdf_field
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(local, dest)
        return local
    raise FileNotFoundError(pdf_field)


def render_page(pdf_path, page_num, png_path):
    doc = fitz.open(pdf_path)
    mat = fitz.Matrix(DPI / 72, DPI / 72)
    doc[page_num - 1].get_pixmap(matrix=mat).save(png_path)
    doc.close()


def infer_markdown(image_path):
    """NuExtract3 image->Markdown, non-thinking + greedy."""
    image = Image.open(image_path).convert("RGB")
    messages = [{"role": "user", "content": [{"type": "image", "image": image}]}]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        mode="markdown",
        enable_thinking=False,
    ).to(model.device)
    with torch.inference_mode():
        gen = model.generate(**inputs, max_new_tokens=4096, do_sample=False)
    gen = gen[:, inputs.input_ids.shape[1]:]
    return processor.batch_decode(
        gen, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()


# ---- convert ----------------------------------------------------------------
cand_dir = OUT_ROOT / CANDIDATE
items = sorted(pages_by_pdf.items())
if NUM_SHARDS > 1:
    items = items[SHARD::NUM_SHARDS]
    print(f"shard {SHARD}/{NUM_SHARDS}: {len(items)} PDFs", flush=True)
if LIMIT:
    items = items[:LIMIT]
    print(f"LIMIT={LIMIT} (plumbing smoke test)", flush=True)

for i, (pdf_field, pages) in enumerate(items, 1):
    try:
        pdf_path = resolve_pdf(pdf_field)
        with tempfile.TemporaryDirectory() as td:
            mds = {}
            for pg in pages:
                png = Path(td) / f"pg{pg}.png"
                render_page(pdf_path, pg, str(png))
                mds[pg] = infer_markdown(png)
    except Exception as e:
        print(f"[WARN] {pdf_field}: {e}", flush=True)
        mds = {pg: "" for pg in pages}
    md_base = os.path.splitext(pdf_field)[0]
    for pg in pages:
        fp = cand_dir / f"{md_base}_pg{pg}_repeat1.md"
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(mds.get(pg, ""))
    n = len(next(iter(mds.values()), "")) if mds else 0
    print(f"[{i}/{len(items)}] {pdf_field} -> {n} chars", flush=True)

(OUT_ROOT / "old_scans.jsonl").write_text(Path(jsonl_local).read_text())
print(f"Done. Wrote {CANDIDATE} candidate to {OUT_ROOT}", flush=True)
