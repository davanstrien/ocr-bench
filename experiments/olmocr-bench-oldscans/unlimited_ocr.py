# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#     "torch==2.10.0",
#     "torchvision==0.25.0",
#     "transformers==4.57.1",
#     "pillow==12.1.1",
#     "einops==0.8.2",
#     "addict==2.4.0",
#     "easydict==1.13",
#     "pymupdf==1.27.2.2",
#     "matplotlib==3.10.8",
#     "psutil==7.2.2",
#     "huggingface-hub>=0.34",
# ]
# ///
"""
Convert the olmOCR-bench `old_scans` subset with baidu/Unlimited-OCR and write
candidate markdown in the layout `olmocr.bench.benchmark` expects -- the same
convention as convert.py (PaddleOCR-VL), so the score job ranks them together.

Unlimited-OCR is a native document parser; it takes IMAGES, so we render each
PDF page to PNG (PyMuPDF) at the model's recommended DPI (300) before inference.
Decoding is deterministic per the card: temperature 0 + the DeepSeek-OCR-style
no-repeat-ngram processor (no_repeat_ngram_size=35), single-image "gundam" config.

Runs as a uv script (`hf jobs uv run`) -- transformers + the model live here, NOT
olmocr (that is only in the score job). Pins transformers==4.57.1 per the card,
which in turn pins an older huggingface_hub without `sync_bucket`; since the uv
image runs as ROOT it can write the bucket FUSE mount directly, so we mount it
read-write and write candidate files straight to /bucket (no sync needed).

    hf jobs uv run --flavor l4x1 -s HF_TOKEN \
        -v hf://buckets/davanstrien/paddleocr-vl16-oldscans:/bucket \
        unlimited_ocr.py

Env:
  CANDIDATE    output subfolder + label (default unlimited_ocr)
  DPI          PDF->PNG render DPI (default 300, the card's recommendation)
  OUT_ROOT     where to write (default /bucket, i.e. the bucket mount)
  NUM_SHARDS   data-parallel fan-out: run this many jobs with SHARD=0..N-1 (default 1)
  SHARD        which shard this job handles (default 0)
  LIMIT        cap number of PDFs (plumbing smoke test; 0 = all)
"""
import json
import os
import re
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

import fitz  # PyMuPDF
from huggingface_hub import hf_hub_download

# Unlimited-OCR's "document parsing." prompt emits DeepSeek-OCR-style grounding:
# `<|det|>category [x1,y1,x2,y2]<|/det|>` before each text span. Strip it to
# recover the plain transcription, matching the other models' clean-text output.
_DET = re.compile(r"<\|det\|>.*?<\|/det\|>", re.S)
_SPECIAL = re.compile(r"<\|[^|]*\|>")


def clean(text):
    return _SPECIAL.sub("", _DET.sub("", text)).strip()

BENCH_REPO = "allenai/olmOCR-bench"
JSONL_PATH = "bench_data/old_scans.jsonl"
MODEL_ID = "baidu/Unlimited-OCR"
CANDIDATE = os.environ.get("CANDIDATE", "unlimited_ocr")
DPI = int(os.environ.get("DPI", "300"))
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
from transformers import AutoModel, AutoTokenizer  # noqa: E402

print(f"cuda available: {torch.cuda.is_available()}", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModel.from_pretrained(
    MODEL_ID, trust_remote_code=True, use_safetensors=True, torch_dtype=torch.bfloat16
).eval().cuda()


def resolve_pdf(pdf_field):
    """Reuse the PDF on the bucket mount if present; else download once and stage
    it under OUT_ROOT/pdfs so sync_bucket adds it for the scorer."""
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


def infer_markdown(image_path, work):
    """Unlimited-OCR document parsing. The transformers `model.infer` saves to
    output_path; capture its return if it gives text, else read the saved file."""
    out_dir = work / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    result = model.infer(
        tokenizer,
        prompt="<image>document parsing.",
        image_file=str(image_path),
        output_path=str(out_dir),
        base_size=1024, image_size=640, crop_mode=True,   # single-image "gundam"
        max_length=32768,
        no_repeat_ngram_size=35, ngram_window=128,
        save_results=True,
    )
    if isinstance(result, str) and result.strip():
        return clean(result), "return"
    for f in sorted(out_dir.glob("**/*")):
        if f.is_file() and f.suffix.lower() in (".mmd", ".md", ".txt"):
            return clean(f.read_text()), f"file:{f.name}"
    return "", "empty"


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
            work = Path(td)
            mds = {}
            for pg in pages:
                png = work / f"pg{pg}.png"
                render_page(pdf_path, pg, str(png))
                mds[pg], src = infer_markdown(png, work)
    except Exception as e:
        print(f"[WARN] {pdf_field}: {e}", flush=True)
        mds, src = {pg: "" for pg in pages}, "error"
    md_base = os.path.splitext(pdf_field)[0]
    for pg in pages:
        fp = cand_dir / f"{md_base}_pg{pg}_repeat1.md"
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(mds.get(pg, ""))
    n = len(next(iter(mds.values()), "")) if mds else 0
    print(f"[{i}/{len(items)}] {pdf_field} -> {n} chars ({src})", flush=True)

(OUT_ROOT / "old_scans.jsonl").write_text(Path(jsonl_local).read_text())
print(f"Done. Wrote {CANDIDATE} candidate to {OUT_ROOT}", flush=True)
