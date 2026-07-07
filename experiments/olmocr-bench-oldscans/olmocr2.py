# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "openai",
#     "pymupdf",
#     "pillow",
#     "huggingface-hub>=0.34",
#     "tenacity",
# ]
# ///
"""
Convert the olmOCR-bench `old_scans` subset with allenai/olmOCR-2-7B-1025 -- the
model olmOCR-bench was built for -- run *faithfully*: olmOCR's own prompt, its
1288px longest-dimension render, greedy decoding, and its YAML-front-matter output
stripped to plain body text (what the bench scores). Writes candidate markdown in
the layout `olmocr.bench.benchmark` expects, same convention as convert.py /
nuextract3.py, so the score job ranks it alongside the other models.

olmOCR-2 is served separately with one command (HF Jobs, exposed port) --
`vllm serve allenai/olmOCR-2-7B-1025 --max-model-len 16384` -- and this client
hits that OpenAI-compatible endpoint. Runs as a uv script (`hf jobs uv run`), root,
writing the bucket mount directly:

    hf jobs uv run --flavor cpu-upgrade -s HF_TOKEN \
        -v hf://buckets/davanstrien/paddleocr-vl16-oldscans:/bucket \
        -e ENDPOINT=https://<jobid>--8000.hf.jobs/v1 \
        olmocr2.py

Env:
  ENDPOINT     OpenAI-compatible base URL of the served model (required), incl. /v1
  API_KEY      bearer token for the endpoint (default: $HF_TOKEN)
  MODEL_ID     served model name (default allenai/olmOCR-2-7B-1025)
  CANDIDATE    output subfolder + label (default olmocr2)
  TARGET_DIM   longest-side render px (default 1288, olmOCR's spec)
  OUT_ROOT     where to write (default /bucket, i.e. the bucket mount)
  NUM_SHARDS   data-parallel fan-out (default 1); SHARD = 0..N-1
  SHARD        which shard this job handles (default 0)
  LIMIT        cap number of PDFs (plumbing smoke test; 0 = all)
  READY_WAIT_S how long to poll the endpoint for readiness (default 600s)
"""
import base64
import json
import os
import shutil
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import fitz  # PyMuPDF
from huggingface_hub import hf_hub_download
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

BENCH_REPO = "allenai/olmOCR-bench"
JSONL_PATH = "bench_data/old_scans.jsonl"

ENDPOINT = os.environ["ENDPOINT"].rstrip("/")
API_KEY = os.environ.get("API_KEY") or os.environ["HF_TOKEN"]
MODEL_ID = os.environ.get("MODEL_ID", "allenai/olmOCR-2-7B-1025")
CANDIDATE = os.environ.get("CANDIDATE", "olmocr2")
TARGET_DIM = int(os.environ.get("TARGET_DIM", "1288"))
OUT_ROOT = Path(os.environ.get("OUT_ROOT", "/bucket"))
NUM_SHARDS = int(os.environ.get("NUM_SHARDS", "1"))
SHARD = int(os.environ.get("SHARD", "0"))
LIMIT = int(os.environ.get("LIMIT", "0"))
READY_WAIT_S = int(os.environ.get("READY_WAIT_S", "600"))

# olmOCR-2 prompt, verbatim from olmocr.prompts.build_no_anchoring_v4_yaml_prompt
PROMPT = (
    "Attached is one page of a document that you must process. "
    "Just return the plain text representation of this document as if you were reading it naturally. Convert equations to LateX and tables to HTML.\n"
    "If there are any figures or charts, label them with the following markdown syntax ![Alt text describing the contents of the figure](page_startx_starty_width_height.png)\n"
    "Return your output as markdown, with a front matter section on top specifying values for the primary_language, is_rotation_valid, rotation_correction, is_table, and is_diagram parameters."
)

client = OpenAI(base_url=ENDPOINT, api_key=API_KEY)

# ---- wait for the served endpoint to come up --------------------------------
deadline = time.time() + READY_WAIT_S
while True:
    try:
        models = client.models.list()
        print(f"endpoint ready: {[m.id for m in models.data]}", flush=True)
        break
    except Exception as e:
        if time.time() > deadline:
            raise RuntimeError(f"endpoint not ready after {READY_WAIT_S}s: {e}")
        print(f"waiting for endpoint... ({e.__class__.__name__})", flush=True)
        time.sleep(15)

# ---- test manifest ----------------------------------------------------------
jsonl_local = hf_hub_download(BENCH_REPO, JSONL_PATH, repo_type="dataset")
tests = [json.loads(ln) for ln in Path(jsonl_local).read_text().splitlines() if ln.strip()]

pages_by_pdf = defaultdict(set)
for t in tests:
    pages_by_pdf[t["pdf"]].add(int(t.get("page", 1)))
print(f"{len(tests)} tests across {len(pages_by_pdf)} PDFs  candidate={CANDIDATE} dim={TARGET_DIM}", flush=True)


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


def render_b64(pdf_path, page_num, png_path):
    """Render one page so the longest side is TARGET_DIM px (olmOCR's spec)."""
    doc = fitz.open(pdf_path)
    page = doc[page_num - 1]
    zoom = TARGET_DIM / max(page.rect.width, page.rect.height)
    page.get_pixmap(matrix=fitz.Matrix(zoom, zoom)).save(png_path)
    doc.close()
    return base64.b64encode(Path(png_path).read_bytes()).decode()


def strip_front_matter(text):
    """olmOCR-2 emits a `--- ... ---` YAML header then the body; the bench scores
    the body. Drop a leading front-matter block if present."""
    t = text.lstrip()
    if t.startswith("---"):
        parts = t.split("---", 2)
        if len(parts) == 3:
            return parts[2].strip()
    return text.strip()


@retry(stop=stop_after_attempt(4), wait=wait_exponential(min=2, max=30))
def infer_markdown(b64):
    resp = client.chat.completions.create(
        model=MODEL_ID,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ],
        }],
        temperature=0.0,
        max_tokens=4096,
    )
    return strip_front_matter(resp.choices[0].message.content or "")


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
                b64 = render_b64(pdf_path, pg, str(png))
                mds[pg] = infer_markdown(b64)
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

print(f"Done. Wrote {CANDIDATE} candidate to {OUT_ROOT}", flush=True)
