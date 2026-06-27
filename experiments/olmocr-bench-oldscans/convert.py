"""
Job 1 (GPU): run PaddleOCR-VL-1.6 over the olmOCR-bench `old_scans` subset and
write candidate markdown in the exact layout `olmocr.bench.benchmark` expects.

Runs with PaddlePaddle's docker image (paddle 3.2.1 + paddleocr 3.6.0 + the v1.6
weights preinstalled) via the image's python3.10 -- not uv. Every import here
(paddleocr, huggingface_hub, stdlib) is already in the image, so there is no
PEP 723 header.

Fidelity: the markdown extraction matches olmOCR-bench's own runner
(`olmocr/bench/runners/run_paddlevl.py`) -- `res.markdown["markdown_texts"]`, per
page, with a bare default pipeline and NO tuning (no max_pixels / prompts / dpi).
The one intentional difference is `pipeline_version="v1.6"`: upstream calls
`PaddleOCRVL()` with no version (an earlier PaddleOCR-VL), while this measures 1.6
as its model card specifies. So we follow the bench runner's extraction and
PaddlePaddle's documented v1.6 defaults.

This image runs as the non-root `paddleocr` user, which CANNOT write the bucket
FUSE mount (root-owned). So we write outputs to a container-local dir and push
them with `sync_bucket()` (mount-free HTTP upload) at the end. The bucket is
mounted read-only purely to deliver this script.

Delivery + run (see README for full commands):
    hf buckets cp convert.py hf://buckets/davanstrien/paddleocr-vl16-oldscans/convert.py
    hf jobs run --flavor l4x1 -s HF_TOKEN \
        -v hf://buckets/davanstrien/paddleocr-vl16-oldscans:/bucket:ro \
        ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-vl:latest-nvidia-gpu \
        python3 /bucket/convert.py

Env:
  OUT_ROOT          local staging dir (default /tmp/olmocr-oldscans-out)
  BUCKET            bucket to sync results to (default below)
  PIPELINE_VERSION  PaddleOCRVL version (default v1.6). "v1" = the original 0.9B
                    PaddleOCR-VL = the version on the olmOCR-bench leaderboard
                    (37.8) -> run it (with a distinct CANDIDATE) for a strict
                    same-version reproduction.
  CANDIDATE         output subfolder + model label (default paddleocr_vl_16)
  LIMIT             cap number of PDFs (plumbing smoke test; 0 = all). With a cap,
                    the un-converted docs are scored FAILED, so the result is not
                    a representative score -- use a smoke run only to check plumbing.
"""
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

from huggingface_hub import hf_hub_download, sync_bucket
from paddleocr import PaddleOCRVL

BENCH_REPO = "allenai/olmOCR-bench"
JSONL_PATH = "bench_data/old_scans.jsonl"
CANDIDATE = os.environ.get("CANDIDATE", "paddleocr_vl_16")   # any name except "pdfs"
PIPELINE_VERSION = os.environ.get("PIPELINE_VERSION", "v1.6")
OUT_ROOT = Path(os.environ.get("OUT_ROOT", "/tmp/olmocr-oldscans-out"))
BUCKET = os.environ.get("BUCKET", "hf://buckets/davanstrien/paddleocr-vl16-oldscans")
LIMIT = int(os.environ.get("LIMIT", "0"))

# ---- test manifest ----------------------------------------------------------
jsonl_local = hf_hub_download(BENCH_REPO, JSONL_PATH, repo_type="dataset")
tests = [json.loads(ln) for ln in Path(jsonl_local).read_text().splitlines() if ln.strip()]

pages_by_pdf = defaultdict(set)
for t in tests:
    pages_by_pdf[t["pdf"]].add(int(t.get("page", 1)))
print(f"{len(tests)} tests across {len(pages_by_pdf)} PDFs -> {OUT_ROOT}", flush=True)

# ---- model (vendor default, no tuning) --------------------------------------
print(f"pipeline_version={PIPELINE_VERSION}  candidate={CANDIDATE}", flush=True)
pipeline = PaddleOCRVL(pipeline_version=PIPELINE_VERSION)


def resolve_pdf(pdf_field):
    """Reuse the PDF already on the bucket mount if present; otherwise download it
    once and stage it under OUT_ROOT/pdfs so sync_bucket adds it to the bucket for
    the scorer (benchmark.py needs <dir>/pdfs). Avoids a separate download + sync,
    and skips re-downloading on later runs (the bucket is mounted at /bucket)."""
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


def page_markdowns(pdf_path):
    """Per-page markdown, exactly as run_paddlevl.py does: res.markdown['markdown_texts']."""
    return [res.markdown["markdown_texts"] for res in pipeline.predict(str(pdf_path))]


# ---- convert ----------------------------------------------------------------
cand_dir = OUT_ROOT / CANDIDATE
items = sorted(pages_by_pdf.items())
if LIMIT:
    items = items[:LIMIT]
    print(f"LIMIT={LIMIT} (plumbing smoke test -- expect a low score)", flush=True)

for i, (pdf_field, pages) in enumerate(items, 1):
    try:
        mds = page_markdowns(resolve_pdf(pdf_field))
    except Exception as e:  # keep going; a missing page just fails its tests
        print(f"[WARN] {pdf_field}: {e}", flush=True)
        mds = []
    md_base = os.path.splitext(pdf_field)[0]          # mirrors benchmark.py exactly
    for pg in pages:
        md = mds[pg - 1] if 0 <= pg - 1 < len(mds) else ""   # 1-indexed page -> 0-indexed
        fp = cand_dir / f"{md_base}_pg{pg}_repeat1.md"
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(md)
    n = len(mds[0]) if mds else 0
    print(f"[{i}/{len(items)}] {pdf_field} -> {n} chars", flush=True)

# the scorer needs the jsonl next to the candidate folder
(OUT_ROOT / "old_scans.jsonl").write_text(Path(jsonl_local).read_text())

# push results to the bucket over HTTP (the FUSE mount is not writable as non-root)
print(f"Syncing {OUT_ROOT} -> {BUCKET}", flush=True)
sync_bucket(str(OUT_ROOT), BUCKET)
print("Done.", flush=True)
