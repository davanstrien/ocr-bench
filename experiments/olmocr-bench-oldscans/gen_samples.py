"""Generate a self-contained samples.html: source scan vs. PaddleOCR-VL-1.6
output for a handful of old_scans docs, with hallucinated CJK glyphs highlighted.
Scans are embedded as base64 JPEG so the page is a single portable file.

Populate the data dir from the bucket first, then render:

    B=hf://buckets/davanstrien/paddleocr-vl16-oldscans
    for id in 1 5 10 27 30 50 56; do
      hf buckets cp $B/pdfs/old_scans/$id.pdf            samples_data/$id.pdf
      hf buckets cp $B/paddleocr_vl_16/old_scans/${id}_pg1_repeat1.md samples_data/$id.md
    done
    uv run --with pypdfium2 --with pillow gen_samples.py --data samples_data
"""
import argparse
import base64
import html
import io
import re
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image  # noqa: F401  (pypdfium2 .to_pil needs Pillow installed)

DOCS = [
    ("5", "Typed letter — near-perfect transcription"),
    ("10", "Typed letter — cursive signature dropped; 'Sincerely,' loops x3"),
    ("1", "Handwritten letter — readable, character-level errors"),
    ("30", "Typed letter — Chinese 场景 inserted mid-sentence"),
    ("56", "Q&A catechism — 源 emitted for 'sources'"),
    ("50", "Dense cursive — garbled + multiple CJK glyphs"),
    ("27", "Ornate blackletter header skipped + cursive garbled"),
]

CJK = re.compile(r"[㐀-鿿＀-￯]+")


def scan_b64(pdf_path: Path, width: int = 1000) -> str:
    pdf = pdfium.PdfDocument(str(pdf_path))
    page = pdf[0]
    scale = width / page.get_size()[0]
    pil = page.render(scale=scale).to_pil().convert("RGB")
    buf = io.BytesIO()
    pil.save(buf, "JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode()


def render_output(text: str) -> str:
    return CJK.sub(lambda m: f"<mark>{m.group(0)}</mark>", html.escape(text))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="samples_data")
    ap.add_argument("--out", default="samples.html")
    args = ap.parse_args()
    data = Path(args.data)

    cards = []
    for did, cap in DOCS:
        img = scan_b64(data / f"{did}.pdf")
        md = (data / f"{did}.md").read_text()
        cards.append(
            f"""
    <section class="card">
      <h2>old_scans/{did} <span>— {html.escape(cap)}</span></h2>
      <div class="pair">
        <div class="scan"><img loading="lazy" src="data:image/jpeg;base64,{img}" alt="scan {did}"></div>
        <pre class="out">{render_output(md)}</pre>
      </div>
    </section>"""
        )

    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PaddleOCR-VL-1.6 — olmOCR-bench old_scans samples</title>
<style>
  body {{ font: 15px/1.5 -apple-system, system-ui, sans-serif; max-width: 1200px; margin: 2rem auto; padding: 0 1rem; color: #1a1a1a; }}
  h1 {{ margin-bottom: .2rem; }}
  .lede {{ color: #555; }}
  mark {{ background: #ffd54f; padding: 0 2px; border-radius: 2px; }}
  .card {{ border: 1px solid #e3e3e3; border-radius: 8px; margin: 1.5rem 0; overflow: hidden; }}
  .card h2 {{ font-size: 1rem; margin: 0; padding: .6rem .9rem; background: #f6f6f6; border-bottom: 1px solid #e3e3e3; }}
  .card h2 span {{ font-weight: 400; color: #666; }}
  .pair {{ display: grid; grid-template-columns: 1fr 1fr; }}
  .scan {{ background: #fafafa; border-right: 1px solid #eee; padding: .5rem; text-align: center; }}
  .scan img {{ max-width: 100%; height: auto; box-shadow: 0 1px 4px rgba(0,0,0,.12); }}
  .out {{ margin: 0; padding: .9rem; white-space: pre-wrap; word-break: break-word; font: 13px/1.55 ui-monospace, monospace; max-height: 82vh; overflow: auto; }}
  @media (max-width: 820px) {{ .pair {{ grid-template-columns: 1fr; }} .scan {{ border-right: none; border-bottom: 1px solid #eee; }} }}
</style></head>
<body>
  <h1>PaddleOCR-VL-1.6 on olmOCR-bench <code>old_scans</code></h1>
  <p class="lede">Source scan (left) vs. the model's markdown output (right) — default v1.6 pipeline, no tuning.
  <mark>Highlighted</mark> spans are hallucinated CJK glyphs on English documents. Overall old_scans score:
  <b>38.6%</b> (preliminary). Scans: Library of Congress via
  <a href="https://huggingface.co/datasets/allenai/olmOCR-bench">allenai/olmOCR-bench</a> (ODC-BY).</p>
  {"".join(cards)}
</body></html>"""

    out = Path(args.out)
    out.write_text(page)
    print(f"wrote {out} ({len(page) // 1024} KB)")


if __name__ == "__main__":
    main()
