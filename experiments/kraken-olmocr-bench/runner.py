"""Pinned Kraken batch inference; smoke and full runs share this Jobs entrypoint."""

import argparse
import cProfile
import hashlib
import importlib.metadata
import json
import os
import platform
import pstats
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

DATASET = "allenai/olmOCR-bench"
DATA_REV = "54a96a6fb6a2bd3b297e59869491db4d3625b711"
MODEL = "small-models-for-glam/kraken-ppocrv6-medium"
MODEL_REV = "b26109b54a73e27b91c3fa83558ccba1ed485e03"
WEIGHTS_SHA = "15313b51ace64cbfa81f8f6ef25ad64f04e5a6fb7f7823e67b107527bc081ac9"
KRAKEN_REV = "3e6893a0d5eb06273494bfcdb9d3cac6159e0611"
IMAGE = "ghcr.io/astral-sh/uv@sha256:85d4cb1afa769a7338e095b927bee941cf5ec92266c7424b3f6c0f2748567248"
CATEGORIES = (
    "arxiv_math", "headers_footers", "long_tiny_text", "multi_column",
    "old_scans", "old_scans_math", "table_tests",
)
# These pinned benchmark pages exceed Pillow's default limit at the fixed
# 300 DPI. Allow their known sizes only when reopening our own rendered PNGs.
LARGE_RENDER_PIXELS = {
    ("old_scans/48.pdf", 1): 253864905,
    ("old_scans/51.pdf", 1): 331857945,
    ("old_scans/57.pdf", 1): 323914932,
}
START = time.perf_counter()


def log(message):
    print(f"[{time.perf_counter() - START:.1f}s] {message}", flush=True)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def read_json(path):
    # Mounted Bucket writes can become visible after close rather than immediately.
    # Bound retries so absent/corrupt checkpoints remain hard failures.
    for attempt in range(31):
        try:
            return json.loads(path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            if attempt == 30:
                raise
            time.sleep(1)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def key(page):
    return f"{Path(page['pdf']).with_suffix('')}_pg{page['page']}"


def manifest(bucket):
    from huggingface_hub import hf_hub_download

    if (bucket / "manifest.json").exists():
        return read_json(bucket / "manifest.json")
    pages = {}
    tests = []
    for category in CATEGORIES:
        source = Path(hf_hub_download(
            DATASET, f"bench_data/{category}.jsonl", repo_type="dataset", revision=DATA_REV,
        ))
        for line in source.read_text().splitlines():
            test = json.loads(line)
            tests.append(test)
            ident = (test["pdf"], test["page"])
            if ident in pages:
                assert pages[ident]["category"] == category
            pages[ident] = {"pdf": test["pdf"], "page": test["page"], "category": category}
    ordered = []
    for category in CATEGORIES:
        group = sorted(
            [p for p in pages.values() if p["category"] == category],
            key=lambda p: hashlib.sha256(f"kraken-smoke-v1:{key(p)}".encode()).hexdigest(),
        )
        for index, page in enumerate(group):
            ordered.append({**page, "smoke": index < 3, "shard": index % 4})
    assert len(ordered) == 1403 and len(tests) == 7019
    write_json(bucket / "manifest.json", ordered)
    return ordered


def render(page):
    import pypdfium2 as pdfium
    from huggingface_hub import hf_hub_download
    from PIL import Image

    path = Path("/tmp/rendered") / f"{key(page)}.png"
    meta_path = path.with_suffix(".json")
    if not path.exists():
        pdf = Path(hf_hub_download(
            DATASET, f"bench_data/pdfs/{page['pdf']}", repo_type="dataset", revision=DATA_REV,
        ))
        path.parent.mkdir(parents=True, exist_ok=True)
        with pdfium.PdfDocument(pdf) as doc:
            pdf_page = doc[page["page"] - 1]
            try:
                bitmap = pdf_page.render(scale=300 / 72)
                im = bitmap.to_pil().convert("RGB")
                im.save(path)
                write_json(meta_path, {
                    "pdf_sha256": digest(pdf), "image_sha256": digest(path),
                    "width": im.width, "height": im.height, "dpi": 300,
                })
                bitmap.close()
            finally:
                pdf_page.close()
    expected_pixels = LARGE_RENDER_PIXELS.get((page["pdf"], page["page"]))
    original_limit = Image.MAX_IMAGE_PIXELS
    try:
        if expected_pixels is not None:
            Image.MAX_IMAGE_PIXELS = expected_pixels
        with Image.open(path) as im:
            if expected_pixels is not None and im.width * im.height != expected_pixels:
                raise RuntimeError(f"Unexpected dimensions for known large page: {key(page)}")
            return im.convert("RGB"), read_json(meta_path)
    finally:
        Image.MAX_IMAGE_PIXELS = original_limit


def load_stack(weights, prepared):
    import torch
    from kraken.configs import RecognitionInferenceConfig, SegmentationInferenceConfig
    from kraken.tasks import RecognitionTaskModel, SegmentationTaskModel

    segmenter = SegmentationTaskModel.load_model()
    recognizer = RecognitionTaskModel.load_model(str(weights))
    seg_config = SegmentationInferenceConfig(
        text_direction="horizontal-lr", accelerator="cuda", precision="32-true",
    )
    rec_config = RecognitionInferenceConfig(
        batch_size=32, num_line_workers=0, padding=16, bidi_reordering=True,
        accelerator="cuda", precision="32-true",
    )

    def prepare_once(net, config):
        # Preserve the high-level task API, including merge and reading-order processing.
        with torch.inference_mode():
            net.prepare_for_inference(config)
        assert next(net.parameters()).device.type == "cuda"

        def already_prepared(requested):
            if requested is not config:
                raise RuntimeError("Prepared model cannot be reused with a different config")

        net.prepare_for_inference = already_prepared

    if prepared:
        for net in segmenter.seg_models:
            prepare_once(net, seg_config)
        prepare_once(recognizer.net, rec_config)
    return segmenter, recognizer, seg_config, rec_config


def canonical_geometry(value):
    # UUIDs identify lines/regions but are randomly allocated by each segmentation call.
    # Canonicalise identities, preserving every coordinate, sequence and membership edge.
    identifiers = {}

    def collect(node):
        if isinstance(node, dict):
            if "id" in node and node["id"] not in identifiers:
                identifiers[node["id"]] = f"id-{len(identifiers)}"
            for child in node.values():
                collect(child)
        elif isinstance(node, list):
            for child in node:
                collect(child)

    collect(value)

    def replace(node):
        if isinstance(node, dict):
            return {k: replace(v) for k, v in node.items() if k != "imagename"}
        if isinstance(node, (list, tuple)):
            return [replace(v) for v in node]
        if isinstance(node, str):
            return identifiers.get(node, node)
        return node

    return replace(value)


def synchronize():
    import torch

    torch.cuda.synchronize()


def recognise(recognizer, config, image, segmentation):
    if not segmentation.lines:
        return []
    records = list(recognizer.predict(image, segmentation, config))
    return [record.prediction for record in records]


def overlay(image, geometry, path):
    from PIL import ImageDraw

    image = image.copy()
    draw = ImageDraw.Draw(image)
    for index, line in enumerate(geometry["lines"] or []):
        boundary = line.get("boundary")
        baseline = line.get("baseline")
        if boundary:
            points = [tuple(p) for p in boundary]
            draw.line(points + [points[0]], fill="red", width=3)
        if baseline:
            points = [tuple(p) for p in baseline]
            draw.line(points, fill="blue", width=3)
            draw.text(points[0], str(index + 1), fill="blue", stroke_width=1)
    image.thumbnail((1400, 1800))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def run_pages(pages, root, weights, staged, resume_check=False):
    from kraken.containers import Segmentation

    start = time.perf_counter()
    segmenter, recognizer, seg_config, rec_config = load_stack(weights, staged)
    if resume_check:
        def unexpected_prediction(*args, **kwargs):
            raise AssertionError("Resume must skip completed predictions")

        segmenter.predict = unexpected_prediction
        recognizer.predict = unexpected_prediction
    synchronize()
    setup_seconds = time.perf_counter() - start
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "selected_pages.json", pages)
    results = []
    existing_errors = read_json(root / "errors.json") if (root / "errors.json").exists() else []

    def paths(page):
        stem = root / "pages" / key(page)
        return Path(str(stem) + ".geometry.json"), Path(str(stem) + ".result.json")

    def segment(page):
        geometry_path, result_path = paths(page)
        image, render_meta = render(page)
        if geometry_path.exists():
            checkpoint = read_json(geometry_path)
            assert checkpoint["render"] == render_meta
            return image, checkpoint, Segmentation(**checkpoint["geometry"])
        synchronize()
        t0 = time.perf_counter()
        profile = cProfile.Profile() if not results and page == pages[0] else None
        if profile:
            profile.enable()
        segmentation = segmenter.predict(im=image, config=seg_config)
        if profile:
            profile.disable()
            with (root / "segmentation-profile.txt").open("w") as stream:
                pstats.Stats(profile, stream=stream).sort_stats("cumulative").print_stats(60)
        synchronize()
        checkpoint = {
            "page": page, "render": render_meta, "geometry": asdict(segmentation),
            "seg_seconds": time.perf_counter() - t0,
        }
        write_json(geometry_path, checkpoint)
        log(f"SEG {key(page)} {checkpoint['seg_seconds']:.2f}s {len(segmentation.lines or [])} lines")
        return image, checkpoint, segmentation

    def recognition(page, image, checkpoint, original_segmentation=None):
        _, result_path = paths(page)
        if result_path.exists():
            result = read_json(result_path)
            if result.get("status") == "ok":
                results.append(result)
                return
        geometry = checkpoint["geometry"]
        segmentation = Segmentation(**geometry)
        assert canonical_geometry(asdict(segmentation)) == canonical_geometry(geometry)
        if original_segmentation is not None:
            segmentation = original_segmentation
        synchronize()
        t0 = time.perf_counter()
        lines = recognise(recognizer, rec_config, image, segmentation)
        synchronize()
        result = {
            "page": page, "status": "ok", "render": checkpoint["render"],
            "seg_seconds": checkpoint["seg_seconds"], "rec_seconds": time.perf_counter() - t0,
            "n_lines": len(segmentation.lines or []), "n_records": len(lines),
            "lines": lines, "text": "\n".join(lines),
        }
        assert result["n_lines"] == result["n_records"]
        output = root / "candidate" / f"{key(page)}_repeat1.md"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(result["text"])
        if page["smoke"]:
            overlay(image, geometry, root / "overlays" / f"{key(page)}.png")
        write_json(result_path, result)
        results.append(result)
        log(f"OCR {key(page)} {result['rec_seconds']:.2f}s {len(result['text'])} chars")

    failures = []

    def failed(page, phase):
        error = {"page": page, "phase": phase, "error": traceback.format_exc()}
        failures.append(error)
        write_json(root / "errors.json", existing_errors + failures)
        log(f"ERROR {key(page)} {phase}: {error['error']}")

    if staged:
        for page in pages:
            try:
                segment(page)
            except Exception:
                failed(page, "segmentation")
        for page in pages:
            try:
                geometry_path, _ = paths(page)
                if not geometry_path.exists():
                    continue
                image, _ = render(page)
                recognition(page, image, read_json(geometry_path))
            except Exception:
                failed(page, "recognition")
    else:
        for page in pages:
            try:
                image, checkpoint, segmentation = segment(page)
                recognition(page, image, checkpoint, segmentation)
            except Exception:
                failed(page, "stock")
    summary = {
        "pages": len(pages), "completed": len(results), "errors": len(failures),
        "staged": staged, "setup_seconds": setup_seconds,
        "elapsed_seconds": time.perf_counter() - start,
        "seg_seconds": sum(p["seg_seconds"] for p in results),
        "rec_seconds": sum(p["rec_seconds"] for p in results),
    }
    write_json(root / ("resume-summary.json" if resume_check else "summary.json"), summary)
    log(json.dumps(summary))
    if failures or len(results) != len(pages):
        raise RuntimeError("Incomplete inference; inspect errors and resume")
    error_path = root / "errors.json"
    if error_path.exists():
        # Retain the original attempt's errors, but clear the active error marker
        # only after every page has completed successfully on this resumed Job.
        previous = read_json(error_path)
        archive = root / "errors-before-successful-resume.json"
        if archive.exists():
            previous = read_json(archive) + previous
        write_json(archive, previous)
        assert read_json(archive) == previous
        error_path.unlink()


def main():
    import torch
    from huggingface_hub import hf_hub_download

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--shard", type=int, choices=range(4), default=0)
    parser.add_argument("--run-id", default="v1")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--bucket", type=Path, default=Path("/bucket"))
    args = parser.parse_args()
    assert torch.cuda.is_available(), "GPU acceleration is required"
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    provenance = {
        "python": sys.version, "platform": platform.platform(), "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__, "cuda": torch.version.cuda, "kraken_commit": KRAKEN_REV,
        "model": MODEL, "model_revision": MODEL_REV, "model_sha256": WEIGHTS_SHA,
        "dataset": DATASET, "dataset_revision": DATA_REV, "image": IMAGE,
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
        "source_sha256": digest(Path(__file__)), "lock_sha256": digest(Path("uv.lock")),
        "cpu_count": os.cpu_count(), "torch_threads": torch.get_num_threads(),
    }
    run_root = args.bucket / "runs" / args.run_id
    suffix = "verification" if args.verify_only else "environment"
    write_json(run_root / f"{args.mode}-{args.shard}-{suffix}.json", provenance)
    log(f"GPU verified: {provenance['gpu']}; torch {torch.__version__}")
    pages = manifest(args.bucket)
    weights = Path(hf_hub_download(MODEL, "medium.safetensors", revision=MODEL_REV))
    assert digest(weights) == WEIGHTS_SHA, "Recognition weights hash mismatch"
    if args.mode == "smoke":
        pages = [p for p in pages if p["smoke"]]
        if not args.verify_only:
            run_pages(pages, run_root / "stock", weights, staged=False)
            import gc

            gc.collect()
            torch.cuda.empty_cache()
            run_pages(pages, run_root / "staged", weights, staged=True)
        probe = run_root / f"io-probe-{time.time_ns()}.json"
        write_json(probe, {"round_trip": True})
        assert read_json(probe) == {"round_trip": True}
        comparisons = []
        for page in pages:
            def artifact(variant, suffix):
                return read_json(Path(str(run_root / variant / "pages" / key(page)) + suffix))
            comparisons.append({
                "page": page,
                "geometry_equal": canonical_geometry(artifact("stock", ".geometry.json")["geometry"])
                == canonical_geometry(artifact("staged", ".geometry.json")["geometry"]),
                "text_equal": artifact("stock", ".result.json")["text"]
                == artifact("staged", ".result.json")["text"],
            })
        write_json(run_root / "equivalence.json", comparisons)
        assert all(p["geometry_equal"] and p["text_equal"] for p in comparisons)
        log("PASS: all 21 pages have identical ordered geometry and raw text")
        run_pages(pages, run_root / "staged", weights, staged=True, resume_check=True)
        log("PASS: all 21 completed pages resume without calling either prediction model")
    else:
        assert not args.verify_only, "Verification-only mode requires a smoke run"
        pages = [p for p in pages if p["shard"] == args.shard]
        run_pages(pages, run_root / f"shard-{args.shard}", weights, staged=True)


if __name__ == "__main__":
    main()
