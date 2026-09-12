"""Build a category chart and bounded diagnostic examples from saved official results."""

import argparse
import json
from pathlib import Path

from runner import key, overlay, read_json, render, write_json


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", type=Path, default=Path("/bucket"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    args = parser.parse_args()
    root = args.bucket / "scores" / args.run_id
    summary = read_json(root / "summary.json")
    names = [
        "old_scans", "multi_column", "long_tiny_text", "headers_footers",
        "table_tests", "old_scans_math", "arxiv_math", "baseline",
    ]
    labels = [
        "Old scans", "Multiple columns", "Long / tiny text", "Headers / footers",
        "Tables", "Old scans: math", "ArXiv math", "Baseline checks",
    ]
    scores = [
        summary["categories"][n if n == "baseline" else f"{n}.jsonl"]["score"] * 100
        for n in names
    ]
    fig, ax = plt.subplots(figsize=(9, 5.5), layout="constrained")
    bars = ax.barh(labels, scores, color="#31688e", height=0.65)
    ax.invert_yaxis()
    ax.set_xlim(0, 108)
    ax.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.bar_label(bars, labels=[f"{v:.1f}%" for v in scores], padding=5)
    ax.set_xlabel("Official test pass rate")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="x", alpha=0.15)
    macro = summary["official_macro_score"] * 100
    qualifier = "Full benchmark" if args.mode == "full" else "21-page smoke — not a benchmark result"
    fig.suptitle(f"Kraken BLLA + PP-OCRv6 medium\n{qualifier}", fontsize=15)
    low, high = [x * 100 for x in summary["official_ci_95"]]
    ax.set_title(
        f"Overall {macro:.1f}% · 95% CI {low:.1f}–{high:.1f}% · "
        f"{summary['pages']:,} pages\n300 DPI RGB · raw line text · one candidate",
        fontsize=10, pad=15,
    )
    fig.savefig(root / "category-scores.png", dpi=180)
    fig.savefig(root / "category-scores.svg")
    plt.close(fig)

    records = [json.loads(line) for line in (root / "test-results.jsonl").read_text().splitlines()]
    pages = read_json(args.bucket / "manifest.json")
    page_map = {(p["pdf"], p["page"]): p for p in pages}
    chosen = {}
    # A fixed rule selects the first failure of each test type, plus one passed test.
    # This is diagnostic selection after scoring, never a change to the scored set.
    for record in records:
        group = f"failed-{record['type']}" if not record["passed"] else "passed-example"
        if group not in chosen:
            chosen[group] = record
    for label, record in sorted(chosen.items()):
        page = page_map[(record["pdf"], record["page"])]
        variant = "staged" if args.mode == "smoke" else f"shard-{page['shard']}"
        source = args.bucket / "runs" / args.run_id / variant
        geometry = read_json(Path(str(source / "pages" / key(page)) + ".geometry.json"))
        result = read_json(Path(str(source / "pages" / key(page)) + ".result.json"))
        image, _ = render(page)
        folder = root / "examples" / label
        folder.mkdir(parents=True, exist_ok=True)
        overlay(image, geometry["geometry"], folder / "lines.png")
        image.thumbnail((1400, 1800))
        image.save(folder / "page.png")
        (folder / "prediction.txt").write_text(result["text"])
        write_json(folder / "test.json", record)
    print(f"Saved category chart and {len(chosen)} diagnostic examples to {root}", flush=True)


if __name__ == "__main__":
    main()
