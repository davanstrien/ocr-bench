"""Blind human A/B validation for OCR judge quality."""

from __future__ import annotations

import json
import os
import random
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import structlog
from datasets import load_dataset

if TYPE_CHECKING:
    import gradio as gr

logger = structlog.get_logger()

# Confidence thresholds
MIN_ANNOTATIONS_FOR_CONFIDENCE = 15
HIGH_AGREEMENT_THRESHOLD = 0.75


@dataclass
class AgreementStats:
    """Tracks agreement between human and VLM judge."""

    agree: int = 0
    soft_disagree: int = 0  # one picks tie, other picks winner
    hard_disagree: int = 0  # both pick winners but opposite
    total: int = 0

    @property
    def agreement_rate(self) -> float:
        """Rate including soft disagreements as partial agreement."""
        return (self.agree + self.soft_disagree) / self.total if self.total else 0.0

    @property
    def hard_disagree_rate(self) -> float:
        return self.hard_disagree / self.total if self.total else 0.0


@dataclass
class ValidationComparison:
    """A single comparison for human validation.

    Built from enriched comparison data published by the judge.
    """

    comparison_id: int
    sample_idx: int
    model_a: str
    model_b: str
    winner: str  # judge's verdict (hidden during annotation)
    reason: str
    agreement: str  # jury agreement (e.g. "2/2")
    text_a: str  # OCR text from model A
    text_b: str  # OCR text from model B
    col_a: str
    col_b: str
    swapped: bool  # position-bias randomization for human display
    display_text_a: str = ""  # text shown to human (may be swapped)
    display_text_b: str = ""


@dataclass
class ValidationSession:
    """Holds state for a validation session."""

    comparisons: list[ValidationComparison]
    model_names: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)
    annotations: list[dict[str, Any]] = field(default_factory=list)
    completed_ids: set[int] = field(default_factory=set)


def _is_split_jury(agreement: str) -> bool:
    """Check if a jury vote was split (e.g. '1/2' not '2/2')."""
    parts = agreement.split("/")
    return len(parts) == 2 and parts[0] != parts[1]


def _interleave_by_sample(
    comparisons: list[ValidationComparison],
) -> list[ValidationComparison]:
    """Interleave comparisons so you see different samples before repeating."""
    by_sample: dict[int, list[ValidationComparison]] = defaultdict(list)
    for comp in comparisons:
        by_sample[comp.sample_idx].append(comp)

    result: list[ValidationComparison] = []
    queues = list(by_sample.values())
    while queues:
        next_round = []
        for q in queues:
            result.append(q.pop(0))
            if q:
                next_round.append(q)
        queues = next_round
    return result


def build_validation_comparisons(
    comparison_rows: list[dict[str, Any]],
    *,
    n: int | None = None,
    prioritize_splits: bool = True,
    seed: int = 42,
) -> list[ValidationComparison]:
    """Build validation comparisons from published judge results.

    Args:
        comparison_rows: Rows from the comparisons config of a results dataset.
        n: Max number of comparisons to include (None = all).
        prioritize_splits: Show split-jury cases first (most informative).
        seed: Random seed for position-bias randomization.
    """
    rng = random.Random(seed)

    comps: list[ValidationComparison] = []
    for i, row in enumerate(comparison_rows):
        swapped = rng.random() < 0.5
        text_a = row.get("text_a", "")
        text_b = row.get("text_b", "")

        if swapped:
            display_a, display_b = text_b, text_a
        else:
            display_a, display_b = text_a, text_b

        comps.append(
            ValidationComparison(
                comparison_id=i,
                sample_idx=row.get("sample_idx", i),
                model_a=row.get("model_a", ""),
                model_b=row.get("model_b", ""),
                winner=row.get("winner", "tie"),
                reason=row.get("reason", ""),
                agreement=row.get("agreement", "1/1"),
                text_a=text_a,
                text_b=text_b,
                col_a=row.get("col_a", ""),
                col_b=row.get("col_b", ""),
                swapped=swapped,
                display_text_a=display_a,
                display_text_b=display_b,
            )
        )

    if prioritize_splits:
        splits = [c for c in comps if _is_split_jury(c.agreement)]
        unanimous = [c for c in comps if not _is_split_jury(c.agreement)]
        ordered = _interleave_by_sample(splits) + _interleave_by_sample(unanimous)
    else:
        ordered = _interleave_by_sample(comps)

    if n is not None and n < len(ordered):
        ordered = ordered[:n]

    # Re-assign comparison IDs after reordering
    return [
        ValidationComparison(
            comparison_id=i,
            sample_idx=c.sample_idx,
            model_a=c.model_a,
            model_b=c.model_b,
            winner=c.winner,
            reason=c.reason,
            agreement=c.agreement,
            text_a=c.text_a,
            text_b=c.text_b,
            col_a=c.col_a,
            col_b=c.col_b,
            swapped=c.swapped,
            display_text_a=c.display_text_a,
            display_text_b=c.display_text_b,
        )
        for i, c in enumerate(ordered)
    ]


def compute_agreement(
    annotations: list[dict[str, Any]],
    comparisons: list[ValidationComparison],
) -> AgreementStats:
    """Compute agreement between human annotations and judge verdicts."""
    comp_by_id = {c.comparison_id: c for c in comparisons}
    stats = AgreementStats()

    for ann in annotations:
        comp = comp_by_id.get(ann.get("comparison_id"))
        if not comp:
            continue

        # Unswap human vote
        human_winner = ann["winner"]
        if comp.swapped:
            if human_winner == "A":
                human_winner = "B"
            elif human_winner == "B":
                human_winner = "A"

        judge_winner = comp.winner
        stats.total += 1

        if human_winner == judge_winner:
            stats.agree += 1
        elif human_winner == "tie" or judge_winner == "tie":
            stats.soft_disagree += 1
        else:
            stats.hard_disagree += 1

    return stats


def save_annotations(
    path: str,
    metadata: dict[str, Any],
    annotations: list[dict[str, Any]],
) -> None:
    """Atomically save annotations to JSON file."""
    data = {"metadata": metadata, "annotations": annotations}
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def load_annotations(path: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load annotations from JSON file. Returns (metadata, annotations)."""
    if not os.path.exists(path):
        return {}, []
    with open(path) as f:
        data = json.load(f)
    return data.get("metadata", {}), data.get("annotations", [])


def _agreement_banner(stats: AgreementStats) -> str:
    """Format agreement stats for display."""
    if stats.total == 0:
        return ""

    parts = [f"Agree: {stats.agree}"]
    if stats.soft_disagree:
        parts.append(f"Soft: {stats.soft_disagree}")
    if stats.hard_disagree:
        parts.append(f"**Hard: {stats.hard_disagree}**")
    parts.append(f"(of {stats.total})")

    confidence = ""
    if stats.total >= MIN_ANNOTATIONS_FOR_CONFIDENCE:
        if stats.hard_disagree_rate == 0:
            confidence = (
                f" -- No hard disagreements after {stats.total} annotations. "
                "Judge rankings reliable for this domain."
            )
        elif stats.hard_disagree_rate <= 0.1:
            confidence = (
                f" -- Very few hard disagreements ({stats.hard_disagree}). "
                "Rankings likely trustworthy."
            )
        elif stats.hard_disagree_rate > 0.25:
            confidence = (
                f" -- Many hard disagreements ({stats.hard_disagree}/{stats.total}). "
                "Judge may not be calibrated for this content."
            )

    return f"Judge: {' | '.join(parts)}{confidence}"


def build_validation_app(
    results_repo: str,
    *,
    n: int = 30,
    output_path: str | None = None,
    prioritize_splits: bool = True,
) -> gr.Blocks:
    """Build the Gradio validation app.

    Args:
        results_repo: HF dataset repo with published judge results.
        n: Number of comparisons to validate.
        output_path: Path to save annotations JSON.
        prioritize_splits: Show split-jury cases first.
    """
    import gradio as gr

    # Load results from Hub
    comparisons_ds = load_dataset(results_repo, name="comparisons", split="train")
    comparison_rows = [dict(row) for row in comparisons_ds]

    comps = build_validation_comparisons(comparison_rows, n=n, prioritize_splits=prioritize_splits)

    if not comps:
        raise ValueError("No comparisons found in results dataset")

    model_names = sorted({c.model_a for c in comps} | {c.model_b for c in comps})

    slug = results_repo.replace("/", "-")
    save_path = output_path or f"human-eval-{slug}.json"

    metadata = {
        "results_repo": results_repo,
        "n_comparisons": len(comps),
        "prioritize_splits": prioritize_splits,
        "models": model_names,
        "started_at": datetime.now(UTC).isoformat(),
    }

    # Resume from existing annotations
    _, existing_annotations = load_annotations(save_path)
    initial_completed = {ann["comparison_id"] for ann in existing_annotations}
    initial_idx = 0
    while initial_idx < len(comps) and initial_idx in initial_completed:
        initial_idx += 1

    if existing_annotations:
        logger.info(
            "resuming_validation",
            n_existing=len(existing_annotations),
            starting_at=initial_idx,
        )

    n_splits = sum(1 for c in comps if _is_split_jury(c.agreement))
    initial_stats = compute_agreement(existing_annotations, comps)
    initial_banner = _agreement_banner(initial_stats)

    with gr.Blocks(title=f"OCR Validation — {results_repo}") as app:
        # State
        current_idx = gr.State(initial_idx)
        annotations_state = gr.State(existing_annotations)
        completed_ids_state = gr.State(initial_completed)

        gr.Markdown("# OCR Blind Human Validation")
        header = (
            f"**Results**: `{results_repo}` | "
            f"**Comparisons**: {len(comps)} | "
            f"**Models**: {len(model_names)}"
        )
        if n_splits:
            header += f" | **Split-jury first**: {n_splits}"
        gr.Markdown(header)

        with gr.Tab("Evaluate"):
            progress = gr.Markdown(f"Comparison {initial_idx + 1} / {len(comps)}")

            with gr.Row():
                feedback_display = gr.Markdown("")
                agreement_display = gr.Markdown(initial_banner)

            jury_hint_display = gr.Markdown("")

            with gr.Row():
                text_a_box = gr.Textbox(
                    label="Output A",
                    value=comps[initial_idx].display_text_a if initial_idx < len(comps) else "",
                    lines=20,
                    interactive=False,
                )
                text_b_box = gr.Textbox(
                    label="Output B",
                    value=comps[initial_idx].display_text_b if initial_idx < len(comps) else "",
                    lines=20,
                    interactive=False,
                )

            with gr.Row():
                btn_a = gr.Button("A is Better", variant="primary", scale=2)
                btn_tie = gr.Button("Tie", scale=1)
                btn_b = gr.Button("B is Better", variant="primary", scale=2)

            with gr.Row():
                btn_skip = gr.Button("Skip", scale=1)
                btn_undo = gr.Button("Undo Last", scale=1)

            def _load_comparison(idx: int) -> tuple[str, str, str, str]:
                """Return (text_a, text_b, progress, jury_hint)."""
                if idx >= len(comps):
                    return (
                        "All comparisons complete!",
                        "",
                        f"Done! {len(comps)}/{len(comps)} completed",
                        "",
                    )
                c = comps[idx]
                hint = "Jury was split on this one" if _is_split_jury(c.agreement) else ""
                return (
                    c.display_text_a,
                    c.display_text_b,
                    f"Comparison {idx + 1} / {len(comps)}",
                    hint,
                )

            def _find_next(idx: int, done: set[int]) -> int:
                while idx < len(comps) and idx in done:
                    idx += 1
                return idx

            def record_vote(
                winner: str,
                idx: int,
                anns: list[dict],
                done: set[int],
            ) -> tuple:
                if idx >= len(comps):
                    ta, tb, prog, hint = _load_comparison(idx)
                    return (
                        idx,
                        anns,
                        done,
                        ta,
                        tb,
                        prog,
                        hint,
                        "",
                        _agreement_banner(compute_agreement(anns, comps)),
                    )

                comp = comps[idx]

                # Unswap for storage
                winner_unswapped = winner
                if comp.swapped:
                    if winner == "A":
                        winner_unswapped = "B"
                    elif winner == "B":
                        winner_unswapped = "A"

                if winner_unswapped == "A":
                    winner_model = comp.model_a
                elif winner_unswapped == "B":
                    winner_model = comp.model_b
                else:
                    winner_model = "tie"

                ann = {
                    "comparison_id": comp.comparison_id,
                    "sample_idx": comp.sample_idx,
                    "model_a": comp.model_a,
                    "model_b": comp.model_b,
                    "swapped": comp.swapped,
                    "winner": winner,
                    "winner_model": winner_model,
                    "timestamp": datetime.now(UTC).isoformat(),
                }

                anns = anns + [ann]
                done = done | {idx}

                # Check agreement
                judge_winner = comp.winner
                human_winner = winner_unswapped
                # Map to same perspective
                if comp.swapped:
                    pass  # already unswapped above
                if human_winner == judge_winner:
                    feedback = "Judge agreed"
                elif human_winner == "tie" or judge_winner == "tie":
                    feedback = "Judge: soft disagree (tie vs winner)"
                else:
                    feedback = "Judge: **hard disagree** (opposite winners)"

                save_annotations(save_path, metadata, anns)

                next_idx = _find_next(idx + 1, done)
                ta, tb, prog, hint = _load_comparison(next_idx)
                stats = compute_agreement(anns, comps)
                return (
                    next_idx,
                    anns,
                    done,
                    ta,
                    tb,
                    prog,
                    hint,
                    feedback,
                    _agreement_banner(stats),
                )

            def skip_current(
                idx: int,
                anns: list[dict],
                done: set[int],
            ) -> tuple:
                next_idx = min(idx + 1, len(comps))
                ta, tb, prog, hint = _load_comparison(next_idx)
                return (
                    next_idx,
                    anns,
                    done,
                    ta,
                    tb,
                    prog,
                    hint,
                    "",
                    _agreement_banner(compute_agreement(anns, comps)),
                )

            def undo_last(
                idx: int,
                anns: list[dict],
                done: set[int],
            ) -> tuple:
                if not anns:
                    ta, tb, prog, hint = _load_comparison(idx)
                    return (
                        idx,
                        anns,
                        done,
                        ta,
                        tb,
                        prog,
                        hint,
                        "",
                        _agreement_banner(compute_agreement(anns, comps)),
                    )

                last = anns[-1]
                anns = anns[:-1]
                done = done - {last["comparison_id"]}
                back_idx = last["comparison_id"]
                save_annotations(save_path, metadata, anns)
                ta, tb, prog, hint = _load_comparison(back_idx)
                return (
                    back_idx,
                    anns,
                    done,
                    ta,
                    tb,
                    prog,
                    hint,
                    "Undid last annotation",
                    _agreement_banner(compute_agreement(anns, comps)),
                )

            outputs = [
                current_idx,
                annotations_state,
                completed_ids_state,
                text_a_box,
                text_b_box,
                progress,
                jury_hint_display,
                feedback_display,
                agreement_display,
            ]

            btn_a.click(
                fn=lambda idx, anns, done: record_vote("A", idx, anns, done),
                inputs=[current_idx, annotations_state, completed_ids_state],
                outputs=outputs,
            )
            btn_b.click(
                fn=lambda idx, anns, done: record_vote("B", idx, anns, done),
                inputs=[current_idx, annotations_state, completed_ids_state],
                outputs=outputs,
            )
            btn_tie.click(
                fn=lambda idx, anns, done: record_vote("tie", idx, anns, done),
                inputs=[current_idx, annotations_state, completed_ids_state],
                outputs=outputs,
            )
            btn_skip.click(
                fn=skip_current,
                inputs=[current_idx, annotations_state, completed_ids_state],
                outputs=outputs,
            )
            btn_undo.click(
                fn=undo_last,
                inputs=[current_idx, annotations_state, completed_ids_state],
                outputs=outputs,
            )

        with gr.Tab("Results"):
            gr.Markdown(
                "Click **Compute Results** to see agreement stats. "
                "Model identities are revealed here."
            )
            btn_compute = gr.Button("Compute Results", variant="primary")
            results_display = gr.Markdown("")

            def show_results(anns: list[dict]) -> str:
                if not anns:
                    return "No annotations yet."
                stats = compute_agreement(anns, comps)
                lines = [
                    f"### Agreement Summary ({stats.total} comparisons)\n",
                    "| | Count | % |",
                    "|---|---|---|",
                    f"| Agree | {stats.agree} | {stats.agree / stats.total:.0%} |"
                    if stats.total
                    else "",
                    (
                        f"| Soft disagree | {stats.soft_disagree}"
                        f" | {stats.soft_disagree / stats.total:.0%} |"
                    )
                    if stats.total
                    else "",
                    (
                        f"| **Hard disagree** | **{stats.hard_disagree}**"
                        f" | **{stats.hard_disagree / stats.total:.0%}** |"
                    )
                    if stats.total
                    else "",
                    "",
                    f"**Agreement rate**: {stats.agreement_rate:.0%}",
                ]
                if stats.hard_disagree_rate > 0.25:
                    lines.append(
                        "\n**Warning**: High hard disagreement rate. "
                        "Judge may not be well-calibrated for this document type."
                    )
                elif stats.total >= MIN_ANNOTATIONS_FOR_CONFIDENCE:
                    lines.append("\n**Judge appears well-calibrated** for this document type.")
                return "\n".join(lines)

            btn_compute.click(
                fn=show_results,
                inputs=[annotations_state],
                outputs=[results_display],
            )

    return app


def launch_validation(results_repo: str, **kwargs) -> None:
    """Launch the validation app.

    Args:
        results_repo: HF dataset repo with published results.
        **kwargs: n, output_path, prioritize_splits, server_port, share.
    """
    app_kwargs = {}
    for k in ("n", "output_path", "prioritize_splits"):
        if k in kwargs:
            app_kwargs[k] = kwargs.pop(k)
    app = build_validation_app(results_repo, **app_kwargs)
    app.launch(**kwargs)
