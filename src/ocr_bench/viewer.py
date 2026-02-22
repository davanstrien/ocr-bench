"""Gradio results viewer — browse leaderboard and pairwise comparisons."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from datasets import load_dataset

if TYPE_CHECKING:
    import gradio as gr


def load_results(repo_id: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load leaderboard and comparisons from a Hub results dataset.

    Returns:
        (leaderboard_rows, comparison_rows)
    """
    leaderboard_ds = load_dataset(repo_id, name="leaderboard", split="train")
    leaderboard_rows = [dict(row) for row in leaderboard_ds]

    comparisons_ds = load_dataset(repo_id, name="comparisons", split="train")
    comparison_rows = [dict(row) for row in comparisons_ds]

    return leaderboard_rows, comparison_rows


def _filter_comparisons(
    comparisons: list[dict[str, Any]],
    winner_filter: str,
    model_filter: str,
) -> list[dict[str, Any]]:
    """Filter comparison rows by winner and model."""
    filtered = comparisons
    if winner_filter and winner_filter != "All":
        filtered = [c for c in filtered if c.get("winner") == winner_filter]
    if model_filter and model_filter != "All":
        filtered = [
            c
            for c in filtered
            if c.get("model_a") == model_filter or c.get("model_b") == model_filter
        ]
    return filtered


def _winner_badge(winner: str) -> str:
    """Return a badge string for the winner."""
    if winner == "A":
        return "Winner: A"
    elif winner == "B":
        return "Winner: B"
    else:
        return "Tie"


def build_viewer(repo_id: str) -> gr.Blocks:
    """Build the Gradio app with leaderboard and comparison browser tabs."""
    import gradio as gr

    leaderboard_rows, comparison_rows = load_results(repo_id)

    # Extract unique models for filter dropdown
    models = sorted(
        {c.get("model_a", "") for c in comparison_rows}
        | {c.get("model_b", "") for c in comparison_rows}
    )

    with gr.Blocks(title=f"OCR Bench — {repo_id}") as app:
        gr.Markdown(f"# OCR Bench Results\n**Dataset**: `{repo_id}`")

        with gr.Tab("Leaderboard"):
            headers = ["model", "elo", "wins", "losses", "ties", "win_pct"]
            table_data = [[row.get(h, "") for h in headers] for row in leaderboard_rows]
            gr.Dataframe(
                value=table_data,
                headers=["Model", "ELO", "Wins", "Losses", "Ties", "Win%"],
                interactive=False,
            )

        with gr.Tab("Browse Comparisons"):
            # Pre-compute initial display from first comparison
            first = comparison_rows[0] if comparison_rows else {}
            init_model_a = f"{first.get('model_a', '')} ({first.get('col_a', '')})" if first else ""
            init_model_b = f"{first.get('model_b', '')} ({first.get('col_b', '')})" if first else ""

            with gr.Row():
                winner_dd = gr.Dropdown(
                    choices=["All", "A", "B", "tie"],
                    value="All",
                    label="Filter by winner",
                )
                model_dd = gr.Dropdown(
                    choices=["All", *models],
                    value="All",
                    label="Filter by model",
                )

            status_text = gr.Markdown(f"**{len(comparison_rows)} comparisons**")
            comp_slider = gr.Slider(
                minimum=0,
                maximum=max(len(comparison_rows) - 1, 0),
                step=1,
                value=0,
                label="Comparison index",
            )

            with gr.Row():
                model_a_label = gr.Textbox(
                    label="Model A", value=init_model_a, interactive=False
                )
                model_b_label = gr.Textbox(
                    label="Model B", value=init_model_b, interactive=False
                )

            with gr.Row():
                text_a_box = gr.Textbox(
                    label="OCR Output A",
                    value=first.get("text_a", ""),
                    lines=12,
                    interactive=False,
                )
                text_b_box = gr.Textbox(
                    label="OCR Output B",
                    value=first.get("text_b", ""),
                    lines=12,
                    interactive=False,
                )

            with gr.Row():
                winner_box = gr.Textbox(
                    label="Verdict",
                    value=_winner_badge(first.get("winner", "tie")) if first else "",
                    interactive=False,
                )
                agreement_box = gr.Textbox(
                    label="Agreement",
                    value=first.get("agreement", ""),
                    interactive=False,
                )

            reason_box = gr.Textbox(
                label="Reason", value=first.get("reason", ""), lines=3, interactive=False
            )

            # State to hold filtered list
            filtered_state = gr.State(comparison_rows)

            def apply_filters(winner_filter: str, model_filter: str):
                filtered = _filter_comparisons(comparison_rows, winner_filter, model_filter)
                n = len(filtered)
                if n == 0:
                    return (
                        filtered,
                        gr.Slider(maximum=0, value=0),
                        "**0 comparisons** (no matches)",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                    )
                first = filtered[0]
                return (
                    filtered,
                    gr.Slider(maximum=n - 1, value=0),
                    f"**{n} comparisons**",
                    f"{first.get('model_a', '')} ({first.get('col_a', '')})",
                    f"{first.get('model_b', '')} ({first.get('col_b', '')})",
                    first.get("text_a", ""),
                    first.get("text_b", ""),
                    _winner_badge(first.get("winner", "tie")),
                    first.get("agreement", ""),
                    first.get("reason", ""),
                )

            def show_comparison(idx: int, filtered: list[dict]):
                idx = int(idx)
                if not filtered or idx >= len(filtered):
                    return "", "", "", "", "", "", ""
                c = filtered[idx]
                return (
                    f"{c.get('model_a', '')} ({c.get('col_a', '')})",
                    f"{c.get('model_b', '')} ({c.get('col_b', '')})",
                    c.get("text_a", ""),
                    c.get("text_b", ""),
                    _winner_badge(c.get("winner", "tie")),
                    c.get("agreement", ""),
                    c.get("reason", ""),
                )

            for dd in [winner_dd, model_dd]:
                dd.change(
                    fn=apply_filters,
                    inputs=[winner_dd, model_dd],
                    outputs=[
                        filtered_state,
                        comp_slider,
                        status_text,
                        model_a_label,
                        model_b_label,
                        text_a_box,
                        text_b_box,
                        winner_box,
                        agreement_box,
                        reason_box,
                    ],
                )

            comp_slider.change(
                fn=show_comparison,
                inputs=[comp_slider, filtered_state],
                outputs=[
                    model_a_label,
                    model_b_label,
                    text_a_box,
                    text_b_box,
                    winner_box,
                    agreement_box,
                    reason_box,
                ],
            )

    return app


def launch_viewer(repo_id: str, **kwargs) -> None:
    """Launch the Gradio app.

    Args:
        repo_id: HF dataset repo id with published results.
        **kwargs: Passed to ``gr.Blocks.launch()`` (e.g. server_port, share).
    """
    app = build_viewer(repo_id)
    app.launch(**kwargs)
