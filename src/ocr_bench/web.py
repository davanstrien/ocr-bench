"""FastAPI + HTMX viewer — unified browse + validate for OCR bench results."""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from ocr_bench.validate import (
    ValidationComparison,
    build_validation_comparisons,
    compute_agreement,
    compute_human_elo,
    load_annotations,
    save_annotations,
)
from ocr_bench.viewer import (
    ImageLoader,
    _filter_comparisons,
    _load_source_metadata,
    load_results,
)

logger = structlog.get_logger()


def _short_model(model: str) -> str:
    """Return just the model name after the org prefix."""
    return model.split("/")[-1] if "/" in model else model


def _build_pair_summary_html(comparisons: list[dict[str, Any]]) -> str:
    """Build a compact HTML table of head-to-head records."""
    from collections import Counter

    pair_counts: dict[tuple[str, str], Counter[str]] = {}
    for c in comparisons:
        ma = c.get("model_a", "")
        mb = c.get("model_b", "")
        winner = c.get("winner", "tie")
        key = (ma, mb) if ma <= mb else (mb, ma)
        if key not in pair_counts:
            pair_counts[key] = Counter()
        if winner == "A":
            actual_winner = ma
        elif winner == "B":
            actual_winner = mb
        else:
            actual_winner = "tie"
        if actual_winner == key[0]:
            pair_counts[key]["W"] += 1
        elif actual_winner == key[1]:
            pair_counts[key]["L"] += 1
        else:
            pair_counts[key]["T"] += 1

    if not pair_counts:
        return ""

    rows = []
    for (ma, mb), counts in sorted(pair_counts.items()):
        short_a = _short_model(ma)
        short_b = _short_model(mb)
        wins, losses, ties = counts["W"], counts["L"], counts["T"]
        rows.append(
            f"<tr><td>{short_a}</td><td>{short_b}</td>"
            f"<td class='num'>{wins}</td><td class='num'>{losses}</td>"
            f"<td class='num'>{ties}</td></tr>"
        )
    return (
        '<table class="pair-table"><thead><tr>'
        "<th>Model A</th><th>Model B</th>"
        '<th class="num">W</th><th class="num">L</th><th class="num">T</th>'
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )


PKG_DIR = Path(__file__).parent
TEMPLATES_DIR = PKG_DIR / "templates"
STATIC_DIR = PKG_DIR / "static"


@dataclass
class ViewerState:
    """In-memory state for the single-user viewer."""

    repo_id: str
    leaderboard_rows: list[dict[str, Any]]
    comparison_rows: list[dict[str, Any]]
    validation_comps: list[ValidationComparison]
    models: list[str]
    img_loader: ImageLoader | None
    save_path: str
    annotations: list[dict[str, Any]] = field(default_factory=list)
    completed_ids: set[int] = field(default_factory=set)
    filtered_indices: list[int] = field(default_factory=list)


def _build_filtered_indices(
    state: ViewerState,
    winner_filter: str = "All",
    model_filter: str = "All",
) -> list[int]:
    """Map nav indices to validation_comps indices, respecting filters."""
    filtered_comps = _filter_comparisons(state.comparison_rows, winner_filter, model_filter)
    # Build a lookup from (sample_idx, model_a, model_b) -> validation comp index
    filtered_sample_keys = {
        (c["sample_idx"], c["model_a"], c["model_b"]) for c in filtered_comps
    }
    return [
        i
        for i, vc in enumerate(state.validation_comps)
        if (vc.sample_idx, vc.model_a, vc.model_b) in filtered_sample_keys
    ]


def create_app(
    repo_id: str,
    *,
    output_path: str | None = None,
    n_validate: int | None = None,
) -> FastAPI:
    """Create the FastAPI app with all routes.

    Args:
        repo_id: HF dataset repo with published judge results.
        output_path: Path to save human annotations JSON.
        n_validate: Max comparisons to include for validation (None = all).
    """
    app = FastAPI(title=f"OCR Bench — {repo_id}")
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

    # --- Load data ---
    leaderboard_rows, comparison_rows = load_results(repo_id)

    metadata = _load_source_metadata(repo_id)
    source_dataset = metadata.get("source_dataset", "")
    from_prs = metadata.get("from_prs", False)

    img_loader: ImageLoader | None = None
    if source_dataset:
        img_loader = ImageLoader(source_dataset, from_prs=from_prs)

    validation_comps = build_validation_comparisons(
        comparison_rows,
        leaderboard_rows=leaderboard_rows,
        n=n_validate,
        prioritize_splits=True,
    )

    models = sorted(
        {c.get("model_a", "") for c in comparison_rows}
        | {c.get("model_b", "") for c in comparison_rows}
    )

    slug = repo_id.replace("/", "-")
    save_path = output_path or f"human-eval-{slug}.json"

    # Resume existing annotations
    _, existing_annotations = load_annotations(save_path)
    completed_ids = {ann["comparison_id"] for ann in existing_annotations}

    state = ViewerState(
        repo_id=repo_id,
        leaderboard_rows=leaderboard_rows,
        comparison_rows=comparison_rows,
        validation_comps=validation_comps,
        models=models,
        img_loader=img_loader,
        save_path=save_path,
        annotations=existing_annotations,
        completed_ids=completed_ids,
        filtered_indices=list(range(len(validation_comps))),
    )

    # Store state on app for access in routes
    app.state.viewer = state

    ann_metadata = {
        "results_repo": repo_id,
        "n_comparisons": len(validation_comps),
        "models": models,
        "started_at": datetime.now(UTC).isoformat(),
    }

    # --- Helpers ---

    def _get_comp_context(
        nav_idx: int,
        *,
        revealed: bool = False,
        voted: bool = False,
        human_vote: str = "",
        winner_filter: str = "All",
        model_filter: str = "All",
    ) -> dict[str, Any]:
        """Build template context for a comparison card."""
        indices = state.filtered_indices
        if nav_idx < 0 or nav_idx >= len(indices):
            return {"comp": None, "nav_idx": nav_idx, "nav_total": len(indices)}

        comp_idx = indices[nav_idx]
        comp = state.validation_comps[comp_idx]

        # Check if already voted
        already_voted = comp.comparison_id in state.completed_ids
        if already_voted:
            voted = True
            revealed = True
            # Find the annotation to get human vote
            for ann in state.annotations:
                if ann["comparison_id"] == comp.comparison_id:
                    human_vote = ann["winner"]
                    break

        # Model names — short form for clean headers
        model_a_name = _short_model(comp.model_a)
        model_b_name = _short_model(comp.model_b)
        if comp.swapped:
            model_a_name, model_b_name = model_b_name, model_a_name

        # Judge verdict (canonical → display)
        judge_winner = comp.winner
        if comp.swapped:
            if judge_winner == "A":
                judge_verdict = "B"
            elif judge_winner == "B":
                judge_verdict = "A"
            else:
                judge_verdict = "tie"
        else:
            judge_verdict = judge_winner

        # Agreement
        agreement_word = ""
        agreement_class = ""
        if voted and human_vote:
            # Unswap human vote for comparison
            unswapped_human = human_vote
            if comp.swapped:
                if human_vote == "A":
                    unswapped_human = "B"
                elif human_vote == "B":
                    unswapped_human = "A"

            if unswapped_human == comp.winner:
                agreement_word = "agreed"
                agreement_class = "agreed"
            elif unswapped_human == "tie" or comp.winner == "tie":
                agreement_word = "soft disagree"
                agreement_class = "soft-disagree"
            else:
                agreement_word = "hard disagree"
                agreement_class = "hard-disagree"

        has_image = img_loader is not None

        return {
            "comp": comp,
            "comp_idx": comp_idx,
            "nav_idx": nav_idx,
            "nav_total": len(indices),
            "revealed": revealed,
            "voted": voted,
            "display_text_a": comp.display_text_a,
            "display_text_b": comp.display_text_b,
            "model_a_name": model_a_name,
            "model_b_name": model_b_name,
            "judge_verdict": judge_verdict,
            "human_vote": human_vote,
            "agreement_word": agreement_word,
            "agreement_class": agreement_class,
            "reason": comp.reason,
            "sample_idx": comp.sample_idx,
            "has_image": has_image,
            "winner_filter": winner_filter,
            "model_filter": model_filter,
        }

    def _stats_context() -> dict[str, Any]:
        """Build template context for the stats panel."""
        stats = compute_agreement(state.annotations, state.validation_comps)
        return {
            "vote_count": stats.total,
            "agreement_pct": round(stats.agreement_rate * 100) if stats.total else 0,
            "hard_disagree_rate": round(stats.hard_disagree_rate * 100) if stats.total else 0,
        }

    def _nav_idx_for_comp_idx(comp_idx: int) -> int:
        """Find the nav_idx for a given comp_idx in filtered_indices."""
        try:
            return state.filtered_indices.index(comp_idx)
        except ValueError:
            return 0

    # --- Routes ---

    @app.get("/", response_class=RedirectResponse)
    async def index():
        return RedirectResponse(url="/leaderboard", status_code=302)

    @app.get("/leaderboard", response_class=HTMLResponse)
    async def leaderboard(request: Request):
        from ocr_bench.publish import _get_model_sizes

        # Build human ELO if we have annotations
        human_board = compute_human_elo(state.annotations, state.validation_comps)
        sizes = _get_model_sizes()

        rows = []
        for row in sorted(state.leaderboard_rows, key=lambda r: r.get("elo", 0), reverse=True):
            model = row.get("model", "")
            short = model.split("/")[-1] if "/" in model else model
            human_elo = None
            human_win_pct = None
            if human_board and model in human_board.elo:
                human_elo = round(human_board.elo[model])
                wp = human_board.win_pct(model)
                human_win_pct = f"{wp:.0f}" if wp is not None else None

            rows.append({
                "model": model,
                "model_short": short,
                "params": row.get("params") or sizes.get(model, ""),
                "elo": round(row.get("elo", 0)),
                "elo_low": row.get("elo_low"),
                "elo_high": row.get("elo_high"),
                "wins": row.get("wins", 0),
                "losses": row.get("losses", 0),
                "ties": row.get("ties", 0),
                "win_pct": row.get("win_pct", 0),
                "human_elo": human_elo,
                "human_win_pct": human_win_pct,
            })

        has_ci = any(r.get("elo_low") is not None for r in rows)

        # Build chart data — params (numeric) vs win%
        chart_points = []
        for r in rows:
            params_str = r.get("params", "")
            if params_str:
                try:
                    params_num = float(params_str.rstrip("B"))
                except ValueError:
                    continue
                chart_points.append({
                    "name": r["model_short"],
                    "params": params_num,
                    "win_pct": r["win_pct"],
                    "elo": r["elo"],
                    "elo_low": r.get("elo_low"),
                    "elo_high": r.get("elo_high"),
                })

        return templates.TemplateResponse(request, "leaderboard.html", {
            "active_tab": "leaderboard",
            "repo_id": state.repo_id,
            "rows": rows,
            "has_ci": has_ci,
            "has_human_elo": human_board is not None,
            "chart_points": chart_points,
        })

    @app.get("/comparisons", response_class=HTMLResponse)
    async def comparisons_page(request: Request):
        state.filtered_indices = _build_filtered_indices(state)
        pair_summary = _build_pair_summary_html(state.comparison_rows)
        ctx = _get_comp_context(0)
        stats = _stats_context()
        return templates.TemplateResponse(request, "comparisons.html", {
            "active_tab": "comparisons",
            "models": state.models,
            "pair_summary": pair_summary,
            "winner_filter": "All",
            "model_filter": "All",
            **ctx,
            **stats,
        })

    @app.get("/comparisons/filter", response_class=HTMLResponse)
    async def comparisons_filter(
        request: Request,
        winner: str = "All",
        model: str = "All",
    ):
        state.filtered_indices = _build_filtered_indices(state, winner, model)
        ctx = _get_comp_context(0, winner_filter=winner, model_filter=model)
        return templates.TemplateResponse(request, "comparison_card.html", ctx)

    @app.get("/comparisons/{nav_idx}", response_class=HTMLResponse)
    async def comparison_at(
        request: Request,
        nav_idx: int,
        winner: str = "All",
        model: str = "All",
    ):
        # Clamp nav_idx
        nav_idx = max(0, min(nav_idx, len(state.filtered_indices) - 1))
        ctx = _get_comp_context(nav_idx, winner_filter=winner, model_filter=model)
        return templates.TemplateResponse(request, "comparison_card.html", ctx)

    @app.post("/vote/{comp_idx}", response_class=HTMLResponse)
    async def vote(request: Request, comp_idx: int, winner: str = Form(...)):
        if comp_idx < 0 or comp_idx >= len(state.validation_comps):
            return HTMLResponse("Invalid comparison", status_code=404)

        comp = state.validation_comps[comp_idx]

        # Idempotent: if already voted, just return revealed card
        if comp.comparison_id not in state.completed_ids:
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

            state.annotations.append(ann)
            state.completed_ids.add(comp.comparison_id)
            save_annotations(state.save_path, ann_metadata, state.annotations)

        nav_idx = _nav_idx_for_comp_idx(comp_idx)
        # Read current filters from request query params (forwarded by htmx)
        winner_filter = request.query_params.get("winner", "All")
        model_filter = request.query_params.get("model", "All")

        ctx = _get_comp_context(
            nav_idx,
            revealed=True,
            voted=True,
            human_vote=winner,
            winner_filter=winner_filter,
            model_filter=model_filter,
        )
        # Auto-advance: tell template this was a fresh vote
        next_nav = nav_idx + 1 if nav_idx + 1 < len(state.filtered_indices) else None
        ctx["just_voted"] = True
        ctx["next_nav_idx"] = next_nav
        ctx["next_url"] = (
            f"/comparisons/{next_nav}"
            + (f"?winner={winner_filter}" if winner_filter != "All" else "")
            + (
                f"{'&' if winner_filter != 'All' else '?'}model={model_filter}"
                if model_filter != "All"
                else ""
            )
            if next_nav is not None
            else None
        )
        response = templates.TemplateResponse(request, "comparison_card.html", ctx)
        response.headers["HX-Trigger"] = "vote-recorded"
        return response

    @app.get("/reveal/{comp_idx}", response_class=HTMLResponse)
    async def reveal(request: Request, comp_idx: int):
        if comp_idx < 0 or comp_idx >= len(state.validation_comps):
            return HTMLResponse("Invalid comparison", status_code=404)

        nav_idx = _nav_idx_for_comp_idx(comp_idx)
        winner_filter = request.query_params.get("winner", "All")
        model_filter = request.query_params.get("model", "All")

        ctx = _get_comp_context(
            nav_idx,
            revealed=True,
            voted=False,
            winner_filter=winner_filter,
            model_filter=model_filter,
        )
        return templates.TemplateResponse(request, "comparison_card.html", ctx)

    @app.get("/stats", response_class=HTMLResponse)
    async def stats(request: Request):
        ctx = _stats_context()
        return templates.TemplateResponse(request, "stats_panel.html", ctx)

    @app.get("/image/{sample_idx}")
    async def image(sample_idx: int):
        if img_loader is None:
            return HTMLResponse("No images available", status_code=404)
        img = img_loader.get(sample_idx)
        if img is None:
            return HTMLResponse("Image not found", status_code=404)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/png")

    return app
