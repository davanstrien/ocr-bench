"""Pairwise VLM judge — prompt templates, structured output schema, comparison building."""

from __future__ import annotations

import base64
import io
import json
import logging
import random
from dataclasses import dataclass
from itertools import combinations
from typing import Any

from PIL import Image

logger = logging.getLogger(__name__)

# --- Judge prompt ---

PAIRWISE_PROMPT = """\
You are an expert OCR quality evaluator. You are given a document image and \
TWO OCR outputs (A and B) extracted from that same image.

Compare them and decide which extraction is better overall.

Evaluation criteria (in priority order):

1. Faithfulness: The output must ONLY contain text from the document. Any added \
commentary, interpretation, or notes (e.g. "it appears the text says...", \
"the document contains...") is a serious error. Penalize heavily.

2. Completeness: ALL visible text must be captured — headers, footers, marginalia, \
stamps, handwritten notes. Missing any section of text is a significant penalty.

3. Accuracy: Correct characters, no hallucinated words or garbled text.

4. Reading order: Text flows naturally as a human would read the document.

5. Formatting: Clean structure. Ignore bounding box tags like <|ref|> <|det|> \
if present. Do NOT prefer fancier markdown formatting — plain accurate text is \
better than nicely formatted but incomplete text.

If both outputs capture the same text with similar accuracy, respond with "tie". \
Only pick a winner when there is a clear quality difference.

Output A:
---
{ocr_text_a}
---

Output B:
---
{ocr_text_b}
---

Respond with JSON only (no markdown fences, no extra text):
{{"winner": "A", "reason": "brief explanation"}}
Use "A", "B", or "tie" for the winner field."""

JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "winner": {"type": "string", "enum": ["A", "B", "tie"]},
        "reason": {"type": "string"},
    },
    "required": ["winner", "reason"],
}

# Max characters of OCR text to include per output in the prompt.
MAX_OCR_TEXT_LENGTH = 2500

# Max image dimension (longer side) before resizing.
MAX_IMAGE_DIM = 1024


# --- Image helpers ---


def image_to_base64(image: Image.Image, max_dim: int = MAX_IMAGE_DIM) -> str:
    """Convert a PIL image to a base64-encoded PNG string, resizing if needed."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    if max(image.size) > max_dim:
        ratio = max_dim / max(image.size)
        new_size = (int(image.width * ratio), int(image.height * ratio))
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# --- Comparison ---


@dataclass
class Comparison:
    """A single pairwise comparison to evaluate."""

    sample_idx: int
    model_a: str
    model_b: str
    col_a: str
    col_b: str
    swapped: bool
    messages: list[dict[str, Any]]


def build_prompt(text_a: str, text_b: str, swapped: bool) -> tuple[str, bool]:
    """Build the pairwise comparison prompt, applying position-bias swap.

    Returns (prompt_text, swapped).
    """
    a = text_a[:MAX_OCR_TEXT_LENGTH]
    b = text_b[:MAX_OCR_TEXT_LENGTH]
    if swapped:
        a, b = b, a
    return PAIRWISE_PROMPT.format(ocr_text_a=a, ocr_text_b=b), swapped


def build_messages(image_b64: str, prompt: str) -> list[dict[str, Any]]:
    """Build chat messages for the judge (image + prompt)."""
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]


def build_comparisons(
    dataset: Any,
    ocr_columns: dict[str, str],
    max_samples: int | None = None,
    seed: int = 42,
) -> list[Comparison]:
    """Build all pairwise comparison prompts from a dataset.

    Args:
        dataset: HF dataset with an "image" column and OCR output columns.
        ocr_columns: Mapping of column_name -> model_name.
        max_samples: If set, randomly sample this many rows.
        seed: Random seed for sampling and position-bias randomization.

    Returns:
        List of Comparison objects with pre-built chat messages.
    """
    col_names = list(ocr_columns.keys())
    model_names = list(ocr_columns.values())
    pairs = list(combinations(range(len(col_names)), 2))

    indices = list(range(len(dataset)))
    if max_samples and max_samples < len(indices):
        random.seed(seed)
        indices = random.sample(indices, max_samples)

    rng = random.Random(seed)
    comparisons: list[Comparison] = []

    for idx in indices:
        row = dataset[idx]
        image_b64 = image_to_base64(row["image"])

        for i, j in pairs:
            col_a, col_b = col_names[i], col_names[j]
            text_a = row[col_a] or ""
            text_b = row[col_b] or ""

            if not text_a.strip() or not text_b.strip():
                continue

            swapped = rng.random() < 0.5
            prompt, swapped = build_prompt(text_a, text_b, swapped)
            messages = build_messages(image_b64, prompt)

            comparisons.append(
                Comparison(
                    sample_idx=idx,
                    model_a=model_names[i],
                    model_b=model_names[j],
                    col_a=col_a,
                    col_b=col_b,
                    swapped=swapped,
                    messages=messages,
                )
            )

    return comparisons


# --- Output parsing ---


def parse_judge_output(text: str) -> dict[str, str]:
    """Parse judge JSON output, handling markdown fences and invalid values.

    Returns dict with "winner" and "reason" keys, or empty dict on failure.
    """
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    try:
        result = json.loads(text)
        winner = result.get("winner", "tie").upper().strip()
        if winner == "TIE":
            winner = "tie"
        if winner not in ("A", "B", "tie"):
            winner = "tie"
        return {"winner": winner, "reason": result.get("reason", "")}
    except json.JSONDecodeError:
        logger.warning("Failed to parse judge output: %s", text[:200])
        return {}
