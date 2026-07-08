"""Pairwise VLM judge — prompt templates, structured output schema, comparison building."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import random
from dataclasses import dataclass
from itertools import combinations
from typing import Any

from PIL import Image

logger = logging.getLogger(__name__)

# --- Judge prompt profiles ---
#
# Named criteria profiles select the pairwise judge prompt. "default" is tuned
# for prose corpora — it deliberately neutralises structure so cosmetic markup
# can't win. "table-fidelity" keeps criteria 1-4 identical but swaps criterion 5
# for an explicit table criterion, because on table-dense corpora the main
# quality signal is whether a pipeline preserves row/column cell association
# rather than flattening it. Add a profile by adding a key here; the CLI exposes
# them via ``--criteria`` and records the chosen profile + a prompt hash in the
# results metadata so two boards judged under different prompts are
# distinguishable. See GitHub issue #44.

# The profile chosen when ``--criteria`` is not given (and the default template
# for direct callers of ``build_prompt`` / ``build_comparisons``).
DEFAULT_CRITERIA = "default"

_DEFAULT_PROMPT = """\
You are an expert OCR quality evaluator. You are given a document image and \
TWO OCR outputs (A and B) extracted from that same image.

Compare them and decide which extraction is better overall.

Evaluation criteria (in priority order):

1. Faithfulness: The output must ONLY contain text actually visible in the document. \
Hallucinating text that is not in the image (garbled strings, repeated tokens, \
nonsensical output) is the most serious error. Added commentary or notes \
(e.g. "it appears the text says...") is also an error, but less severe than \
hallucination. If a page is blank or has minimal text, saying so is acceptable — \
fabricating content is always worse.

2. Completeness: ALL visible text must be captured — headers, footers, marginalia, \
stamps, handwritten notes. Missing any section of text is a significant penalty.

3. Accuracy: Correct characters, no garbled or fabricated words.

4. Reading order: Text flows naturally as a human would read the document.

5. Formatting: Clean structure. Ignore bounding box tags like <|ref|> <|det|> \
if present. Markdown formatting markers (#, **, *, etc.) are neutral — do not \
penalise or reward their presence. Judge only on the actual text content, not \
on whether it is wrapped in markup. Plain text and markdown-formatted text that \
contain the same words are equivalent.

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

_TABLE_FIDELITY_PROMPT = """\
You are an expert OCR quality evaluator. You are given a document image and \
TWO OCR outputs (A and B) extracted from that same image.

Compare them and decide which extraction is better overall.

Evaluation criteria (in priority order):

1. Faithfulness: The output must ONLY contain text actually visible in the document. \
Hallucinating text that is not in the image (garbled strings, repeated tokens, \
nonsensical output) is the most serious error. Added commentary or notes \
(e.g. "it appears the text says...") is also an error, but less severe than \
hallucination. If a page is blank or has minimal text, saying so is acceptable — \
fabricating content is always worse.

2. Completeness: ALL visible text must be captured — headers, footers, marginalia, \
stamps, handwritten notes. Missing any section of text is a significant penalty.

3. Table fidelity: When the document contains tabular data, the extraction must \
preserve the table's row and column structure — each value must stay associated \
with its correct row and column headers. A flattened table that runs cells \
together so it is no longer clear which value belongs to which row or column is a \
significant error. Markup style is neutral: a well-aligned plain-text table \
(using spacing or simple separators) that preserves the cell relationships is just \
as good as a markdown or HTML table — judge the preserved row/column relationships, \
not the syntax used to express them.

4. Accuracy: Correct characters, no garbled or fabricated words.

5. Reading order: Text flows naturally as a human would read the document.

Ignore bounding box tags like <|ref|> <|det|> if present. For non-tabular text, \
do not penalise or reward markdown formatting markers (#, **, *, etc.) — plain \
text and markdown-formatted text that contain the same words are equivalent.

If both outputs capture the same text — including, where tables are present, the \
same table structure — with similar accuracy, respond with "tie". Only pick a \
winner when there is a clear quality difference.

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

CRITERIA_PROFILES: dict[str, str] = {
    "default": _DEFAULT_PROMPT,
    "table-fidelity": _TABLE_FIDELITY_PROMPT,
}


def prompt_hash(prompt_template: str) -> str:
    """Stable 12-hex-char sha256 fingerprint of a judge prompt template.

    Hashes the template (with its ``{ocr_text_a}`` placeholders intact), not a
    formatted instance, so every run using the same profile yields the same
    hash. Recorded in the results metadata for provenance.
    """
    return hashlib.sha256(prompt_template.encode()).hexdigest()[:12]


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

# Default minimum stripped-text length for a pair to be worth judging. A pair
# where BOTH outputs fall below this is skipped — neither model produced
# meaningful text, so the comparison would only add noise to the ELO.
DEFAULT_MIN_CHARS = 20

# Judge-shaped verdict for two identical outputs. Recorded as a tie without
# spending a judge call; ``agreement="auto"`` marks it as machine-decided.
_AUTO_TIE_RESULT: dict[str, str] = {
    "winner": "tie",
    "reason": "identical outputs — auto-tie",
    "agreement": "auto",
}


# --- Image helpers ---


def image_to_base64(image: Image.Image, max_dim: int = MAX_IMAGE_DIM) -> str:
    """Convert a PIL image to a base64-encoded JPEG string, resizing if needed."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    if max(image.size) > max_dim:
        ratio = max_dim / max(image.size)
        new_size = (int(image.width * ratio), int(image.height * ratio))
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=85)
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
    text_a: str = ""
    text_b: str = ""
    # Pre-decided verdict for pairs that never reach a judge (identical
    # outputs → auto-tie). ``None`` means the judge must score this pair.
    auto_result: dict[str, str] | None = None


def build_prompt(
    text_a: str,
    text_b: str,
    swapped: bool,
    prompt_template: str = CRITERIA_PROFILES[DEFAULT_CRITERIA],
) -> tuple[str, bool]:
    """Build the pairwise comparison prompt, applying position-bias swap.

    ``prompt_template`` is a criteria profile template (see ``CRITERIA_PROFILES``)
    with ``{ocr_text_a}``/``{ocr_text_b}`` placeholders; it defaults to the
    ``default`` profile so existing callers and tests are unaffected.

    Returns (prompt_text, swapped).
    """
    a = text_a[:MAX_OCR_TEXT_LENGTH]
    b = text_b[:MAX_OCR_TEXT_LENGTH]
    if swapped:
        a, b = b, a
    return prompt_template.format(ocr_text_a=a, ocr_text_b=b), swapped


def build_messages(image_b64: str, prompt: str) -> list[dict[str, Any]]:
    """Build chat messages for the judge (image + prompt)."""
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]


def _normalize_pair(a: str, b: str) -> tuple[str, str]:
    """Return a canonical (sorted) pair for symmetric lookup."""
    return (a, b) if a <= b else (b, a)


def sample_indices(
    dataset_len: int, max_samples: int | None = None, seed: int = 42
) -> list[int]:
    """Compute shuffled sample indices (cheap — no image loading).

    Args:
        dataset_len: Total number of rows in the dataset.
        max_samples: If set, randomly sample this many indices.
        seed: Random seed for reproducible sampling.

    Returns:
        List of integer indices into the dataset.
    """
    indices = list(range(dataset_len))
    if max_samples and max_samples < len(indices):
        random.seed(seed)
        indices = random.sample(indices, max_samples)
    return indices


def build_comparisons(
    dataset: Any,
    ocr_columns: dict[str, str],
    max_samples: int | None = None,
    seed: int = 42,
    skip_samples: dict[tuple[str, str], set[int]] | None = None,
    indices: list[int] | None = None,
    min_chars: int = DEFAULT_MIN_CHARS,
    prompt_template: str = CRITERIA_PROFILES[DEFAULT_CRITERIA],
) -> list[Comparison]:
    """Build pairwise comparison prompts from a dataset.

    Args:
        dataset: HF dataset with an "image" column and OCR output columns.
        ocr_columns: Mapping of column_name -> model_name.
        max_samples: If set, randomly sample this many rows. Ignored when
            ``indices`` is provided.
        seed: Random seed for sampling and position-bias randomization.
        skip_samples: Maps a (model_a, model_b) pair to the set of sample
            indices already judged for that pair; those exact (pair, sample)
            combinations are excluded. Pairs are normalized so (a, b) and
            (b, a) match. This is (pair, sample)-level, not pair-level: a pair
            judged on only some samples is still built for the rest, so a
            resumed run tops up partially-judged pairs. If None, nothing is
            skipped.
        indices: Explicit row indices to use. When provided, ``max_samples``
            and ``seed`` are not used for index selection (seed is still used
            for position-bias randomization).
        min_chars: Skip a pair when BOTH outputs are shorter than this (after
            stripping) — neither model produced meaningful text. Set to 0 to
            disable the filter.
        prompt_template: Criteria profile template (see ``CRITERIA_PROFILES``)
            used to build each comparison prompt. Defaults to the ``default``
            profile.

    Returns:
        List of Comparison objects. Pairs needing a judge carry pre-built chat
        messages; identical-output pairs carry ``auto_result`` (an auto-tie)
        and empty messages, and never reach the judge.
    """
    col_names = list(ocr_columns.keys())
    model_names = list(ocr_columns.values())
    pairs = list(combinations(range(len(col_names)), 2))

    # Normalize skip map for symmetric (order-independent) pair lookup.
    normalized_skip: dict[tuple[str, str], set[int]] = {}
    if skip_samples:
        for (a, b), sample_ids in skip_samples.items():
            normalized_skip.setdefault(_normalize_pair(a, b), set()).update(sample_ids)

    if indices is None:
        indices = sample_indices(len(dataset), max_samples, seed)

    rng = random.Random(seed)
    comparisons: list[Comparison] = []

    # Pre-fetch text columns to avoid triggering image decode per row.
    # HF Dataset supports column access (dataset["col"]), plain lists don't.
    text_cols_data: dict[str, list] | None = None
    if hasattr(dataset, "column_names"):
        text_cols_data = {col: dataset[col] for col in col_names}

    for idx in indices:
        # Which pairs still need judging for THIS sample. A pair is skipped only
        # for the specific samples already judged, so partially-judged pairs get
        # topped up on resume rather than dropped.
        needed_pairs = [
            (i, j)
            for i, j in pairs
            if idx
            not in normalized_skip.get(_normalize_pair(model_names[i], model_names[j]), ())
        ]
        if not needed_pairs:
            continue  # Skip image encoding entirely

        # Fetch text once per row (avoids re-decoding the image on list-backed
        # datasets that key the row dict lazily).
        if text_cols_data is not None:
            texts = {col: (text_cols_data[col][idx] or "") for col in col_names}
        else:
            row = dataset[idx]
            texts = {col: (row[col] or "") for col in col_names}

        # Classify each pair before touching the image. Each entry is
        # (i, j, text_a, text_b, auto_result) where auto_result is None for a
        # pair the judge must score, or an auto-tie dict for identical outputs.
        valid_pairs: list[tuple[int, int, str, str, dict[str, str] | None]] = []
        for i, j in needed_pairs:
            text_a = texts[col_names[i]]
            text_b = texts[col_names[j]]
            stripped_a, stripped_b = text_a.strip(), text_b.strip()
            # Neither model produced meaningful text — a wasted judge call.
            if len(stripped_a) < min_chars and len(stripped_b) < min_chars:
                continue
            # Identical outputs are an unambiguous tie — record without judging.
            if stripped_a == stripped_b:
                valid_pairs.append((i, j, text_a, text_b, _AUTO_TIE_RESULT.copy()))
            else:
                valid_pairs.append((i, j, text_a, text_b, None))

        if not valid_pairs:
            continue

        # Only decode/encode the image if at least one pair needs the judge;
        # rows whose pairs are all auto-ties skip the image entirely.
        needs_judge = any(auto is None for (_, _, _, _, auto) in valid_pairs)
        image_b64 = image_to_base64(dataset[idx]["image"]) if needs_judge else ""

        for i, j, text_a, text_b, auto in valid_pairs:
            if auto is not None:
                comparisons.append(
                    Comparison(
                        sample_idx=idx,
                        model_a=model_names[i],
                        model_b=model_names[j],
                        col_a=col_names[i],
                        col_b=col_names[j],
                        swapped=False,
                        messages=[],
                        text_a=text_a,
                        text_b=text_b,
                        auto_result=auto,
                    )
                )
                continue

            swapped = rng.random() < 0.5
            prompt, swapped = build_prompt(text_a, text_b, swapped, prompt_template)
            messages = build_messages(image_b64, prompt)

            comparisons.append(
                Comparison(
                    sample_idx=idx,
                    model_a=model_names[i],
                    model_b=model_names[j],
                    col_a=col_names[i],
                    col_b=col_names[j],
                    swapped=swapped,
                    messages=messages,
                    text_a=text_a,
                    text_b=text_b,
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
        # A truncated response can be a bare opening fence with no body
        parts = text.split("\n", 1)
        text = parts[1].rsplit("```", 1)[0].strip() if len(parts) == 2 else ""
    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Failed to parse judge output: %s", text[:200])
        return {}
    if not isinstance(result, dict):
        logger.warning("Judge output is not a JSON object: %s", text[:200])
        return {}
    winner = result.get("winner", "tie")
    if not isinstance(winner, str):
        logger.warning("Judge output has non-string winner: %s", text[:200])
        return {}
    winner = winner.upper().strip()
    if winner == "TIE":
        winner = "tie"
    if winner not in ("A", "B", "tie"):
        winner = "tie"
    reason = result.get("reason", "")
    if not isinstance(reason, str):
        reason = json.dumps(reason)
    return {"winner": winner, "reason": reason}
