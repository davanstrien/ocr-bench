"""Pairwise VLM judge — prompt templates, structured output schema, comparison building."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import random
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from itertools import combinations
from typing import Any

from PIL import Image

logger = logging.getLogger(__name__)

# --- Error sentinels ---
#
# OCR scripts write a placeholder string when a page fails (e.g. an image that
# won't decode, or a model crash mid-batch) rather than aborting the whole job.
# The job then "completes" with sentinel strings sitting in the output column.
# These are NOT transcriptions and must never be scored as such — a job that
# emitted only sentinels is a *failed run*, not a bad model (see issue #46).
KNOWN_SENTINELS: frozenset[str] = frozenset({"[OCR ERROR]", "[OCR FAILED]"})

# Bounds for the generic bracketed-token match. A genuine error placeholder is a
# terse token (a few ALL-CAPS words ending in ERROR/FAILED), never a heading or
# a paragraph — so anything longer or wordier is treated as real text. These
# bounds keep archival ALL-CAPS headings like "[SECTION FAILED BANKS 1866]" from
# being mistaken for sentinels.
_MAX_SENTINEL_LEN = 40
_MAX_SENTINEL_WORDS = 4

# A bracketed ALL-CAPS token whose FINAL word (before the closing bracket) is
# ERROR or FAILED, e.g. "[OCR ERROR]", "[SURYA LAYOUT ERROR]", "[GOT-OCR FAILED]".
# Requiring the keyword to be terminal (not just present) is what excludes
# archival headings that merely contain the word, e.g. "[SECTION FAILED BANKS
# 1866]". The whole stripped string must be this single token.
_SENTINEL_TOKEN_RE = re.compile(r"^\[[A-Z0-9 ._/-]{0,30}(?:ERROR|FAILED)\]$")


def is_sentinel(text: str | None) -> bool:
    """True if ``text`` is an OCR error sentinel rather than a transcription.

    Matches the known literals (case-insensitively) plus any short (≤40 char,
    ≤4 word) string that strips to a single bracketed ALL-CAPS token ending in
    ``ERROR``/``FAILED``. Empty or ``None`` values are *not* sentinels (they are
    handled as missing output on their own).
    """
    if not text:
        return False
    stripped = text.strip()
    if not stripped:
        return False
    if stripped.upper() in KNOWN_SENTINELS:
        return True
    if len(stripped) > _MAX_SENTINEL_LEN:
        return False
    if not _SENTINEL_TOKEN_RE.match(stripped):
        return False
    return len(stripped[1:-1].split()) <= _MAX_SENTINEL_WORDS


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
# Backward-compatible name for callers that imported the original single prompt.
PAIRWISE_PROMPT = CRITERIA_PROFILES[DEFAULT_CRITERIA]


def prompt_hash(prompt_template: str) -> str:
    """Stable 12-hex-char sha256 fingerprint of a judge prompt template.

    Hashes the template (with its ``{ocr_text_a}`` placeholders intact), not a
    formatted instance, so every run using the same profile yields the same
    hash. Recorded in the results metadata for provenance.
    """
    return hashlib.sha256(prompt_template.encode()).hexdigest()[:12]


def validate_prompt_template(template: str) -> None:
    """Raise ``ValueError`` if a prompt template won't work with ``build_prompt``.

    A valid template must contain both ``{ocr_text_a}`` and ``{ocr_text_b}`` and
    no other format fields — any unknown ``{field}`` or a stray unescaped brace
    would crash ``str.format`` at judge time. This mirrors how ``build_prompt``
    formats the template, so a template that passes here is safe to judge with.
    Used to vet user-supplied ``--criteria-file`` templates before a run starts.
    """
    try:
        template.format(ocr_text_a="A", ocr_text_b="B")
    except (KeyError, IndexError, ValueError) as exc:
        raise ValueError(
            f"template is not a valid format string ({exc!s}). Only "
            "{ocr_text_a} and {ocr_text_b} may appear as placeholders; escape "
            "any literal brace as {{ or }}"
        ) from exc
    missing = [f for f in ("{ocr_text_a}", "{ocr_text_b}") if f not in template]
    if missing:
        raise ValueError(
            f"template is missing required placeholder(s): {', '.join(missing)} "
            "(see the default profile for the expected shape)"
        )


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


# --- HTML normalization (format-neutral length budget) ---
#
# Verbose formats (HTML tables) spend many characters on markup, so the
# MAX_OCR_TEXT_LENGTH cap amputates them mid-content while compact markdown
# fits whole — biasing the judge before it ever reads the words. Flattening
# HTML to bare text makes the cap apply to content chars across all formats.

# Cell boundary marker inside a flattened table row. Pipe (with spaces) keeps
# cell boundaries visible and reads like a markdown table row — the judge is
# told markdown formatting is neutral, so this carries table structure without
# adding penalisable markup.
_CELL_DELIM = " | "

# Known HTML tag names. The normalizer only enters the HTML path when one of
# THESE is present, and inside that path only these are treated as markup —
# any other ``<name>`` (a GLAM transcription token like <torn>/<illegible>/
# <gap>, or a markdown autolink <https://…>) is re-emitted verbatim, never
# dropped. Names must be lowercase (HTMLParser lowercases tags before dispatch).
_KNOWN_HTML_TAGS = frozenset(
    {
        # tables
        "table", "thead", "tbody", "tfoot", "tr", "td", "th", "caption",
        "colgroup", "col",
        # blocks
        "p", "div", "span", "br", "hr", "pre", "code", "blockquote",
        "figure", "figcaption", "section", "article", "header", "footer",
        "nav", "aside", "main", "address",
        # lists
        "ul", "ol", "li", "dl", "dt", "dd",
        # headings
        "h1", "h2", "h3", "h4", "h5", "h6",
        # inline
        "b", "i", "u", "s", "em", "strong", "small", "sub", "sup", "mark",
        "del", "ins", "a", "img", "abbr", "cite", "q", "label", "font",
        "center", "wbr", "big", "tt",
        # document / non-visible
        "html", "head", "body", "script", "style", "template", "title",
        "meta", "link",
    }
)

# Tags whose boundaries start a new line in the flattened text.
_BLOCK_TAGS = frozenset(
    {
        "p", "div", "tr", "li", "br", "table", "thead", "tbody", "tfoot",
        "ul", "ol", "dl", "dt", "dd", "section", "article", "header", "footer",
        "nav", "aside", "main", "blockquote", "pre", "figure", "figcaption",
        "hr", "caption", "h1", "h2", "h3", "h4", "h5", "h6",
    }
)

# Cell tags whose boundaries insert a cell delimiter within a row.
_CELL_TAGS = frozenset({"td", "th"})

# Tags whose *content* is not visible document text — suppressed entirely.
_SKIP_CONTENT_TAGS = frozenset({"script", "style", "template"})

# Matches a KNOWN HTML tag (open/close/self-close) followed by a tag terminator
# so ``<torn>`` / ``<https://…>`` don't count as HTML. Longest-name-first
# alternation so e.g. "table" is tried before "td"/"tr" and "span" before "s".
_KNOWN_TAG_RE = re.compile(
    r"</?(?:"
    + "|".join(re.escape(t) for t in sorted(_KNOWN_HTML_TAGS, key=lambda t: len(t), reverse=True))
    + r")(?=[\s/>])",
    re.IGNORECASE,
)

# Runs of horizontal whitespace, collapsed to a single space per line.
_HWS_RE = re.compile(r"[ \t\r\f\v]+")


class _JudgeTextExtractor(HTMLParser):
    """Flatten known HTML to bare content, preserving table cell/row boundaries.

    ``convert_charrefs=True`` (the default) means HTML entities arrive already
    unescaped in ``handle_data``, so no explicit entity handling is needed.
    Known block/cell tags emit newline / cell-delimiter boundaries and are then
    dropped; ``<script>``/``<style>``/``<template>`` content is suppressed; any
    unknown-named ``<tag>`` is re-emitted verbatim so transcription tokens and
    autolinks embedded in real HTML survive. ``get_text`` collapses whitespace
    conservatively.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip_depth = 0  # >0 while inside a script/style/template subtree

    def _boundary(self, tag: str) -> None:
        if tag in _CELL_TAGS:
            self._parts.append(_CELL_DELIM)
        elif tag in _BLOCK_TAGS:
            self._parts.append("\n")

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        if tag in _SKIP_CONTENT_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return  # inside a script/style/template subtree — suppress entirely
        if tag in _KNOWN_HTML_TAGS:
            self._boundary(tag)
        else:
            # Unknown token (e.g. <torn>, <gap>, <https://…>) inside genuine
            # HTML — keep it verbatim rather than dropping it as markup.
            self._parts.append(self.get_starttag_text() or f"<{tag}>")

    def handle_startendtag(self, tag: str, attrs: Any) -> None:
        # Self-closing form, e.g. <br/> or <td/>.
        if tag in _SKIP_CONTENT_TAGS:
            return  # self-closing script/style/template carries no content
        if self._skip_depth:
            return
        if tag in _KNOWN_HTML_TAGS:
            self._boundary(tag)
        else:
            self._parts.append(self.get_starttag_text() or f"<{tag}/>")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_CONTENT_TAGS:
            if self._skip_depth > 0:
                self._skip_depth -= 1
            return
        if self._skip_depth:
            return
        if tag in _KNOWN_HTML_TAGS:
            # Closing a block puts following content on its own line; cells rely
            # on the next cell's opening delimiter, so no boundary on cell close.
            if tag in _BLOCK_TAGS:
                self._parts.append("\n")
        else:
            self._parts.append(f"</{tag}>")

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            self._parts.append(data)

    def get_text(self) -> str:
        raw = "".join(self._parts)
        lines = []
        for line in raw.split("\n"):
            line = _HWS_RE.sub(" ", line).strip()
            # Drop the spurious leading delimiter emitted before a row's first cell.
            if line.startswith("|"):
                line = line[1:].lstrip()
            if line:
                lines.append(line)
        return "\n".join(lines)


def normalize_for_judge(text: str) -> str:
    """Flatten HTML markup to bare content for the judge's length budget.

    Only text containing a KNOWN HTML tag enters the HTML path; plain text and
    markdown pass through unchanged, and unknown ``<tag>`` tokens (GLAM
    transcription conventions like ``<torn>``/``<illegible>``/``<gap>``, or
    markdown autolinks ``<https://…>``) are preserved verbatim. In the HTML
    path: table cells become ``' | '``-delimited lines so cell boundaries
    survive, ``<br>`` becomes a newline, ``<script>``/``<style>``/
    ``<template>`` content is dropped, entities are unescaped, and whitespace
    is collapsed conservatively.

    Never raises — non-str input and malformed HTML degrade to best-effort
    text (the guards live inside the protected region so nothing before the
    return can throw).
    """
    try:
        if text is None:
            return ""
        if not isinstance(text, str):
            text = (
                text.decode("utf-8", "replace")
                if isinstance(text, (bytes, bytearray))
                else str(text)
            )
        if not text or not _KNOWN_TAG_RE.search(text):
            return text
        parser = _JudgeTextExtractor()
        parser.feed(text)
        parser.close()
        return parser.get_text()
    except Exception:
        # HTMLParser is lenient, but guarantee we never raise on garbage in.
        logger.warning("normalize_for_judge failed; passing text through unchanged")
        return text if isinstance(text, str) else ""


# Harness-voice note appended AFTER an output's block (never inside it) when the
# output was truncated — so the judge doesn't read the cut as the model's own
# incompleteness, and can't mistake a marker inside the fenced text for
# model-added commentary (which criterion 1 penalises).
def _truncation_note(label: str) -> str:
    return (
        f"(Evaluator note: Output {label} above was truncated for length by the "
        f"evaluation harness — judge only what is shown; do not penalise the cut "
        f"as incompleteness.)"
    )


# Anchor for inserting the note(s): the fixed response-format instruction that
# closes PAIRWISE_PROMPT. Inserting before it places the note(s) after both
# output blocks and keeps the JSON instruction last, without editing the
# (byte-stable) prompt constant.
_JSON_INSTRUCTION_ANCHOR = "Respond with JSON only (no markdown fences"


def _apply_cap(text: str, max_len: int) -> tuple[str, bool]:
    """Truncate ``text`` to ``max_len`` chars. Returns (text, truncated).

    No marker is added to the text itself — truncation is disclosed to the
    judge via a harness-voice note composed in ``build_prompt`` and recorded as
    a ``truncated_a``/``truncated_b`` flag on the row.
    """
    if len(text) <= max_len:
        return text, False
    return text[:max_len], True


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
    truncated_a: bool = False
    truncated_b: bool = False


def build_prompt(
    text_a: str,
    text_b: str,
    swapped: bool,
    prompt_template: str = CRITERIA_PROFILES[DEFAULT_CRITERIA],
    max_len: int = MAX_OCR_TEXT_LENGTH,
    normalize: bool = True,
) -> tuple[str, bool, bool, bool]:
    """Build the selected criteria prompt, normalization, cap, and position swap.

    ``prompt_template`` supplies the judge rubric. When ``normalize`` is true,
    HTML is flattened before the character cap so markup verbosity cannot consume
    the content budget. Truncation is disclosed in harness voice outside the OCR
    output text. Returned truncation flags always refer to the original A/B order.
    """
    norm_a = normalize_for_judge(text_a) if normalize else text_a
    norm_b = normalize_for_judge(text_b) if normalize else text_b
    a, trunc_a = _apply_cap(norm_a, max_len)
    b, trunc_b = _apply_cap(norm_b, max_len)
    # Map original-order truncation to judge-visible positions for the note.
    disp_a_truncated, disp_b_truncated = (trunc_b, trunc_a) if swapped else (trunc_a, trunc_b)
    if swapped:
        a, b = b, a
    prompt = prompt_template.format(ocr_text_a=a, ocr_text_b=b)

    notes = []
    if disp_a_truncated:
        notes.append(_truncation_note("A"))
    if disp_b_truncated:
        notes.append(_truncation_note("B"))
    if notes:
        note_block = "\n".join(notes) + "\n\n"
        if _JSON_INSTRUCTION_ANCHOR in prompt:
            prompt = prompt.replace(
                _JSON_INSTRUCTION_ANCHOR, note_block + _JSON_INSTRUCTION_ANCHOR, 1
            )
        else:
            # Custom criteria templates need not use the built-in JSON wording;
            # appending still keeps the harness note outside both OCR values.
            prompt = f"{prompt.rstrip()}\n\n{note_block.rstrip()}"

    return prompt, swapped, trunc_a, trunc_b


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
    max_ocr_text_len: int = MAX_OCR_TEXT_LENGTH,
    judge_image_dim: int = MAX_IMAGE_DIM,
    normalize: bool = True,
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
            used to build each comparison prompt.
        max_ocr_text_len: Per-output character cap applied after HTML
            normalization when building each prompt.
        judge_image_dim: Longer-side pixel cap for the judge image.
        normalize: When True (default), flatten HTML to bare content before the
            cap (format-neutral). When False, cap the text as-is (raw mode).

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
            # A sentinel side is a failed output, not a transcription — exclude
            # the pair so an error string never competes (issue #46). Sentinels
            # are counted per model separately (integrity.compute_model_stats).
            if is_sentinel(text_a) or is_sentinel(text_b):
                continue
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
        image_b64 = (
            image_to_base64(dataset[idx]["image"], max_dim=judge_image_dim)
            if needs_judge
            else ""
        )

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
            prompt, swapped, trunc_a, trunc_b = build_prompt(
                text_a,
                text_b,
                swapped,
                prompt_template=prompt_template,
                max_len=max_ocr_text_len,
                normalize=normalize,
            )
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
                    truncated_a=trunc_a,
                    truncated_b=trunc_b,
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
