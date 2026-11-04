"""Free checks on OCR outputs before a paid judge run (``ocr-bench audit --screen``).

Each check compares one model's output on a page with the other models' outputs
on the same page. No judge and no ground truth are used. The checks find outputs
that a judge would score low anyway, so a weak model can be found, and dropped,
before the judge run:

- **loop**: the end of the output repeats itself (generation ran on to the token
  limit), or one non-space character repeats 200 or more times;
- **long**: the output is more than five times as long as the median of the
  other models' outputs on the page;
- **text on a blank page**: most outputs on the page have almost no text, and
  this model outputs 100 or more characters;
- **empty on a text page**: most outputs on the page have 200 or more
  characters, and this model outputs almost nothing;
- **peer agreement**: how similar the output is to the other models' outputs
  (median normalised Levenshtein similarity). Loops are left out, so this score
  does not repeat the loop check.

These are signals, not verdicts. A model can transcribe faint text that the other
models leave out, and a page can have one correct model and several wrong ones.
The loop check measures how well the tail compresses, so a loop whose text
changes on each repeat (for example a counting list ``- 1``, ``- 2``, …) can pass.
"""

from __future__ import annotations

import math
import re
import statistics
import zlib
from dataclasses import asdict, dataclass, field

from rapidfuzz.distance import Levenshtein
from rapidfuzz.process import cdist

from ocr_bench.judge import is_sentinel, normalize_for_judge

# A loop sits at the end of an output, so only the tail is compressed. Text
# shorter than LOOP_MIN_CHARS is too short to measure this way.
LOOP_TAIL_CHARS = 2000
LOOP_MIN_CHARS = 500
LOOP_TAIL_COMPRESSION = 0.15  # compressed size / original size of the tail
CHAR_RUN_LENGTH = 200
# One non-space character repeated CHAR_RUN_LENGTH times or more.
CHAR_RUN = re.compile(r"(\S)\1{" + str(CHAR_RUN_LENGTH - 1) + ",}")

LONG_RATIO = 5.0
# Floor for the others' median in the length ratio, so that a page where the
# others output a few characters does not flag every normal output as long.
LONG_MIN_PEER_CHARS = 50

NEAR_EMPTY_CHARS = 20  # the audit's "<20ch" threshold
TEXT_ON_BLANK_CHARS = 100
TEXT_PAGE_CHARS = 200

# Peer similarity cost grows with text length; loops are flagged on their own.
PEER_MAX_CHARS = 20_000

# The page checks need outputs from at least this many models on the page.
MIN_MODELS = 3

# The suggested --max-ocr-text-len is rounded up to a multiple of this.
CAP_STEP = 500

THRESHOLDS = {
    "loop_tail_chars": LOOP_TAIL_CHARS,
    "loop_min_chars": LOOP_MIN_CHARS,
    "loop_tail_compression": LOOP_TAIL_COMPRESSION,
    "char_run_length": CHAR_RUN_LENGTH,
    "long_ratio": LONG_RATIO,
    "long_min_peer_chars": LONG_MIN_PEER_CHARS,
    "near_empty_chars": NEAR_EMPTY_CHARS,
    "text_on_blank_chars": TEXT_ON_BLANK_CHARS,
    "text_page_chars": TEXT_PAGE_CHARS,
    "peer_max_chars": PEER_MAX_CHARS,
    "min_models": MIN_MODELS,
    "cap_step": CAP_STEP,
}


def visible_chars(text: str) -> int:
    """Count the characters that are not whitespace."""
    return sum(1 for ch in text if not ch.isspace())


def _compression_ratio(text: str) -> float:
    data = text.encode()
    if not data:
        return 1.0
    return len(zlib.compress(data, 9)) / len(data)


def loop_kind(text: str) -> str | None:
    """Return ``"repeat"`` or ``"char run"`` if the text looks like a loop, else None.

    ``repeat``: the output has at least LOOP_MIN_CHARS characters and its last
    LOOP_TAIL_CHARS characters compress to less than LOOP_TAIL_COMPRESSION of
    their size. ``char run``: one non-space character repeats CHAR_RUN_LENGTH
    times or more.
    """
    if len(text) >= LOOP_MIN_CHARS:
        tail = text[-LOOP_TAIL_CHARS:]
        if _compression_ratio(tail) < LOOP_TAIL_COMPRESSION:
            return "repeat"
    if CHAR_RUN.search(text):
        return "char run"
    return None


def _peer_text(text: str) -> str:
    """Text for the peer similarity: case-folded, whitespace collapsed, capped."""
    return " ".join(text.casefold().split())[:PEER_MAX_CHARS]


@dataclass
class FlaggedOutput:
    """One output that at least one check flagged, with the evidence."""

    row: int  # 0-based dataset row
    checks: list[str]  # "loop", "long", "text_on_blank", "empty_on_text"
    length: int  # characters of judge input (what --max-ocr-text-len caps)
    visible: int  # characters that are not whitespace
    others_median_visible: float | None  # None when the page checks did not run
    loop_kind: str | None = None


@dataclass
class ModelScreen:
    """Screen results for one model. Row numbers are 0-based dataset rows."""

    name: str  # config or column name
    model: str
    pages: int = 0  # rows with an output that is not an error sentinel
    checked_pages: int = 0  # of those, rows with outputs from MIN_MODELS or more models
    loop_rows: list[int] = field(default_factory=list)
    char_run_rows: list[int] = field(default_factory=list)  # loops found by CHAR_RUN only
    long_rows: list[int] = field(default_factory=list)
    blank_pages: int = 0  # checked rows where most outputs have almost no text
    text_on_blank_rows: list[int] = field(default_factory=list)
    text_pages: int = 0  # checked rows where most outputs have TEXT_PAGE_CHARS or more
    empty_on_text_rows: list[int] = field(default_factory=list)
    peer_pages: int = 0  # rows with a peer similarity (the output is not a loop)
    peer_agreement: float | None = None  # mean over rows of the median similarity
    flagged: list[FlaggedOutput] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "model": self.model,
            "pages": self.pages,
            "checked_pages": self.checked_pages,
            "loop": {
                "count": len(self.loop_rows),
                "pages": self.pages,
                "rows": self.loop_rows,
                "char_run_rows": self.char_run_rows,
            },
            "long": {
                "count": len(self.long_rows),
                "pages": self.checked_pages,
                "rows": self.long_rows,
            },
            "text_on_blank": {
                "count": len(self.text_on_blank_rows),
                "pages": self.blank_pages,
                "rows": self.text_on_blank_rows,
            },
            "empty_on_text": {
                "count": len(self.empty_on_text_rows),
                "pages": self.text_pages,
                "rows": self.empty_on_text_rows,
            },
            "peer_agreement": self.peer_agreement,
            "peer_pages": self.peer_pages,
            "flagged": [asdict(f) for f in self.flagged],
        }


@dataclass
class ScreenReport:
    """Screen results for all models, plus a text-cap suggestion."""

    models: list[ModelScreen]
    rows: int
    max_ocr_text_len: int
    # Judge-input lengths of the outputs that no check flagged.
    unflagged_outputs: int = 0
    unflagged_median: int = 0
    unflagged_p95: int = 0
    longest_unflagged: int = 0
    unflagged_over_cap: int = 0  # unflagged outputs that the current cap truncates

    @property
    def suggested_max_ocr_text_len(self) -> int | None:
        """A cap that fits every unflagged output, or None if the current cap does."""
        if self.longest_unflagged <= self.max_ocr_text_len:
            return None
        return math.ceil(self.longest_unflagged / CAP_STEP) * CAP_STEP

    def to_dict(self) -> dict:
        return {
            "rows": self.rows,
            "thresholds": dict(THRESHOLDS),
            "max_ocr_text_len": self.max_ocr_text_len,
            "unflagged_outputs": self.unflagged_outputs,
            "unflagged_median": self.unflagged_median,
            "unflagged_p95": self.unflagged_p95,
            "longest_unflagged": self.longest_unflagged,
            "unflagged_over_cap": self.unflagged_over_cap,
            "suggested_max_ocr_text_len": self.suggested_max_ocr_text_len,
            "models": [m.to_dict() for m in self.models],
        }


def _p95(values: list[int]) -> int:
    if len(values) < 2:
        return values[0] if values else 0
    return round(statistics.quantiles(values, n=20, method="inclusive")[-1])


def screen_outputs(
    texts: dict[str, list],
    models: dict[str, str],
    max_ocr_text_len: int,
    normalize: bool = True,
) -> ScreenReport:
    """Run the screen checks over row-aligned outputs.

    ``texts`` maps a config or column name to its outputs, one per row, in the
    same row order for every name. ``models`` maps the same names to model
    labels. Text is measured as the judge sees it: after HTML normalisation
    when ``normalize`` is true. Error sentinels are left out of every check,
    for the model itself and as a peer.
    """
    names = list(texts)
    if len(names) < MIN_MODELS:
        raise ValueError(f"the screen needs at least {MIN_MODELS} models, got {len(names)}")
    row_counts = {len(values) for values in texts.values()}
    if len(row_counts) != 1:
        raise ValueError("the screen needs the same number of rows for every model")
    n_rows = row_counts.pop()

    prepared: dict[str, list[str | None]] = {}
    for name in names:
        column: list[str | None] = []
        for value in texts[name]:
            raw = value or ""
            if is_sentinel(raw):
                column.append(None)
            else:
                column.append(normalize_for_judge(raw) if normalize else raw)
        prepared[name] = column

    results = {name: ModelScreen(name=name, model=models.get(name, name)) for name in names}
    peer_scores: dict[str, list[float]] = {name: [] for name in names}
    lengths: list[int] = []  # judge-input lengths of unflagged outputs

    for row in range(n_rows):
        page: dict[str, str] = {}
        for name in names:
            text = prepared[name][row]
            if text is not None:
                page[name] = text
        page_names = list(page)
        visible = {name: visible_chars(page[name]) for name in page_names}
        loops = {name: loop_kind(page[name]) for name in page_names}

        # Page checks: is this a blank page or a text page?
        checked = len(page_names) >= MIN_MODELS
        blank_page = False
        text_page = False
        if checked:
            near_empty = sum(1 for name in page_names if visible[name] < NEAR_EMPTY_CHARS)
            with_text = sum(1 for name in page_names if visible[name] >= TEXT_PAGE_CHARS)
            blank_page = near_empty * 2 > len(page_names)
            text_page = with_text * 2 > len(page_names)

        # Peer similarity among the outputs that are not loops.
        similarity: dict[str, float] = {}
        peer_names = [name for name in page_names if loops[name] is None]
        if len(peer_names) >= MIN_MODELS:
            docs = [_peer_text(page[name]) for name in peer_names]
            matrix = cdist(docs, docs, scorer=Levenshtein.normalized_similarity, workers=-1)
            for i, name in enumerate(peer_names):
                scores = [float(matrix[i][j]) for j in range(len(peer_names)) if j != i]
                similarity[name] = statistics.median(scores)

        for name in page_names:
            text = page[name]
            result = results[name]
            result.pages += 1
            checks: list[str] = []
            kind = loops[name]
            if kind is not None:
                checks.append("loop")
                result.loop_rows.append(row)
                if kind == "char run":
                    result.char_run_rows.append(row)

            others_median: float | None = None
            if checked:
                result.checked_pages += 1
                others_median = statistics.median(
                    visible[other] for other in page_names if other != name
                )
                if visible[name] > LONG_RATIO * max(others_median, LONG_MIN_PEER_CHARS):
                    checks.append("long")
                    result.long_rows.append(row)
                if blank_page:
                    result.blank_pages += 1
                    if visible[name] >= TEXT_ON_BLANK_CHARS:
                        checks.append("text_on_blank")
                        result.text_on_blank_rows.append(row)
                if text_page:
                    result.text_pages += 1
                    if visible[name] < NEAR_EMPTY_CHARS:
                        checks.append("empty_on_text")
                        result.empty_on_text_rows.append(row)

            if name in similarity:
                peer_scores[name].append(similarity[name])

            if checks:
                result.flagged.append(
                    FlaggedOutput(
                        row=row,
                        checks=checks,
                        length=len(text),
                        visible=visible[name],
                        others_median_visible=others_median,
                        loop_kind=kind,
                    )
                )
            else:
                lengths.append(len(text))

    for name, scores in peer_scores.items():
        if scores:
            results[name].peer_pages = len(scores)
            results[name].peer_agreement = statistics.fmean(scores)

    return ScreenReport(
        models=[results[name] for name in names],
        rows=n_rows,
        max_ocr_text_len=max_ocr_text_len,
        unflagged_outputs=len(lengths),
        unflagged_median=round(statistics.median(lengths)) if lengths else 0,
        unflagged_p95=_p95(lengths),
        longest_unflagged=max(lengths, default=0),
        unflagged_over_cap=sum(1 for length in lengths if length > max_ocr_text_len),
    )
