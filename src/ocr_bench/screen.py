"""Free checks on OCR outputs before a paid judge run (``ocr-bench audit --screen``).

Each check compares one model's output on a page with the other models' outputs
on the same page. No judge and no ground truth are used. The checks find outputs
that a judge would score low anyway, so a weak model can be found, and dropped,
before the judge run:

- **loop**: the end of the output repeats itself (generation ran on to the token
  limit), or one character repeats 200 or more times;
- **long**: the output is more than five times as long as the median of the
  other models' outputs on the page;
- **text on a blank page**: most other models output almost nothing, and this
  model outputs 100 or more characters;
- **empty on a text page**: the other models output 200 or more characters, and
  this model outputs almost nothing;
- **peer agreement**: how similar the output is to the other models' outputs
  (median normalised Levenshtein similarity).

These are signals, not verdicts. A model can transcribe faint text that the other
models leave out, and a page can have one correct model and several wrong ones.
"""

from __future__ import annotations

import math
import re
import statistics
import zlib
from dataclasses import dataclass, field

from rapidfuzz.distance import Levenshtein
from rapidfuzz.process import cdist

from ocr_bench.judge import is_sentinel, normalize_for_judge

# A loop sits at the end of an output, so only the tail is compressed. Text
# shorter than LOOP_MIN_CHARS is too short to measure this way.
LOOP_TAIL_CHARS = 2000
LOOP_MIN_CHARS = 500
LOOP_TAIL_COMPRESSION = 0.15  # compressed size / original size of the tail
# One non-space character repeated this many times or more (200 in a row).
CHAR_RUN = re.compile(r"(\S)\1{199,}")

LONG_RATIO = 5.0
# Floor for the peer median in the length ratio, so that a page where the others
# output a few characters does not flag every normal output as long.
LONG_MIN_PEER_CHARS = 50

NEAR_EMPTY_CHARS = 20  # the audit's "<20ch" threshold
TEXT_ON_BLANK_CHARS = 100
TEXT_PAGE_CHARS = 200

# Peer similarity cost grows with text length; loops are flagged on their own.
PEER_MAX_CHARS = 20_000

# Each model needs at least two other outputs on a page for a peer check.
MIN_MODELS = 3

# The suggested --max-ocr-text-len is rounded up to a multiple of this.
CAP_STEP = 500


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
    their size. ``char run``: one non-space character repeats 200 or more times.
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
class ModelScreen:
    """Screen results for one model. Row numbers are 0-based dataset rows."""

    name: str  # config or column name
    model: str
    pages: int = 0  # rows with an output that is not an error sentinel
    loop_rows: list[int] = field(default_factory=list)
    char_run_rows: list[int] = field(default_factory=list)  # loops found by CHAR_RUN only
    long_rows: list[int] = field(default_factory=list)
    blank_pages: int = 0  # rows where most other models output almost nothing
    text_on_blank_rows: list[int] = field(default_factory=list)
    text_pages: int = 0  # rows where the other models output TEXT_PAGE_CHARS or more
    empty_on_text_rows: list[int] = field(default_factory=list)
    peer_pages: int = 0  # rows with a peer similarity
    peer_agreement: float | None = None  # mean over rows of the median similarity

    def _rate(self, n: int, total: int) -> float:
        return n / total if total else 0.0

    @property
    def loop_rate(self) -> float:
        return self._rate(len(self.loop_rows), self.pages)

    @property
    def long_rate(self) -> float:
        return self._rate(len(self.long_rows), self.pages)

    @property
    def text_on_blank_rate(self) -> float:
        return self._rate(len(self.text_on_blank_rows), self.blank_pages)

    @property
    def empty_on_text_rate(self) -> float:
        return self._rate(len(self.empty_on_text_rows), self.text_pages)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "model": self.model,
            "pages": self.pages,
            "loop": {"count": len(self.loop_rows), "rate": self.loop_rate, "rows": self.loop_rows},
            "char_run_rows": self.char_run_rows,
            "long": {"count": len(self.long_rows), "rate": self.long_rate, "rows": self.long_rows},
            "text_on_blank": {
                "count": len(self.text_on_blank_rows),
                "blank_pages": self.blank_pages,
                "rate": self.text_on_blank_rate,
                "rows": self.text_on_blank_rows,
            },
            "empty_on_text": {
                "count": len(self.empty_on_text_rows),
                "text_pages": self.text_pages,
                "rate": self.empty_on_text_rate,
                "rows": self.empty_on_text_rows,
            },
            "peer_agreement": self.peer_agreement,
            "peer_pages": self.peer_pages,
        }


@dataclass
class ScreenReport:
    """Screen results for all models, plus a text-cap suggestion."""

    models: list[ModelScreen]
    rows: int
    max_ocr_text_len: int
    # Longest judge-input text among outputs that no check flagged.
    longest_unflagged: int = 0
    unflagged_outputs: int = 0
    unflagged_over_cap: int = 0  # unflagged outputs that the current cap truncates

    @property
    def suggested_max_ocr_text_len(self) -> int | None:
        """A cap that fits every unflagged output, or None if the current cap does."""
        if self.longest_unflagged <= self.max_ocr_text_len:
            return None
        return math.ceil(self.longest_unflagged / CAP_STEP) * CAP_STEP

    def by_peer_agreement(self) -> list[ModelScreen]:
        """Models with the highest peer agreement first; models without one last."""
        return sorted(
            self.models,
            key=lambda m: (m.peer_agreement is None, -(m.peer_agreement or 0.0)),
        )

    def to_dict(self) -> dict:
        return {
            "rows": self.rows,
            "max_ocr_text_len": self.max_ocr_text_len,
            "longest_unflagged": self.longest_unflagged,
            "unflagged_outputs": self.unflagged_outputs,
            "unflagged_over_cap": self.unflagged_over_cap,
            "suggested_max_ocr_text_len": self.suggested_max_ocr_text_len,
            "models": [m.to_dict() for m in self.by_peer_agreement()],
        }


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
        page = {name: prepared[name][row] for name in names if prepared[name][row] is not None}
        page_names = list(page)
        visible = {name: visible_chars(page[name] or "") for name in page_names}
        similarity = None
        if len(page_names) >= MIN_MODELS:
            docs = [_peer_text(page[name] or "") for name in page_names]
            similarity = cdist(docs, docs, scorer=Levenshtein.normalized_similarity, workers=-1)

        for i, name in enumerate(page_names):
            text = page[name] or ""
            result = results[name]
            result.pages += 1
            flagged = False
            kind = loop_kind(text)
            if kind is not None:
                flagged = True
                result.loop_rows.append(row)
                if kind == "char run":
                    result.char_run_rows.append(row)

            others = [other for other in page_names if other != name]
            if similarity is not None:
                other_visible = [visible[other] for other in others]
                peer_median = statistics.median(other_visible)
                if visible[name] > LONG_RATIO * max(peer_median, LONG_MIN_PEER_CHARS):
                    flagged = True
                    result.long_rows.append(row)
                near_empty = sum(1 for v in other_visible if v < NEAR_EMPTY_CHARS)
                if near_empty * 2 > len(others):
                    result.blank_pages += 1
                    if visible[name] >= TEXT_ON_BLANK_CHARS:
                        flagged = True
                        result.text_on_blank_rows.append(row)
                elif peer_median >= TEXT_PAGE_CHARS:
                    result.text_pages += 1
                    if visible[name] < NEAR_EMPTY_CHARS:
                        result.empty_on_text_rows.append(row)
                scores = [float(similarity[i][j]) for j in range(len(page_names)) if j != i]
                peer_scores[name].append(statistics.median(scores))

            if not flagged:
                lengths.append(len(text))

    for name, scores in peer_scores.items():
        if scores:
            results[name].peer_pages = len(scores)
            results[name].peer_agreement = statistics.fmean(scores)

    return ScreenReport(
        models=[results[name] for name in names],
        rows=n_rows,
        max_ocr_text_len=max_ocr_text_len,
        longest_unflagged=max(lengths, default=0),
        unflagged_outputs=len(lengths),
        unflagged_over_cap=sum(1 for length in lengths if length > max_ocr_text_len),
    )
