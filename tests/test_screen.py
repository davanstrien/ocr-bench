"""Tests for the judge-free screen checks."""

from __future__ import annotations

import json
import random
import string

import pytest

from ocr_bench.screen import (
    LOOP_MIN_CHARS,
    ScreenReport,
    loop_kind,
    screen_outputs,
    visible_chars,
)


def _prose(n_chars: int, seed: int = 0) -> str:
    """Text that does not compress much, like a real page."""
    rng = random.Random(seed)
    alphabet = string.ascii_lowercase + "      "
    return "".join(rng.choice(alphabet) for _ in range(n_chars))


def _screen(texts: dict[str, list], cap: int = 2500) -> ScreenReport:
    return screen_outputs(texts, {name: f"model-{name}" for name in texts}, cap)


def _by_name(report: ScreenReport) -> dict:
    return {m.name: m for m in report.models}


class TestVisibleChars:
    def test_ignores_whitespace(self):
        assert visible_chars("a b\n\tc  ") == 3
        assert visible_chars("") == 0


class TestLoopKind:
    def test_repeated_tail_is_a_loop(self):
        assert loop_kind(_prose(300) + "the same line again\n" * 120) == "repeat"

    def test_normal_text_is_not_a_loop(self):
        assert loop_kind(_prose(5000)) is None

    def test_only_the_tail_is_checked(self):
        # A repeated block followed by 2,000+ characters of normal text is not a
        # loop: generation recovered.
        assert loop_kind("the same line again\n" * 120 + _prose(2500)) is None

    def test_short_repeated_text_is_not_measured(self):
        text = "ab " * 100
        assert len(text) < LOOP_MIN_CHARS
        assert loop_kind(text) is None

    def test_character_run_of_200_is_a_loop(self):
        assert loop_kind("Title\n" + "." * 200) == "char run"

    def test_character_run_of_199_is_not(self):
        assert loop_kind("Title\n" + "." * 199) is None

    def test_whitespace_run_is_not_a_character_run(self):
        assert loop_kind("Title" + " " * 300 + "end") is None


class TestScreenOutputs:
    def test_needs_three_models(self):
        with pytest.raises(ValueError, match="at least 3"):
            _screen({"a": ["x"], "b": ["y"]})

    def test_needs_equal_row_counts(self):
        with pytest.raises(ValueError, match="same number of rows"):
            _screen({"a": ["x"], "b": ["y"], "c": ["z", "w"]})

    def test_text_on_blank_page(self):
        report = _screen({"a": [""], "b": [" "], "c": ["p. 3"], "d": [_prose(150)]})
        models = _by_name(report)
        assert models["d"].blank_pages == 1
        assert models["d"].text_on_blank_rows == [0]
        assert models["a"].blank_pages == 1  # b and c are near-empty: 2 of 3 others
        assert models["a"].text_on_blank_rows == []

    def test_blank_page_needs_most_of_the_others(self):
        # For d, only 2 of 4 others are near-empty: not a blank page.
        texts = {"a": [""], "b": [""], "c": [_prose(300, 1)], "e": [_prose(300, 2)]}
        texts["d"] = [_prose(150, 3)]
        assert _by_name(_screen(texts))["d"].blank_pages == 0

    def test_empty_on_text_page(self):
        texts = {name: [_prose(400, seed)] for seed, name in enumerate("abc")}
        texts["d"] = ["page 4"]
        models = _by_name(_screen(texts))
        assert models["d"].text_pages == 1
        assert models["d"].empty_on_text_rows == [0]
        assert models["a"].empty_on_text_rows == []

    def test_long_output(self):
        texts = {name: [_prose(100, seed)] for seed, name in enumerate("abc")}
        texts["d"] = [_prose(700, 9)]
        texts["e"] = [_prose(400, 8)]  # 4x the others' median: not flagged
        models = _by_name(_screen(texts))
        assert models["d"].long_rows == [0]
        assert models["e"].long_rows == []

    def test_long_output_uses_a_floor_for_short_pages(self):
        # The others' median is 10 characters; 5x the floor of 50 is 250.
        texts = {name: ["x" * 10] for name in "abc"}
        texts["d"] = [_prose(240)]
        assert _by_name(_screen(texts))["d"].long_rows == []

    def test_sentinels_are_left_out(self):
        texts = {
            "a": ["[OCR ERROR]", _prose(300, 1)],
            "b": [_prose(300, 2), _prose(300, 3)],
            "c": [_prose(300, 4), _prose(300, 5)],
        }
        models = _by_name(_screen(texts))
        assert models["a"].pages == 1
        # Row 0 has only two outputs, so no model gets a peer check there.
        assert models["b"].peer_pages == 1
        assert models["b"].pages == 2

    def test_peer_agreement_orders_models(self):
        page = _prose(500, 1)
        texts = {"a": [page], "b": [page], "c": [page + " extra"], "d": [_prose(500, 2)]}
        report = _screen(texts)
        order = [m.name for m in report.by_peer_agreement()]
        assert order[-1] == "d"
        models = _by_name(report)
        # a's similarities: 1.0 (b), about 0.99 (c), low (d); the median is c's.
        assert models["a"].peer_agreement > 0.95
        assert models["d"].peer_agreement < 0.6

    def test_html_is_measured_as_the_judge_sees_it(self):
        empty_table = "<table><tr><td></td></tr></table>"
        texts = {name: [_prose(400, seed)] for seed, name in enumerate("abc")}
        texts["d"] = [empty_table]
        assert _by_name(_screen(texts))["d"].empty_on_text_rows == [0]

    def test_cap_suggestion_leaves_out_flagged_outputs(self):
        texts = {name: [_prose(3100, seed)] for seed, name in enumerate("abc")}
        texts["d"] = [_prose(300) + "loop line\n" * 900]  # 9,300 characters, a loop
        report = _screen(texts, cap=2500)
        assert _by_name(report)["d"].loop_rows == [0]
        assert report.longest_unflagged == 3100
        assert report.unflagged_outputs == 3
        assert report.unflagged_over_cap == 3
        assert report.suggested_max_ocr_text_len == 3500

    def test_no_cap_suggestion_when_the_cap_fits(self):
        texts = {name: [_prose(900, seed)] for seed, name in enumerate("abc")}
        report = _screen(texts, cap=2500)
        assert report.unflagged_over_cap == 0
        assert report.suggested_max_ocr_text_len is None

    def test_char_run_rows_are_listed(self):
        texts = {name: [_prose(400, seed)] for seed, name in enumerate("abc")}
        texts["d"] = ["Contents " + "." * 250]
        models = _by_name(_screen(texts))
        assert models["d"].loop_rows == [0]
        assert models["d"].char_run_rows == [0]

    def test_to_dict_is_json(self):
        texts = {name: [_prose(400, seed), ""] for seed, name in enumerate("abc")}
        data = json.loads(json.dumps(_screen(texts).to_dict()))
        assert data["rows"] == 2
        assert {m["name"] for m in data["models"]} == {"a", "b", "c"}
        assert set(data["models"][0]) >= {"loop", "long", "text_on_blank", "peer_agreement"}
