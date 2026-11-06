"""Tests for the judge-free screen checks."""

from __future__ import annotations

import json
import random
import string

import pytest

from ocr_bench.screen import (
    COUNTING_RUN,
    LOOP_MIN_CHARS,
    ScreenReport,
    longest_counting_run,
    loop_kind,
    screen_outputs,
    visible_chars,
)


def _prose(n_chars: int, seed: int = 0) -> str:
    """Text that does not compress much, like a real page."""
    rng = random.Random(seed)
    alphabet = string.ascii_lowercase + "      "
    return "".join(rng.choice(alphabet) for _ in range(n_chars))


_SURNAMES = [
    "Abbott",
    "Baker",
    "Carter",
    "Dawson",
    "Ellis",
    "Foster",
    "Graham",
    "Hughes",
    "Irving",
    "Jarvis",
    "Knight",
    "Lawson",
    "Mason",
    "Norris",
    "Owen",
    "Parker",
    "Quinn",
    "Reeves",
    "Stone",
    "Turner",
    "Upton",
    "Vance",
    "Walker",
    "Young",
    "Zeal",
]
_COINS = ["Æ. Head r.", "AR. Bust l.", "Æ. Wreath.", "AR. Eagle.", "Æ. Galley.", "AV. Victory."]


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

    def test_counting_list_is_a_loop(self):
        # DeepSeek-OCR's output for a near-blank Britannica page: "- 1" to "- 1656".
        # Each line has a new number, so the tail does not compress.
        text = "".join(f"- {n}\n" for n in range(1, 1657))
        assert loop_kind(text) == "counting"

    def test_inline_counting_is_a_loop(self):
        text = _prose(400) + " ".join(f"{n}." for n in range(800, 1200))
        assert loop_kind(text) == "counting"

    def test_counting_run_threshold(self):
        # The same text in every entry, so only the count decides.
        below = _prose(300) + "".join(f"{n} Ditto\n" for n in range(1, COUNTING_RUN))
        at = below + f"{COUNTING_RUN} Ditto\n"
        assert len(below) >= LOOP_MIN_CHARS
        assert loop_kind(below) is None
        assert loop_kind(at) == "counting"

    def test_padded_table_counting_is_a_loop(self):
        # Cell padding changes with the digit count; whitespace in the gap is collapsed.
        text = _prose(300) + "| " + " | ".join(f"{n:<5}" for n in range(950, 1100)) + " |"
        assert loop_kind(text) == "counting"

    # Real numbered lists, from a review probe. Each entry has its own text, so
    # the gaps between the numbers differ.
    @pytest.mark.parametrize(
        "text",
        [
            "LIST OF MEMBERS.\n"
            + "".join(f"{i}. {_SURNAMES[i % 25]}, {'ABCDEFGHJ'[i % 9]}.\n" for i in range(1, 71)),
            "CATALOGUE OF COINS.\n"
            + "".join(f"{i}. {_COINS[(i * 7) % 6]}\n" for i in range(213, 281)),
            "".join(f"{i} {_prose(25, i)}\n" for i in range(1, 80)),
            "".join(f"{i} {_prose(50, i)}\n" for i in range(1, 80)),
            "LIST OF PLATES.\n"
            + "".join(f"Plate {i}. {_SURNAMES[i % 25]} Hall.\n" for i in range(1, 60)),
            _prose(600)
            + "\n"
            + "".join(f"{i}. {_SURNAMES[(i * 3) % 25]}.\n" for i in range(1, 56)),
        ],
        ids=[
            "membership roll",
            "coin catalogue",
            "numbered verse",
            "numbered long verse",
            "list of plates",
            "prose then 55 names",
        ],
    )
    def test_numbered_list_with_distinct_entries_is_not_a_loop(self, text):
        assert len(text) >= LOOP_MIN_CHARS
        assert loop_kind(text) is None

    # The known false positive: 50 or more numbered entries with the same text.
    # (Longer repeated text, such as "S. 1. Repealed." on every line, already
    # compresses enough to be flagged as a repeat.)
    @pytest.mark.parametrize(
        "text",
        [
            "".join(f"{i} Ditto\n" for i in range(200, 261)),
            "".join(f"{i}\nfol.\n" for i in range(100, 170)),
        ],
        ids=["ditto lots", "folio numbers"],
    )
    def test_numbered_list_with_the_same_entry_is_flagged(self, text):
        assert loop_kind(text) == "counting"

    def test_numeric_table_is_not_a_loop(self):
        # Digits differ on every row, but the numbers do not count up. Replacing
        # every digit with 0 first would make this table compress like a loop.
        rng = random.Random(3)
        rows = [" | ".join(str(rng.randint(1, 9999)) for _ in range(8)) for _ in range(80)]
        text = "\n".join(f"| {row} |" for row in rows)
        assert len(text) >= LOOP_MIN_CHARS
        assert loop_kind(text) is None


class TestLongestCountingRun:
    def test_counts_numbers_that_go_up_by_one(self):
        assert longest_counting_run("- 7\n- 8\n- 9") == 3
        assert longest_counting_run("7. 8. 9. 10.") == 4

    def test_a_gap_starts_a_new_run(self):
        assert longest_counting_run("1 2 3 5 6") == 3

    def test_different_text_between_numbers_ends_the_run(self):
        assert longest_counting_run("7. Abbott\n8. Baker\n9. Carter\n") == 2
        # A new gap starts a run at the number before it.
        assert longest_counting_run("1, 2, 3; 4; 5; 6") == 4

    def test_whitespace_in_the_gap_is_collapsed(self):
        assert longest_counting_run("| 998  | 999  | 1000 | 1001 |") == 4

    def test_decimal_is_two_numbers(self):
        assert longest_counting_run("1.10") == 1

    def test_no_numbers(self):
        assert longest_counting_run("no digits here") == 0

    def test_very_long_digit_string_ends_the_run(self):
        # int() refuses digit strings over 4,300 characters; such a string is
        # not read as a number.
        assert longest_counting_run("1 2 " + "7" * 5000 + " 3 4") == 2


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

    def test_blank_page_needs_most_of_the_outputs(self):
        # Only 2 of 5 outputs are near-empty: not a blank page.
        texts = {"a": [""], "b": [""], "c": [_prose(300, 1)], "e": [_prose(300, 2)]}
        texts["d"] = [_prose(150, 3)]
        assert _by_name(_screen(texts))["d"].blank_pages == 0

    def test_even_split_is_neither_blank_nor_text(self):
        # 3 near-empty and 3 long outputs: no page-type flag either way. (The
        # length check still compares each output with the others' median.)
        texts = {name: [""] for name in "abc"}
        texts.update({name: [_prose(400, seed)] for seed, name in enumerate("def")})
        models = _by_name(_screen(texts))
        assert all(m.blank_pages == 0 and m.text_pages == 0 for m in models.values())
        assert all(not m.text_on_blank_rows for m in models.values())
        assert all(not m.empty_on_text_rows for m in models.values())

    def test_blank_and_text_pages_are_page_properties(self):
        # The same page counts as blank for every model on it.
        report = _screen({"a": [""], "b": [""], "c": ["x"], "d": [_prose(150)]})
        assert {m.blank_pages for m in report.models} == {1}

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

    def test_number_cells_count_as_empty_output(self):
        # A number in an OCR column used to crash is_sentinel.
        texts = {name: [_prose(400, seed)] for seed, name in enumerate("abc")}
        texts["d"] = [12345]
        models = _by_name(_screen(texts))
        assert models["d"].pages == 1
        assert models["d"].empty_on_text_rows == [0]

    def test_peer_agreement_orders_models(self):
        page = _prose(500, 1)
        texts = {"a": [page], "b": [page], "c": [page + " extra"], "d": [_prose(500, 2)]}
        report = _screen(texts)
        assert [m.name for m in report.models] == ["a", "b", "c", "d"]  # input order
        models = _by_name(report)
        assert min(models.values(), key=lambda m: m.peer_agreement).name == "d"
        # a's similarities: 1.0 (b), about 0.99 (c), low (d); the median is c's.
        assert models["a"].peer_agreement > 0.95
        assert models["d"].peer_agreement < 0.6

    def test_peer_agreement_leaves_out_loops(self):
        page = _prose(600, 1)
        loop = _prose(300, 2) + "the same line again\n" * 120
        texts = {"a": [page, page], "b": [page, page], "c": [page, page], "d": [page, loop]}
        models = _by_name(_screen(texts))
        assert models["d"].loop_rows == [1]
        assert models["d"].peer_pages == 1  # only row 0
        assert models["d"].peer_agreement == pytest.approx(1.0)
        assert models["a"].peer_pages == 2  # row 1 is scored against b and c only
        assert models["a"].peer_agreement == pytest.approx(1.0)

    def test_flagged_outputs_carry_evidence(self):
        texts = {name: [_prose(400, seed)] for seed, name in enumerate("abc")}
        texts["d"] = [""]
        flagged = _by_name(_screen(texts))["d"].flagged
        assert len(flagged) == 1
        assert flagged[0].row == 0
        assert flagged[0].checks == ["empty_on_text"]
        assert flagged[0].visible == 0
        assert flagged[0].others_median_visible > 200

    def test_empty_on_text_is_left_out_of_the_cap(self):
        texts = {name: ["ab" * 100] for name in "bc"}
        texts["a"] = [" " * 300 + "x"]  # 301 characters, 1 visible
        texts["d"] = ["cd" * 100]
        report = _screen(texts, cap=250)
        assert _by_name(report)["a"].empty_on_text_rows == [0]
        assert report.unflagged_outputs == 3
        assert report.longest_unflagged == 200
        assert report.suggested_max_ocr_text_len is None

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

    def test_counting_rows_are_listed(self):
        texts = {name: [_prose(400, seed)] for seed, name in enumerate("abc")}
        texts["d"] = ["".join(f"- {n}\n" for n in range(1, 400))]
        models = _by_name(_screen(texts))
        assert models["d"].loop_rows == [0]
        assert models["d"].counting_rows == [0]
        assert models["d"].char_run_rows == []

    def test_to_dict_is_json(self):
        texts = {name: [_prose(400, seed), ""] for seed, name in enumerate("abc")}
        data = json.loads(json.dumps(_screen(texts).to_dict()))
        assert data["rows"] == 2
        assert [m["name"] for m in data["models"]] == ["a", "b", "c"]
        assert set(data["models"][0]) >= {"loop", "long", "text_on_blank", "peer_agreement"}
        assert data["thresholds"]["loop_tail_compression"] == 0.15
        assert data["models"][0]["loop"]["char_run_rows"] == []
        assert data["models"][0]["loop"]["counting_rows"] == []
        assert data["thresholds"]["counting_run"] == COUNTING_RUN
