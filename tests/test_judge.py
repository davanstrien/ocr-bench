"""Tests for the pairwise judge module."""

import json

from PIL import Image

from ocr_bench.judge import (
    Comparison,
    build_comparisons,
    build_messages,
    build_prompt,
    image_to_base64,
    is_sentinel,
    parse_judge_output,
)


class TestImageToBase64:
    def test_returns_base64_string(self):
        img = Image.new("RGB", (100, 100), color="red")
        b64 = image_to_base64(img)
        assert isinstance(b64, str)
        assert len(b64) > 0

    def test_converts_rgba_to_rgb(self):
        img = Image.new("RGBA", (100, 100), color=(255, 0, 0, 128))
        b64 = image_to_base64(img)
        assert isinstance(b64, str)

    def test_resizes_large_image(self):
        img = Image.new("RGB", (4000, 3000))
        b64 = image_to_base64(img, max_dim=1024)
        # Decode and check size
        import base64
        import io

        decoded = base64.b64decode(b64)
        result = Image.open(io.BytesIO(decoded))
        assert max(result.size) <= 1024

    def test_small_image_not_resized(self):
        img = Image.new("RGB", (200, 100))
        b64 = image_to_base64(img, max_dim=1024)
        import base64
        import io

        decoded = base64.b64decode(b64)
        result = Image.open(io.BytesIO(decoded))
        assert result.size == (200, 100)


class TestBuildPrompt:
    def test_not_swapped(self):
        prompt, swapped = build_prompt("text A", "text B", swapped=False)
        assert "text A" in prompt
        assert "text B" in prompt
        assert not swapped
        # A should appear before B in the prompt
        assert prompt.index("text A") < prompt.index("text B")

    def test_swapped(self):
        prompt, swapped = build_prompt("text A", "text B", swapped=True)
        assert swapped
        # When swapped, B text appears in the A position
        assert prompt.index("text B") < prompt.index("text A")

    def test_truncates_long_text(self):
        long_text = "x" * 5000
        prompt, _ = build_prompt(long_text, "short", swapped=False)
        # The full 5000-char string should not appear
        assert "x" * 5000 not in prompt
        # But 2500 chars should
        assert "x" * 2500 in prompt


class TestBuildMessages:
    def test_message_structure(self):
        msgs = build_messages("abc123", "test prompt")
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"
        content = msgs[0]["content"]
        assert len(content) == 2
        assert content[0]["type"] == "image_url"
        assert "abc123" in content[0]["image_url"]["url"]
        assert content[1]["type"] == "text"
        assert content[1]["text"] == "test prompt"


class TestParseJudgeOutput:
    def test_valid_json(self):
        text = json.dumps({"winner": "A", "reason": "better accuracy"})
        result = parse_judge_output(text)
        assert result["winner"] == "A"
        assert result["reason"] == "better accuracy"

    def test_winner_b(self):
        result = parse_judge_output('{"winner": "B", "reason": "more complete"}')
        assert result["winner"] == "B"

    def test_tie(self):
        result = parse_judge_output('{"winner": "tie", "reason": "similar quality"}')
        assert result["winner"] == "tie"

    def test_uppercase_tie(self):
        result = parse_judge_output('{"winner": "TIE", "reason": "equal"}')
        assert result["winner"] == "tie"

    def test_lowercase_winner(self):
        result = parse_judge_output('{"winner": "a", "reason": "test"}')
        assert result["winner"] == "A"

    def test_markdown_fences(self):
        text = '```json\n{"winner": "A", "reason": "test"}\n```'
        result = parse_judge_output(text)
        assert result["winner"] == "A"

    def test_invalid_winner_defaults_to_tie(self):
        result = parse_judge_output('{"winner": "C", "reason": "confused"}')
        assert result["winner"] == "tie"

    def test_invalid_json_returns_empty(self):
        result = parse_judge_output("not json at all")
        assert result == {}

    def test_missing_reason(self):
        result = parse_judge_output('{"winner": "A"}')
        assert result["winner"] == "A"
        assert result["reason"] == ""

    def test_null_winner_returns_empty(self):
        """Valid JSON with a non-string winner is a failure, not a crash."""
        result = parse_judge_output('{"winner": null, "reason": "unsure"}')
        assert result == {}

    def test_list_winner_returns_empty(self):
        result = parse_judge_output('{"winner": ["A"], "reason": "odd"}')
        assert result == {}

    def test_non_object_json_returns_empty(self):
        assert parse_judge_output('"A"') == {}
        assert parse_judge_output("[1, 2, 3]") == {}
        assert parse_judge_output("42") == {}

    def test_non_string_reason_coerced(self):
        result = parse_judge_output('{"winner": "A", "reason": {"note": "x"}}')
        assert result["winner"] == "A"
        assert isinstance(result["reason"], str)


class TestIsSentinel:
    def test_known_literals(self):
        assert is_sentinel("[OCR ERROR]")
        assert is_sentinel("[OCR FAILED]")

    def test_known_literals_case_insensitive(self):
        assert is_sentinel("[ocr error]")
        assert is_sentinel("[Ocr Failed]")

    def test_surrounding_whitespace(self):
        assert is_sentinel("  [OCR ERROR]\n")

    def test_census_variants_match(self):
        # Per-script sentinel formats seen in the uv-scripts census.
        for variant in (
            "[OCR ERROR]",
            "[OCR FAILED]",
            "[SURYA LAYOUT ERROR]",
            "[LIFT LOAD ERROR]",
            "[GOT-OCR FAILED]",
            "[DOTS.OCR ERROR]",
        ):
            assert is_sentinel(variant), variant

    def test_normal_text_is_not_sentinel(self):
        assert not is_sentinel("The quick brown fox")
        assert not is_sentinel("Page 3 of the manuscript")

    def test_empty_and_none_are_not_sentinels(self):
        # Empty/None are "missing" but handled separately from sentinels.
        assert not is_sentinel("")
        assert not is_sentinel("   ")
        assert not is_sentinel(None)

    def test_lowercase_prose_mentioning_error_is_not_sentinel(self):
        assert not is_sentinel("an error occurred while reading the page")
        assert not is_sentinel("[the scan failed to load, see appendix]")

    def test_archival_heading_with_nonfinal_keyword_is_not_sentinel(self):
        # ALL-CAPS bracketed archival headings that merely CONTAIN ERROR/FAILED
        # (not as the final word) must not be flagged (review finding #3).
        assert not is_sentinel("[SECTION FAILED BANKS 1866]")
        assert not is_sentinel("[ERROR REPORT OF THE YEAR 1877]")

    def test_wordy_or_long_bracket_ending_in_keyword_is_not_sentinel(self):
        # Ends in ERROR but is a >4-word heading — not a terse sentinel.
        assert not is_sentinel("[NOTE ON THE FATAL ERROR]")  # 5 words
        # A long transcription that merely opens with a bracket.
        long_text = "[NOTICE] " + "the page reads " * 20
        assert len(long_text) > 40
        assert not is_sentinel(long_text)

    def test_bracket_token_must_be_whole_string(self):
        # A sentinel embedded in real text is not a whole-column failure.
        assert not is_sentinel("Title page\n[OCR ERROR]\nmore text")


class TestComparison:
    def test_text_fields_default_empty(self):
        comp = Comparison(
            sample_idx=0,
            model_a="a",
            model_b="b",
            col_a="col_a",
            col_b="col_b",
            swapped=False,
            messages=[],
        )
        assert comp.text_a == ""
        assert comp.text_b == ""

    def test_text_fields_set(self):
        comp = Comparison(
            sample_idx=0,
            model_a="a",
            model_b="b",
            col_a="col_a",
            col_b="col_b",
            swapped=False,
            messages=[],
            text_a="hello",
            text_b="world",
        )
        assert comp.text_a == "hello"
        assert comp.text_b == "world"


class TestBuildComparisons:
    def _make_dataset(self):
        """Create a minimal fake dataset (list of dicts with PIL images)."""
        return [
            {
                "image": Image.new("RGB", (100, 100), color="red"),
                "ocr_model_a": "text from model A",
                "ocr_model_b": "text from model B",
            },
        ]

    def test_comparisons_have_text_fields(self):
        ds = self._make_dataset()
        ocr_columns = {"ocr_model_a": "ModelA", "ocr_model_b": "ModelB"}
        # min_chars=0 isolates this from the blank-pair filter (toy text is short)
        comps = build_comparisons(ds, ocr_columns, min_chars=0)
        assert len(comps) == 1
        comp = comps[0]
        assert comp.text_a == "text from model A"
        assert comp.text_b == "text from model B"

    def test_skips_empty_text(self):
        ds = [
            {
                "image": Image.new("RGB", (100, 100)),
                "ocr_a": "",
                "ocr_b": "has text",
            },
        ]
        comps = build_comparisons(ds, {"ocr_a": "A", "ocr_b": "B"})
        assert len(comps) == 0

    def test_skips_sentinel_side_like_empty(self):
        """A sentinel output is treated as missing — the pair is excluded even
        though the other side is a long, valid transcription."""
        ds = [
            {
                "image": Image.new("RGB", (100, 100)),
                "ocr_a": "[OCR ERROR]",
                "ocr_b": "a proper full transcription of the whole page here",
            },
        ]
        comps = build_comparisons(ds, {"ocr_a": "A", "ocr_b": "B"})
        assert len(comps) == 0

    def test_sentinel_excludes_only_affected_pairs(self):
        """With 3 models where one is a sentinel, only the pair between the two
        valid models survives (texts are long enough to clear min_chars)."""
        ds = [
            {
                "image": Image.new("RGB", (60, 60)),
                "col_a": "[SURYA LAYOUT ERROR]",
                "col_b": "a full transcription from model b here",
                "col_c": "a different transcription from model c",
            },
        ]
        ocr_columns = {"col_a": "A", "col_b": "B", "col_c": "C"}
        comps = build_comparisons(ds, ocr_columns)
        pair_set = {(c.model_a, c.model_b) for c in comps}
        assert pair_set == {("B", "C")}

    def test_max_samples(self):
        ds = [
            {
                "image": Image.new("RGB", (50, 50)),
                "col_a": f"text A {i}",
                "col_b": f"text B {i}",
            }
            for i in range(10)
        ]
        comps = build_comparisons(ds, {"col_a": "A", "col_b": "B"}, max_samples=3, min_chars=0)
        assert len(comps) == 3

    def test_skip_samples_excludes_judged_pair_sample(self):
        """skip_samples excludes the exact (pair, sample) already judged."""
        ds = [
            {
                "image": Image.new("RGB", (50, 50)),
                "col_a": "text a",
                "col_b": "text b",
                "col_c": "text c",
            },
        ]
        ocr_columns = {"col_a": "ModelA", "col_b": "ModelB", "col_c": "ModelC"}
        # 3 models = 3 pairs; sample 0 of (A, B) already judged. min_chars=0
        # keeps the toy text (this exercises skip_samples, not the blank filter).
        comps = build_comparisons(
            ds, ocr_columns, skip_samples={("ModelA", "ModelB"): {0}}, min_chars=0
        )
        pair_set = {(c.model_a, c.model_b) for c in comps}
        assert ("ModelA", "ModelB") not in pair_set
        assert len(comps) == 2  # ModelA-ModelC, ModelB-ModelC

    def test_skip_samples_symmetric(self):
        """Skipping (A, B) sample 0 should also skip (B, A) sample 0."""
        ds = [
            {
                "image": Image.new("RGB", (50, 50)),
                "col_a": "text a",
                "col_b": "text b",
            },
        ]
        ocr_columns = {"col_a": "ModelA", "col_b": "ModelB"}
        # Skip in reverse pair order.
        comps = build_comparisons(
            ds, ocr_columns, skip_samples={("ModelB", "ModelA"): {0}}
        )
        assert len(comps) == 0

    def test_skip_samples_tops_up_unjudged_samples(self):
        """A pair judged on only some samples is still built for the rest.

        This is the resume top-up guarantee: (pair, sample)-level skip, not
        pair-level — a partially-judged pair is not frozen forever.
        """
        ds = [
            {
                "image": Image.new("RGB", (50, 50)),
                "col_a": f"text a {i}",
                "col_b": f"text b {i}",
            }
            for i in range(4)
        ]
        ocr_columns = {"col_a": "ModelA", "col_b": "ModelB"}
        # (A, B) judged on samples 0 and 1 only; expect 2 and 3 to remain.
        # min_chars=0 keeps the toy text (orthogonal to the blank filter).
        comps = build_comparisons(
            ds, ocr_columns, skip_samples={("ModelA", "ModelB"): {0, 1}}, min_chars=0
        )
        judged = {c.sample_idx for c in comps}
        assert judged == {2, 3}

    def test_skip_samples_none_includes_all(self):
        """Default skip_samples=None should include all pairs."""
        ds = [
            {
                "image": Image.new("RGB", (50, 50)),
                "col_a": "text a",
                "col_b": "text b",
                "col_c": "text c",
            },
        ]
        ocr_columns = {"col_a": "A", "col_b": "B", "col_c": "C"}
        comps = build_comparisons(ds, ocr_columns, skip_samples=None, min_chars=0)
        assert len(comps) == 3  # All C(3,2) pairs

    def test_skip_all_pairs_skips_image_encoding(self):
        """When all pairs for a row are skipped, image_to_base64 is not called."""
        ds = [
            {
                "image": Image.new("RGB", (50, 50)),
                "col_a": "text a",
                "col_b": "text b",
            },
        ]
        ocr_columns = {"col_a": "ModelA", "col_b": "ModelB"}
        from unittest.mock import patch

        with patch("ocr_bench.judge.image_to_base64") as mock_img:
            build_comparisons(
                ds, ocr_columns, skip_samples={("ModelA", "ModelB"): {0}}
            )
            mock_img.assert_not_called()


class TestBlankPairFiltering:
    """min_chars: skip pairs where neither model produced meaningful text."""

    LONG_A = "This is a full page of transcribed printed text."
    LONG_B = "An entirely different long transcription of the page."

    def _ds(self, text_a: str, text_b: str):
        return [{"image": Image.new("RGB", (50, 50)), "col_a": text_a, "col_b": text_b}]

    def test_both_below_threshold_skipped(self):
        comps = build_comparisons(self._ds("short", "tiny"), {"col_a": "A", "col_b": "B"})
        assert comps == []

    def test_both_empty_skipped(self):
        comps = build_comparisons(self._ds("", ""), {"col_a": "A", "col_b": "B"})
        assert comps == []

    def test_one_long_one_blank_kept(self):
        # Empty vs full page is a real signal (the full-text model should win),
        # so it must NOT be filtered — only both-below-threshold is skipped.
        comps = build_comparisons(self._ds("", self.LONG_A), {"col_a": "A", "col_b": "B"})
        assert len(comps) == 1
        assert comps[0].auto_result is None

    def test_both_long_distinct_kept_for_judging(self):
        comps = build_comparisons(
            self._ds(self.LONG_A, self.LONG_B), {"col_a": "A", "col_b": "B"}
        )
        assert len(comps) == 1
        assert comps[0].auto_result is None
        assert comps[0].messages  # image + prompt built

    def test_custom_min_chars_threshold(self):
        # "hello world" (11 chars) survives a threshold of 5 but not the default.
        ds = self._ds("hello world", "goodbye moon")
        assert build_comparisons(ds, {"col_a": "A", "col_b": "B"}, min_chars=5) != []
        assert build_comparisons(ds, {"col_a": "A", "col_b": "B"}, min_chars=20) == []


class TestAutoTie:
    """Identical outputs become an auto-tie without a judge call."""

    LONG = "This is a full page of transcribed printed text, identical."

    def _ds(self, text_a: str, text_b: str):
        return [{"image": Image.new("RGB", (50, 50)), "col_a": text_a, "col_b": text_b}]

    def test_identical_outputs_auto_tie_shape(self):
        comps = build_comparisons(self._ds(self.LONG, self.LONG), {"col_a": "A", "col_b": "B"})
        assert len(comps) == 1
        comp = comps[0]
        assert comp.auto_result == {
            "winner": "tie",
            "reason": "identical outputs — auto-tie",
            "agreement": "auto",
        }
        assert comp.messages == []  # never sent to the judge
        assert comp.swapped is False

    def test_identical_after_strip(self):
        # Whitespace-only differences still count as identical.
        comps = build_comparisons(
            self._ds(self.LONG, f"  {self.LONG}\n"), {"col_a": "A", "col_b": "B"}
        )
        assert len(comps) == 1
        assert comps[0].auto_result is not None

    def test_all_auto_tie_row_skips_image_encoding(self):
        from unittest.mock import patch

        with patch("ocr_bench.judge.image_to_base64") as mock_img:
            comps = build_comparisons(
                self._ds(self.LONG, self.LONG), {"col_a": "A", "col_b": "B"}
            )
            mock_img.assert_not_called()
        assert comps[0].auto_result is not None

    def test_identical_but_short_is_skipped_not_auto_tie(self):
        # Both below min_chars wins over the identical check — a blank/near-blank
        # page shouldn't pad the tie count.
        comps = build_comparisons(self._ds("abc", "abc"), {"col_a": "A", "col_b": "B"})
        assert comps == []

    def test_mixed_row_encodes_image_once(self):
        # One identical pair (auto-tie) + one distinct pair (judged) on the same
        # row: the image is still needed for the judged pair.
        ds = [
            {
                "image": Image.new("RGB", (50, 50)),
                "col_a": self.LONG,
                "col_b": self.LONG,
                "col_c": "A different long transcription for the third model here.",
            }
        ]
        comps = build_comparisons(ds, {"col_a": "A", "col_b": "B", "col_c": "C"})
        autos = [c for c in comps if c.auto_result is not None]
        judged = [c for c in comps if c.auto_result is None]
        assert len(autos) == 1  # A vs B identical
        assert len(judged) == 2  # A vs C, B vs C
        assert all(c.messages for c in judged)
