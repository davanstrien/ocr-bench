"""Tests for the pairwise judge module."""

import json

import pytest
from PIL import Image

from ocr_bench.judge import (
    CRITERIA_PROFILES,
    DEFAULT_CRITERIA,
    PAIRWISE_PROMPT,
    Comparison,
    build_comparisons,
    build_messages,
    build_prompt,
    image_to_base64,
    is_sentinel,
    normalize_for_judge,
    parse_judge_output,
    prompt_hash,
    validate_prompt_template,
)

# sha256(prompt template)[:12] — pins the exact bytes of each criteria profile.
# If a prompt is edited, the matching test fails loudly; a new value is
# intentional churn, not an accident. The DEFAULT hash must never change (it is
# byte-identical to the pre-#44 hardcoded prompt); update the table-fidelity one
# deliberately when its prompt changes.
DEFAULT_PROMPT_HASH = "8d86832723b5"
TABLE_FIDELITY_PROMPT_HASH = "fe138e71ecc3"


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


class TestNormalizeForJudge:
    def test_plain_text_unchanged(self):
        text = "Just some plain OCR text with no markup at all."
        assert normalize_for_judge(text) == text

    def test_markdown_untouched(self):
        # Headers, bold, lists, and pipe-tables have no HTML tags → no-op.
        md = (
            "# Title\n\nSome **bold** text.\n\n- item one\n- item two\n\n"
            "| a | b |\n|---|---|\n| 1 | 2 |"
        )
        assert normalize_for_judge(md) == md

    def test_empty_string(self):
        assert normalize_for_judge("") == ""

    def test_table_cells_become_delimited(self):
        html = "<table><tr><td>Name</td><td>Age</td></tr><tr><td>Alice</td><td>30</td></tr></table>"
        out = normalize_for_judge(html)
        assert out == "Name | Age\nAlice | 30"

    def test_bare_cells_delimited(self):
        # Requirement's canonical example: <td>x</td><td>y</td> → delimited.
        assert normalize_for_judge("<td>x</td><td>y</td>") == "x | y"

    def test_th_header_cells(self):
        html = "<table><tr><th>Col1</th><th>Col2</th></tr><tr><td>a</td><td>b</td></tr></table>"
        out = normalize_for_judge(html)
        assert out == "Col1 | Col2\na | b"

    def test_cell_boundaries_preserved(self):
        # The whole point of table fidelity: two cells stay two cells.
        out = normalize_for_judge("<tr><td>left</td><td>right</td></tr>")
        assert "left" in out and "right" in out
        assert "|" in out  # boundary survived
        assert "leftright" not in out

    def test_br_becomes_newline(self):
        assert normalize_for_judge("line one<br>line two") == "line one\nline two"

    def test_self_closing_br(self):
        assert normalize_for_judge("line one<br/>line two") == "line one\nline two"

    def test_other_tags_dropped(self):
        assert normalize_for_judge("<b>bold</b> and <i>italic</i>") == "bold and italic"

    def test_span_and_div_content_kept(self):
        out = normalize_for_judge("<div><span>alpha</span> beta</div>")
        assert "alpha" in out and "beta" in out
        assert "<" not in out

    def test_entities_unescaped(self):
        assert normalize_for_judge("<p>Rock &amp; Roll &lt;3</p>") == "Rock & Roll <3"

    def test_numeric_entities(self):
        assert normalize_for_judge("<p>caf&#233;</p>") == "café"

    def test_whitespace_collapsed_conservatively(self):
        out = normalize_for_judge("<p>too    many     spaces</p>")
        assert out == "too many spaces"

    def test_blank_lines_collapsed(self):
        out = normalize_for_judge("<div><p>a</p><p>b</p></div>")
        assert out == "a\nb"

    def test_garbage_never_raises(self):
        # Malformed / adversarial input must return a string, not raise.
        for junk in ["<<<>>>", "<td><td><td", "<table><tr><td>unclosed", "< not a tag >", "<>"]:
            result = normalize_for_judge(junk)
            assert isinstance(result, str)

    def test_math_expression_not_treated_as_html(self):
        # "a < b and c > d" has no letter immediately after '<', so it is not
        # mistaken for a tag and passes through unchanged.
        text = "a < b and c > d"
        assert normalize_for_judge(text) == text

    def test_html_heavy_shorter_than_raw(self):
        # A table's normalized form must be shorter than its raw markup.
        html = (
            "<table>"
            + "".join(f"<tr><td>row{i}col1</td><td>row{i}col2</td></tr>" for i in range(20))
            + "</table>"
        )
        assert len(normalize_for_judge(html)) < len(html)

    # --- Fix 1: GLAM transcription tokens and autolinks are not markup ---

    def test_transcription_tokens_alone_untouched(self):
        # <illegible>/<torn>/<gap> are transcription conventions, not HTML —
        # with no known HTML tag present the text passes through verbatim.
        for text in [
            "the <illegible> word",
            "a page with a <torn> corner",
            "text with a <gap/> here",
            "<gap reason='torn'/> at line start",
        ]:
            assert normalize_for_judge(text) == text

    def test_transcription_token_survives_inside_real_html(self):
        # Inside genuine HTML, an unknown-named tag is re-emitted verbatim, not
        # dropped — so a <torn> in an HTML-table cell survives.
        out = normalize_for_judge("<p>The word <torn> is unclear</p>")
        assert out == "The word <torn> is unclear"
        out2 = normalize_for_judge("<table><tr><td>a</td><td><illegible></td></tr></table>")
        assert "<illegible>" in out2
        assert out2 == "a | <illegible>"

    def test_autolink_alone_untouched(self):
        text = "see <https://example.com/page> for details"
        assert normalize_for_judge(text) == text

    def test_autolink_survives_inside_real_html(self):
        out = normalize_for_judge("<p>See <https://example.com></p>")
        assert out == "See <https://example.com>"

    def test_unknown_tag_with_attrs_reemitted_verbatim(self):
        # Casing and attributes of an unknown tag are preserved via
        # get_starttag_text().
        out = normalize_for_judge("<p>x <Unclear cert='low'> y</p>")
        assert "<Unclear cert='low'>" in out

    # --- Fix 2: script/style/template content is not visible text ---

    def test_style_content_suppressed(self):
        out = normalize_for_judge("<style>.foo { color: red; }</style>Visible text")
        assert out == "Visible text"

    def test_script_content_suppressed(self):
        out = normalize_for_judge("<p>Before</p><script>alert('x');</script><p>After</p>")
        assert out == "Before\nAfter"

    def test_template_content_suppressed(self):
        out = normalize_for_judge(
            "<p>Before</p><template><td>hidden</td></template><p>After</p>"
        )
        assert out == "Before\nAfter"

    # --- Fix 3: never-raises honours non-str input ---

    def test_none_input_returns_empty(self):
        assert normalize_for_judge(None) == ""

    def test_bytes_input_safe(self):
        # bytes decode to text and normalize; no raise.
        assert normalize_for_judge(b"<td>x</td>") == "x"
        assert normalize_for_judge(b"plain bytes, no tags") == "plain bytes, no tags"

    def test_non_str_non_bytes_safe(self):
        # An unexpected type must not raise (str() fallback).
        result = normalize_for_judge(12345)  # type: ignore[arg-type]
        assert isinstance(result, str)


class TestCriteriaProfiles:
    def test_default_profile_present(self):
        assert DEFAULT_CRITERIA == "default"
        assert "default" in CRITERIA_PROFILES

    def test_default_profile_byte_equal(self):
        """The default profile must be the original prompt, byte-for-byte."""
        assert prompt_hash(CRITERIA_PROFILES["default"]) == DEFAULT_PROMPT_HASH

    def test_table_fidelity_profile_present(self):
        assert "table-fidelity" in CRITERIA_PROFILES

    def test_table_fidelity_byte_equal(self):
        """Pin the table-fidelity prompt bytes; edits must update this hash."""
        assert prompt_hash(CRITERIA_PROFILES["table-fidelity"]) == TABLE_FIDELITY_PROMPT_HASH

    def test_table_fidelity_criteria_ordering(self):
        """Table fidelity is criterion 3 (below completeness, above accuracy);
        accuracy and reading order shift to 4 and 5. Position now matches the
        intended severity, so no override language is needed."""
        tf = CRITERIA_PROFILES["table-fidelity"]
        for label in (
            "1. Faithfulness",
            "2. Completeness",
            "3. Table fidelity",
            "4. Accuracy",
            "5. Reading order",
        ):
            assert label in tf
        # Ordering holds positionally, and the old severity-override phrasing is gone.
        assert tf.index("2. Completeness") < tf.index("3. Table fidelity")
        assert tf.index("3. Table fidelity") < tf.index("4. Accuracy")
        assert "rank it just below completeness" not in tf

    def test_table_fidelity_criterion_content(self):
        tf = CRITERIA_PROFILES["table-fidelity"]
        assert "row and column" in tf
        assert "significant error" in tf
        # Markup style stays neutral — the relationships are judged, not syntax.
        assert "plain-text table" in tf
        assert "<|ref|>" in tf  # bbox-tag-ignore note retained
        # The default's structure-neutralising criterion 5 is gone.
        assert "5. Formatting" not in tf

    def test_table_fidelity_tie_line_covers_non_table_pages(self):
        """The tie line must not require table structure on pages without tables;
        it gates on tables only 'where tables are present'."""
        tf = CRITERIA_PROFILES["table-fidelity"]
        assert "where tables are present" in tf

    def test_profiles_have_distinct_prompts(self):
        assert CRITERIA_PROFILES["default"] != CRITERIA_PROFILES["table-fidelity"]

    def test_all_profiles_have_placeholders(self):
        for tmpl in CRITERIA_PROFILES.values():
            assert "{ocr_text_a}" in tmpl
            assert "{ocr_text_b}" in tmpl


class TestPromptHash:
    def test_is_12_hex_chars(self):
        h = prompt_hash("anything")
        assert len(h) == 12
        assert all(c in "0123456789abcdef" for c in h)

    def test_deterministic(self):
        assert prompt_hash("same") == prompt_hash("same")

    def test_differs_between_profiles(self):
        assert prompt_hash(CRITERIA_PROFILES["default"]) != prompt_hash(
            CRITERIA_PROFILES["table-fidelity"]
        )


class TestValidatePromptTemplate:
    def test_valid_template_passes(self):
        validate_prompt_template("Judge A: {ocr_text_a} vs B: {ocr_text_b}")  # no raise

    def test_builtin_profiles_pass(self):
        for tmpl in CRITERIA_PROFILES.values():
            validate_prompt_template(tmpl)  # includes escaped {{ }} JSON example

    def test_escaped_braces_allowed(self):
        validate_prompt_template('{ocr_text_a} {ocr_text_b} reply {{"winner": "A"}}')

    def test_missing_ocr_text_a(self):
        with pytest.raises(ValueError, match=r"\{ocr_text_a\}"):
            validate_prompt_template("only b: {ocr_text_b}")

    def test_missing_ocr_text_b(self):
        with pytest.raises(ValueError, match=r"\{ocr_text_b\}"):
            validate_prompt_template("only a: {ocr_text_a}")

    def test_missing_both(self):
        with pytest.raises(ValueError, match="missing required placeholder"):
            validate_prompt_template("no placeholders here")

    def test_unknown_format_field(self):
        with pytest.raises(ValueError, match="not a valid format string"):
            validate_prompt_template("{ocr_text_a} {ocr_text_b} {surprise}")

    def test_stray_unescaped_brace(self):
        with pytest.raises(ValueError, match="escape any literal brace"):
            validate_prompt_template("{ocr_text_a} {ocr_text_b} and a lone {")


class TestBuildPrompt:
    def test_not_swapped(self):
        prompt, swapped, trunc_a, trunc_b = build_prompt("text A", "text B", swapped=False)
        assert "text A" in prompt
        assert "text B" in prompt
        assert not swapped
        assert not trunc_a
        assert not trunc_b
        # A should appear before B in the prompt
        assert prompt.index("text A") < prompt.index("text B")

    def test_swapped(self):
        prompt, swapped, _, _ = build_prompt("text A", "text B", swapped=True)
        assert swapped
        # When swapped, B text appears in the A position
        assert prompt.index("text B") < prompt.index("text A")

    def test_truncates_long_text(self):
        long_text = "x" * 5000
        prompt, _, trunc_a, trunc_b = build_prompt(long_text, "short", swapped=False)
        # The full 5000-char string should not appear
        assert "x" * 5000 not in prompt
        # But 2500 chars should
        assert "x" * 2500 in prompt
        # The truncated side is flagged; the short side is not
        assert trunc_a
        assert not trunc_b

    def test_truncation_note_is_harness_voice_outside_output_block(self):
        # The disclosure is a harness-voice evaluator note, NOT a marker spliced
        # into the fenced output text (which the judge could read as model-added
        # commentary and penalise).
        prompt, _, trunc_a, _ = build_prompt("x" * 5000, "short", swapped=False)
        assert trunc_a
        assert "Evaluator note: Output A above was truncated" in prompt
        assert "evaluation harness" in prompt
        # The note sits after the output blocks, before the JSON instruction —
        # never between an output's fences.
        assert prompt.index("Evaluator note") < prompt.index("Respond with JSON only")
        # The raw truncated text carries no inline marker.
        assert "[..." not in prompt
        assert "continues beyond" not in prompt

    def test_no_truncation_note_when_untruncated(self):
        prompt, _, trunc_a, trunc_b = build_prompt("short a", "short b", swapped=False)
        assert not trunc_a and not trunc_b
        assert "Evaluator note" not in prompt

    def test_truncation_note_tracks_display_position_when_swapped(self):
        # Original A is truncated; when displayed swapped it occupies position B,
        # so the note must reference "Output B", while the returned flags stay in
        # original order (truncated_a=True).
        prompt, swapped, trunc_a, trunc_b = build_prompt("x" * 5000, "short", swapped=True)
        assert swapped and trunc_a and not trunc_b
        assert "Output B above was truncated" in prompt
        assert "Output A above was truncated" not in prompt

    def test_pairwise_prompt_constant_unmodified(self):
        # The note is composed after .format(); the prompt constant stays
        # byte-stable (no truncation machinery leaked into it).
        assert "Evaluator note" not in PAIRWISE_PROMPT
        assert "truncated" not in PAIRWISE_PROMPT

    def test_truncation_flags_track_original_order_when_swapped(self):
        # A is long (truncated), B is short. Even when displayed swapped, the
        # flags stay keyed to the original A/B texts.
        _, swapped, trunc_a, trunc_b = build_prompt("x" * 5000, "short", swapped=True)
        assert swapped
        assert trunc_a
        assert not trunc_b

    def test_configurable_cap(self):
        prompt, _, trunc_a, _ = build_prompt("y" * 100, "short", swapped=False, max_len=50)
        assert trunc_a
        assert "y" * 50 in prompt
        assert "y" * 51 not in prompt

    def test_html_normalized_before_cap(self):
        # 60 chars of markup wrapping 8 chars of content: with a cap of 40 the
        # raw string would truncate mid-markup, but normalized content fits.
        html = "<table><tr><td>Name</td><td>Age</td></tr></table>"
        prompt, _, trunc_a, _ = build_prompt(html, "short", swapped=False, max_len=40)
        assert not trunc_a
        assert "Name | Age" in prompt
        assert "<td>" not in prompt

    def test_normalize_true_is_default(self):
        # Default path flattens HTML (no explicit normalize=).
        prompt, _, _, _ = build_prompt("<td>x</td><td>y</td>", "short", swapped=False)
        assert "x | y" in prompt
        assert "<td>" not in prompt

    def test_raw_mode_skips_normalization(self):
        # normalize=False: the raw HTML markup reaches the prompt untouched.
        html = "<table><tr><td>Name</td><td>Age</td></tr></table>"
        prompt, _, _, _ = build_prompt(html, "short", swapped=False, normalize=False)
        assert html in prompt
        assert "Name | Age" not in prompt

    def test_raw_mode_caps_markup(self):
        # In raw mode the cap counts markup chars — long HTML truncates where
        # normalized content would have fit. 5 rows: raw ~125 chars, normalized
        # ~24 chars ("cell" x5 as lines); cap at 60 splits them.
        html = "<table>" + "<tr><td>cell</td></tr>" * 5 + "</table>"
        _, _, trunc_norm, _ = build_prompt(html, "short", swapped=False, max_len=60)
        _, _, trunc_raw, _ = build_prompt(html, "short", swapped=False, max_len=60, normalize=False)
        assert not trunc_norm  # flattened content fits under 60
        assert trunc_raw  # raw markup blows the cap

    def test_defaults_to_default_profile(self):
        """With no template arg, build_prompt uses the default criteria."""
        prompt, _, _, _ = build_prompt("a", "b", swapped=False)
        assert "5. Formatting" in prompt
        assert "5. Table fidelity" not in prompt

    def test_threads_explicit_template(self):
        prompt, _, _, _ = build_prompt(
            "a", "b", swapped=False, prompt_template=CRITERIA_PROFILES["table-fidelity"]
        )
        assert "3. Table fidelity" in prompt
        assert "a" in prompt
        assert "b" in prompt

    def test_custom_template_gets_truncation_disclosure(self):
        prompt, _, truncated_a, _ = build_prompt(
            "x" * 100,
            "b",
            swapped=False,
            prompt_template="Custom rubric\nA={ocr_text_a}\nB={ocr_text_b}",
            max_len=20,
        )
        assert truncated_a
        assert "Evaluator note: Output A above was truncated" in prompt


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

    def test_defaults_to_default_criteria(self):
        ds = self._make_dataset()
        ocr_columns = {"ocr_model_a": "ModelA", "ocr_model_b": "ModelB"}
        # min_chars=0 isolates this from the blank-pair filter (toy text is short)
        comps = build_comparisons(ds, ocr_columns, min_chars=0)
        prompt_text = comps[0].messages[0]["content"][1]["text"]
        assert "5. Formatting" in prompt_text
        assert "5. Table fidelity" not in prompt_text

    def test_threads_prompt_template_into_messages(self):
        """The selected criteria profile reaches each comparison's judge prompt."""
        ds = self._make_dataset()
        ocr_columns = {"ocr_model_a": "ModelA", "ocr_model_b": "ModelB"}
        # min_chars=0 isolates this from the blank-pair filter (toy text is short)
        comps = build_comparisons(
            ds, ocr_columns, prompt_template=CRITERIA_PROFILES["table-fidelity"], min_chars=0
        )
        prompt_text = comps[0].messages[0]["content"][1]["text"]
        assert "3. Table fidelity" in prompt_text
        assert "5. Formatting" not in prompt_text

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

    def test_truncation_flags_default_false(self):
        ds = self._make_dataset()
        ocr_columns = {"ocr_model_a": "ModelA", "ocr_model_b": "ModelB"}
        # min_chars=0 isolates the truncation-flag defaults from the blank-pair
        # filter (the fixture's texts are under the default 20-char threshold).
        comps = build_comparisons(ds, ocr_columns, min_chars=0)
        assert comps[0].truncated_a is False
        assert comps[0].truncated_b is False

    def test_max_ocr_text_len_threaded(self):
        ds = [
            {
                "image": Image.new("RGB", (50, 50)),
                "col_a": "x" * 500,
                "col_b": "short",
            },
        ]
        comps = build_comparisons(ds, {"col_a": "A", "col_b": "B"}, max_ocr_text_len=100)
        assert comps[0].truncated_a is True
        assert comps[0].truncated_b is False

    def test_judge_image_dim_threaded(self):
        from unittest.mock import patch

        ds = [
            {
                "image": Image.new("RGB", (50, 50)),
                "col_a": "text a",
                "col_b": "text b",
            },
        ]
        with patch("ocr_bench.judge.image_to_base64", return_value="b64") as mock_img:
            build_comparisons(
                ds, {"col_a": "A", "col_b": "B"}, judge_image_dim=1536, min_chars=0
            )
            mock_img.assert_called_once()
            assert mock_img.call_args.kwargs["max_dim"] == 1536

    def test_normalize_flag_threaded_to_prompt(self):
        # A cell-heavy HTML output vs a plain one. Under raw mode the HTML markup
        # reaches the prompt untouched; under the default it is flattened.
        html = "<table><tr><td>Alpha</td><td>Beta</td></tr></table>"
        ds = [{"image": Image.new("RGB", (50, 50)), "col_a": html, "col_b": "plain text here"}]
        cols = {"col_a": "A", "col_b": "B"}

        raw = build_comparisons(ds, cols, normalize=False)
        raw_prompt = raw[0].messages[0]["content"][1]["text"]
        assert "<td>" in raw_prompt

        norm = build_comparisons(ds, cols)  # normalize=True default
        norm_prompt = norm[0].messages[0]["content"][1]["text"]
        assert "<td>" not in norm_prompt
        assert "Alpha | Beta" in norm_prompt


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
