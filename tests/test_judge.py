"""Tests for the pairwise judge module."""

import json

from PIL import Image

from ocr_bench.judge import (
    Comparison,
    build_comparisons,
    build_messages,
    build_prompt,
    image_to_base64,
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
        comps = build_comparisons(ds, ocr_columns)
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

    def test_max_samples(self):
        ds = [
            {
                "image": Image.new("RGB", (50, 50)),
                "col_a": f"text {i}",
                "col_b": f"text {i}",
            }
            for i in range(10)
        ]
        comps = build_comparisons(ds, {"col_a": "A", "col_b": "B"}, max_samples=3)
        assert len(comps) == 3
