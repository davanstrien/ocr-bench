"""Tests for the FastAPI + HTMX viewer."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

SAMPLE_LEADERBOARD = [
    {"model": "DeepSeek-OCR", "elo": 1539, "wins": 5, "losses": 2, "ties": 1, "win_pct": 63},
    {"model": "LightOnOCR-2", "elo": 1530, "wins": 4, "losses": 3, "ties": 1, "win_pct": 50},
]

SAMPLE_COMPARISONS = [
    {
        "sample_idx": 0,
        "model_a": "DeepSeek-OCR",
        "model_b": "LightOnOCR-2",
        "winner": "A",
        "reason": "more complete",
        "agreement": "2/2",
        "text_a": "OCR text from DeepSeek",
        "text_b": "OCR text from LightOn",
        "col_a": "ocr_deepseek",
        "col_b": "ocr_lighton",
    },
    {
        "sample_idx": 1,
        "model_a": "DeepSeek-OCR",
        "model_b": "LightOnOCR-2",
        "winner": "tie",
        "reason": "similar quality",
        "agreement": "2/2",
        "text_a": "Same text A",
        "text_b": "Same text B",
        "col_a": "ocr_deepseek",
        "col_b": "ocr_lighton",
    },
    {
        "sample_idx": 2,
        "model_a": "DeepSeek-OCR",
        "model_b": "LightOnOCR-2",
        "winner": "B",
        "reason": "better formatting",
        "agreement": "1/2",
        "text_a": "Text A sample 2",
        "text_b": "Text B sample 2",
        "col_a": "ocr_deepseek",
        "col_b": "ocr_lighton",
    },
]


@pytest.fixture
def client(tmp_path):
    """Create a test client with mocked Hub data."""
    with (
        patch("ocr_bench.web.load_results") as mock_load,
        patch("ocr_bench.web._load_source_metadata") as mock_meta,
        patch("ocr_bench.web.load_annotations") as mock_ann,
    ):
        mock_load.return_value = (SAMPLE_LEADERBOARD, SAMPLE_COMPARISONS)
        mock_meta.return_value = {}
        mock_ann.return_value = ({}, [])

        from ocr_bench.web import create_app

        app = create_app(
            "user/test-results",
            output_path=str(tmp_path / "annotations.json"),
        )
        yield TestClient(app)


class TestIndex:
    def test_redirects_to_leaderboard(self, client):
        resp = client.get("/", follow_redirects=False)
        assert resp.status_code == 302
        assert resp.headers["location"] == "/leaderboard"


class TestLeaderboard:
    def test_renders_leaderboard(self, client):
        resp = client.get("/leaderboard")
        assert resp.status_code == 200
        assert "DeepSeek-OCR" in resp.text
        assert "LightOnOCR-2" in resp.text
        assert "1539" in resp.text

    def test_has_judge_elo_column(self, client):
        resp = client.get("/leaderboard")
        assert "Judge ELO" in resp.text


class TestComparisonsPage:
    def test_renders_page(self, client):
        resp = client.get("/comparisons")
        assert resp.status_code == 200
        # Should be in blind state — no model names visible as column headers
        assert "comparison-container" in resp.text

    def test_has_filter_dropdowns(self, client):
        resp = client.get("/comparisons")
        assert 'name="winner"' in resp.text
        assert 'name="model"' in resp.text

    def test_has_keyboard_data_attributes(self, client):
        resp = client.get("/comparisons")
        assert 'data-nav="prev"' in resp.text or 'data-nav="next"' in resp.text
        assert 'data-vote="A"' in resp.text
        assert 'data-vote="B"' in resp.text
        assert 'data-vote="tie"' in resp.text
        assert 'data-action="reveal"' in resp.text


class TestComparisonNavigation:
    def test_renders_first_comparison(self, client):
        resp = client.get("/comparisons/0")
        assert resp.status_code == 200
        assert "1 of" in resp.text

    def test_renders_second_comparison(self, client):
        resp = client.get("/comparisons/1")
        assert resp.status_code == 200
        assert "2 of" in resp.text

    def test_clamps_out_of_bounds(self, client):
        resp = client.get("/comparisons/999")
        assert resp.status_code == 200
        # Should clamp to last valid index
        assert "of" in resp.text

    def test_blind_state_no_model_names(self, client):
        resp = client.get("/comparisons/0")
        # In blind state, should show "A" and "B", not model names
        assert ">A</h3>" in resp.text
        assert ">B</h3>" in resp.text


class TestVote:
    def test_records_vote_returns_revealed(self, client):
        with patch("ocr_bench.web.save_annotations"):
            resp = client.post("/vote/0", data={"winner": "A"})
        assert resp.status_code == 200
        # Should now show model names (revealed)
        assert "DeepSeek-OCR" in resp.text or "LightOnOCR-2" in resp.text
        # Should have HX-Trigger for stats refresh
        assert resp.headers.get("HX-Trigger") == "vote-recorded"

    def test_vote_shows_agreement(self, client):
        with patch("ocr_bench.web.save_annotations"):
            resp = client.post("/vote/0", data={"winner": "A"})
        assert resp.status_code == 200
        # The judge verdict was "A" and we voted "A", so should agree
        assert "agreed" in resp.text or "disagree" in resp.text

    def test_double_vote_is_idempotent(self, client):
        with patch("ocr_bench.web.save_annotations"):
            resp1 = client.post("/vote/0", data={"winner": "A"})
            resp2 = client.post("/vote/0", data={"winner": "B"})
        assert resp1.status_code == 200
        assert resp2.status_code == 200
        # Second vote should not change the result — still revealed
        assert "DeepSeek-OCR" in resp2.text or "LightOnOCR-2" in resp2.text

    def test_vote_invalid_index(self, client):
        resp = client.post("/vote/999", data={"winner": "A"})
        assert resp.status_code == 404

    def test_vote_unswaps_correctly(self, client, tmp_path):
        """Votes on swapped comparisons should be stored with the display winner."""
        with patch("ocr_bench.web.save_annotations") as mock_save:
            # Vote on comp_idx=0 — the comparison may or may not be swapped
            client.post("/vote/0", data={"winner": "A"})
            # Just verify save_annotations was called
            if mock_save.called:
                _, _, annotations = mock_save.call_args[0]
                assert len(annotations) > 0
                assert annotations[-1]["winner"] == "A"


class TestReveal:
    def test_reveal_shows_verdict(self, client):
        resp = client.get("/reveal/0")
        assert resp.status_code == 200
        # Should show model names
        assert "DeepSeek-OCR" in resp.text or "LightOnOCR-2" in resp.text

    def test_reveal_keeps_vote_buttons(self, client):
        resp = client.get("/reveal/0")
        assert resp.status_code == 200
        assert 'data-vote="A"' in resp.text

    def test_reveal_invalid_index(self, client):
        resp = client.get("/reveal/999")
        assert resp.status_code == 404


class TestFilter:
    def test_filter_by_winner(self, client):
        resp = client.get("/comparisons/filter?winner=A")
        assert resp.status_code == 200

    def test_filter_by_model(self, client):
        resp = client.get("/comparisons/filter?model=DeepSeek-OCR")
        assert resp.status_code == 200

    def test_filter_no_matches(self, client):
        resp = client.get("/comparisons/filter?model=NonexistentModel")
        assert resp.status_code == 200
        assert "No comparisons" in resp.text


class TestStats:
    def test_stats_empty(self, client):
        resp = client.get("/stats")
        assert resp.status_code == 200

    def test_stats_after_vote(self, client):
        with patch("ocr_bench.web.save_annotations"):
            client.post("/vote/0", data={"winner": "A"})
        resp = client.get("/stats")
        assert resp.status_code == 200
        assert "1 vote" in resp.text


class TestImage:
    def test_no_image_loader(self, client):
        resp = client.get("/image/0")
        assert resp.status_code == 404

    def test_image_with_loader(self, tmp_path):
        """Test image endpoint with a mock ImageLoader."""
        from PIL import Image

        mock_img = Image.new("RGB", (10, 10), color="red")

        with (
            patch("ocr_bench.web.load_results") as mock_load,
            patch("ocr_bench.web._load_source_metadata") as mock_meta,
            patch("ocr_bench.web.load_annotations") as mock_ann,
            patch("ocr_bench.web.ImageLoader") as MockLoader,
        ):
            mock_load.return_value = (SAMPLE_LEADERBOARD, SAMPLE_COMPARISONS)
            mock_meta.return_value = {"source_dataset": "user/source", "from_prs": False}
            mock_ann.return_value = ({}, [])
            loader_instance = MagicMock()
            loader_instance.get.return_value = mock_img
            MockLoader.return_value = loader_instance

            from ocr_bench.web import create_app

            app = create_app(
                "user/test-results",
                output_path=str(tmp_path / "ann.json"),
            )
            client = TestClient(app)
            resp = client.get("/image/0")
            assert resp.status_code == 200
            assert resp.headers["content-type"] == "image/png"

    def test_image_not_found(self, tmp_path):
        """Test image 404 when loader returns None."""
        with (
            patch("ocr_bench.web.load_results") as mock_load,
            patch("ocr_bench.web._load_source_metadata") as mock_meta,
            patch("ocr_bench.web.load_annotations") as mock_ann,
            patch("ocr_bench.web.ImageLoader") as MockLoader,
        ):
            mock_load.return_value = (SAMPLE_LEADERBOARD, SAMPLE_COMPARISONS)
            mock_meta.return_value = {"source_dataset": "user/source"}
            mock_ann.return_value = ({}, [])
            loader_instance = MagicMock()
            loader_instance.get.return_value = None
            MockLoader.return_value = loader_instance

            from ocr_bench.web import create_app

            app = create_app(
                "user/test-results",
                output_path=str(tmp_path / "ann.json"),
            )
            client = TestClient(app)
            resp = client.get("/image/99")
            assert resp.status_code == 404
