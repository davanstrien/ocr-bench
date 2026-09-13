"""Tests for the Kat57 PAGE XML conversion experiment."""

from __future__ import annotations

import io
import zipfile

import pytest
from PIL import Image as PILImage

from experiments.kat57.prepare import (
    CardPair,
    card_row,
    deterministic_sample,
    discover_pairs,
    parse_page_xml,
)

PAGE_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15">
  <Page imageFilename="kat57-00145-00204.png" imageWidth="10" imageHeight="10">
    <TextRegion id="r1">
      <TextLine id="l1"><TextEquiv><Unicode>Caf&#233;</Unicode></TextEquiv></TextLine>
      <TextLine id="l2"><TextEquiv><Unicode>Second line</Unicode></TextEquiv></TextLine>
    </TextRegion>
  </Page>
</PcGts>
"""


def png_bytes() -> bytes:
    output = io.BytesIO()
    PILImage.new("RGB", (10, 10), "white").save(output, format="PNG")
    return output.getvalue()


def make_archive(tmp_path, *, include_image=True, include_xml=True):
    path = tmp_path / "kat57.zip"
    with zipfile.ZipFile(path, "w") as archive:
        if include_image:
            archive.writestr("kat57-gt-dataset/images/kat57-00145-00204.png", png_bytes())
        if include_xml:
            archive.writestr("kat57-gt-dataset/pagexml/kat57-00145-00204.xml", PAGE_XML)
    return path


def test_parse_page_xml_preserves_document_order():
    lines, reference = parse_page_xml(PAGE_XML)
    assert lines == ["Café", "Second line"]
    assert reference == "Café\nSecond line"


def test_discover_pairs_and_build_row(tmp_path):
    with zipfile.ZipFile(make_archive(tmp_path)) as archive:
        pairs = discover_pairs(archive)
        assert len(pairs) == 1
        row = card_row(archive, pairs[0])

    assert row["id"] == "kat57-00145-00204"
    assert row["drawer_id"] == "00145"
    assert row["card_id"] == "00204"
    assert row["reference"] == "Café\nSecond line"
    assert row["line_count"] == 2
    assert row["image"]["bytes"].startswith(b"\x89PNG")


@pytest.mark.parametrize(("include_image", "include_xml"), [(False, True), (True, False)])
def test_discover_pairs_rejects_unmatched_members(tmp_path, include_image, include_xml):
    with zipfile.ZipFile(
        make_archive(tmp_path, include_image=include_image, include_xml=include_xml)
    ) as archive:
        with pytest.raises(ValueError, match="archive has"):
            discover_pairs(archive)


def test_deterministic_sample_is_reproducible_and_returned_in_source_order():
    pairs = [CardPair(f"kat57-00001-{i:05d}", f"{i}.png", f"{i}.xml") for i in range(20)]

    first = deterministic_sample(pairs, 5, seed=57)
    second = deterministic_sample(list(reversed(pairs)), 5, seed=57)
    different = deterministic_sample(pairs, 5, seed=58)

    assert first == second
    assert first == sorted(first, key=lambda pair: pair.card_id)
    assert first != different


@pytest.mark.parametrize("sample_size", [0, 21])
def test_deterministic_sample_rejects_invalid_size(sample_size):
    pairs = [CardPair("kat57-00001-00001", "1.png", "1.xml")]
    with pytest.raises(ValueError, match="Sample size"):
        deterministic_sample(pairs, sample_size, seed=57)
