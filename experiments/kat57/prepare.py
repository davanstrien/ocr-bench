# /// script
# requires-python = ">=3.11"
# dependencies = ["datasets>=4.0.0", "huggingface-hub>=0.30", "pillow"]
# ///
"""Convert Kat57's paired PNG/PAGE XML archive to a Hugging Face dataset."""

from __future__ import annotations

import argparse
import hashlib
import os
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

from datasets import Dataset, Features, Image, Sequence, Value
from huggingface_hub import HfApi, get_token

ARCHIVE_URL = "https://zenodo.org/api/records/14679534/files/kat57-gt-dataset.zip/content"
ARCHIVE_MD5 = "0895d5785cd0ddf9f9c6488484ac447e"

FEATURES = Features(
    {
        "id": Value("string"),
        "drawer_id": Value("string"),
        "card_id": Value("string"),
        "image": Image(),
        "reference": Value("string"),
        "lines": Sequence(Value("string")),
        "line_count": Value("int32"),
        "page_xml": Value("string"),
        "source_image": Value("string"),
    }
)


@dataclass(frozen=True)
class CardPair:
    card_id: str
    image_member: str
    xml_member: str


def parse_page_xml(xml_bytes: bytes) -> tuple[list[str], str]:
    """Extract each PAGE TextLine transcription in document order."""
    root = ElementTree.fromstring(xml_bytes)
    lines = []
    for element in root.iter():
        if element.tag.rsplit("}", 1)[-1] != "TextLine":
            continue
        text = next(
            (
                child.text or ""
                for child in element.iter()
                if child.tag.rsplit("}", 1)[-1] == "Unicode"
            ),
            "",
        )
        if text:
            lines.append(text)
    return lines, "\n".join(lines)


def discover_pairs(archive: zipfile.ZipFile) -> list[CardPair]:
    """Match PNG and PAGE XML members by filename stem."""
    members: dict[str, dict[str, str]] = {".png": {}, ".xml": {}}
    for name in archive.namelist():
        suffix = Path(name).suffix.lower()
        if suffix in members and not name.startswith("__MACOSX/"):
            members[suffix][Path(name).stem] = name

    images, xml_files = members[".png"], members[".xml"]
    if not images or images.keys() != xml_files.keys():
        raise ValueError(
            f"Kat57 archive has {len(images)} PNGs and {len(xml_files)} PAGE XML files"
        )
    return [CardPair(stem, images[stem], xml_files[stem]) for stem in sorted(images)]


def deterministic_sample(pairs: list[CardPair], size: int, seed: int) -> list[CardPair]:
    """Choose a stable seeded sample and return it in source order."""
    if not 1 <= size <= len(pairs):
        raise ValueError(f"Sample size must be between 1 and {len(pairs)}")
    chosen = sorted(
        pairs,
        key=lambda pair: hashlib.sha256(f"{seed}\0{pair.card_id}".encode()).digest(),
    )[:size]
    return sorted(chosen, key=lambda pair: pair.card_id)


def card_row(archive: zipfile.ZipFile, pair: CardPair) -> dict[str, object]:
    xml_bytes = archive.read(pair.xml_member)
    lines, reference = parse_page_xml(xml_bytes)
    parts = pair.card_id.split("-")
    return {
        "id": pair.card_id,
        "drawer_id": parts[-2],
        "card_id": parts[-1],
        "image": {
            "bytes": archive.read(pair.image_member),
            "path": Path(pair.image_member).name,
        },
        "reference": reference,
        "lines": lines,
        "line_count": len(lines),
        "page_xml": xml_bytes.decode(),
        "source_image": Path(pair.image_member).name,
    }


def md5sum(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - compare with Zenodo's published checksum
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_archive(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {ARCHIVE_URL}")
    urllib.request.urlretrieve(ARCHIVE_URL, destination)  # noqa: S310 - fixed HTTPS URL
    return destination


def publish(
    archive_path: Path,
    repo_id: str,
    *,
    shard_size: int,
    max_samples: int | None,
    sample_seed: int,
    private: bool,
    work_dir: Path,
) -> None:
    """Upload small Parquet shards so the 36 GB source fits on a CPU Job."""
    if md5sum(archive_path) != ARCHIVE_MD5:
        raise ValueError("Archive checksum does not match the Kat57 Zenodo release")

    token = get_token() or os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("Authenticate with `hf auth login` or HF_TOKEN")
    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    if any(path.startswith("data/") for path in api.list_repo_files(repo_id, repo_type="dataset")):
        raise RuntimeError(f"{repo_id} already contains data; choose an empty repository")

    work_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        pairs = discover_pairs(archive)
        if max_samples is not None:
            pairs = deterministic_sample(pairs, max_samples, sample_seed)
        shard_count = (len(pairs) + shard_size - 1) // shard_size
        for shard_index, start in enumerate(range(0, len(pairs), shard_size)):
            rows = [card_row(archive, pair) for pair in pairs[start : start + shard_size]]
            local_path = work_dir / f"train-{shard_index:05d}-of-{shard_count:05d}.parquet"
            Dataset.from_list(rows, features=FEATURES).to_parquet(local_path)
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=f"data/{local_path.name}",
                repo_id=repo_id,
                repo_type="dataset",
            )
            local_path.unlink()
            print(f"Uploaded shard {shard_index + 1}/{shard_count}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo_id")
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--work-dir", type=Path, default=Path("kat57-work"))
    parser.add_argument("--shard-size", type=int, default=128)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--sample-seed", type=int, default=57)
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    if args.shard_size < 1 or (args.max_samples is not None and args.max_samples < 1):
        parser.error("shard size and max samples must be positive")
    archive = args.archive or args.work_dir / "kat57-gt-dataset.zip"
    downloaded = args.archive is None
    if downloaded:
        download_archive(archive)
    elif not archive.is_file():
        parser.error(f"archive not found: {archive}")
    try:
        publish(
            archive,
            args.repo_id,
            shard_size=args.shard_size,
            max_samples=args.max_samples,
            sample_seed=args.sample_seed,
            private=args.private,
            work_dir=args.work_dir,
        )
    finally:
        if downloaded and archive.exists():
            archive.unlink()


if __name__ == "__main__":
    main()
