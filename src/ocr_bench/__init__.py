"""OCR model evaluation toolkit — VLM-as-judge with per-dataset leaderboards."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ocr-bench")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0.0.0"
