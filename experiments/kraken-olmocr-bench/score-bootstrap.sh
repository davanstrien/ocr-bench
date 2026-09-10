#!/bin/sh
set -eu
mkdir -p /tmp/kraken-score
cp /input/score.py /input/report.py /input/runner.py /input/scoring/pyproject.toml /tmp/kraken-score/
cd /tmp/kraken-score
if [ -f /input/scoring/uv.lock ]; then
    cp /input/scoring/uv.lock uv.lock
elif [ -f /bucket/scoring-environment/uv.lock ]; then
    cp /bucket/scoring-environment/uv.lock uv.lock
else
    uv lock
    mkdir -p /bucket/scoring-environment
    cp uv.lock /bucket/scoring-environment/uv.lock
    cp pyproject.toml /bucket/scoring-environment/pyproject.toml
fi
uv sync --locked
sh -n /input/launch.sh /input/bootstrap.sh /input/score-bootstrap.sh
uv run --locked ruff check score.py runner.py report.py
uv run --locked python -m playwright install --with-deps chromium
uv run --locked python score.py "$@"
uv run --locked python report.py "$@"
