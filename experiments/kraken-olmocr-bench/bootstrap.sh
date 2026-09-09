#!/bin/sh
set -eu
mkdir -p /tmp/kraken-bench
cp /input/*.py /input/pyproject.toml /tmp/kraken-bench/
cd /tmp/kraken-bench
if [ -f /input/uv.lock ]; then
    cp /input/uv.lock uv.lock
elif [ -f /bucket/environment/uv.lock ]; then
    cp /bucket/environment/uv.lock uv.lock
else
    uv lock
    mkdir -p /bucket/environment
    cp uv.lock /bucket/environment/uv.lock
    cp pyproject.toml /bucket/environment/pyproject.toml
fi
uv sync --locked
uv run --locked ruff check runner.py
uv run --locked python runner.py "$@"
