#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
uv run ruff format --check src tests
uv run ruff check src tests
uv run mypy src tests
uv run pytest
