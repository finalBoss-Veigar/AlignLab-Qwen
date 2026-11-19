#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [ ! -x .venv/bin/python ]; then
  if command -v python3.11 >/dev/null; then py=python3.11
  elif command -v python3.12 >/dev/null; then py=python3.12
  else echo 'Install Python 3.11 or 3.12 first.' >&2; exit 1; fi
  "$py" -m venv .venv
fi
.venv/bin/python -m pip install -r requirements/dev.lock.txt
.venv/bin/python -m pip install --no-build-isolation --no-deps -e .
.venv/bin/python -m alignlab --help
.venv/bin/python scripts/bootstrap_sources.py
