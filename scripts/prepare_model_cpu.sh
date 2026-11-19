#!/usr/bin/env bash
# Separate lightweight environment: no CUDA or model loading needed to fetch weights.
set -euo pipefail
cd "$(dirname "$0")/.."
[[ "${1:-}" == --confirm-download ]] || { echo 'Requires --confirm-download'; exit 2; }
python_binary="${PYTHON_BIN:-python3}"
if ! command -v "$python_binary" >/dev/null 2>&1 && [[ -x .venv-train/bin/python ]]; then
  python_binary=.venv-train/bin/python
fi
if [[ ! -d .venv-download ]]; then "$python_binary" -m venv .venv-download; fi
.venv-download/bin/python -m pip install huggingface-hub==0.30.2
export HF_HOME="$PWD/models/hf-cache"
.venv-download/bin/python scripts/fetch_model.py --confirm-download --max-workers 2
