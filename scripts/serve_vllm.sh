#!/usr/bin/env bash
set -euo pipefail
# Separate Linux vLLM venv; only a local high-precision merged model directory.
model_path="${1:?Usage: bash scripts/serve_vllm.sh /absolute/merged_model [--execute]}"
if [ ! -d "$model_path" ]; then echo 'Local model directory missing' >&2; exit 1; fi
args=(vllm serve "$model_path" --host 127.0.0.1 --port 8001 --dtype auto --max-model-len 1024 --served-model-name alignlab-local)
if [ "${2:---dry-run}" != '--execute' ]; then
  printf 'DRY_RUN: configuration only; server not started\n'
  printf '%q ' "${args[@]}"
  printf '\n'
  exit 0
fi
export HF_HUB_OFFLINE=1
exec "${args[@]}"
