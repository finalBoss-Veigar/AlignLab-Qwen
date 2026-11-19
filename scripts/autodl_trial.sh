#!/usr/bin/env bash
# Explicit AutoDL-only wrapper: setup + five-step SFT/export/DPO, then power off.
# Run with: nohup bash scripts/autodl_trial.sh --execute > trial-launch.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "${1:-}" != --execute ]]; then
  echo 'Not started. Requires --execute on an authorized AutoDL instance.'
  exit 2
fi
[[ "$(uname -s)" == Linux && -d /root/autodl-tmp && -x /usr/bin/shutdown ]] || {
  echo 'AutoDL Linux data mount and /usr/bin/shutdown required'; exit 2;
}
command -v timeout >/dev/null
run_dir="runs/autodl-trial-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$run_dir"
exec > >(tee -a "$run_dir/console.log") 2>&1
finish() {
  code=$?
  trap - EXIT
  printf '{"exit_code":%s,"ended_utc":"%s","shutdown_requested":true}\n' \
    "$code" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$run_dir/status.json"
  sync
  # AutoDL's documented power-off command stops GPU billing, not paid disk fees.
  /usr/bin/shutdown >> "$run_dir/shutdown.log" 2>&1 || true
  exit "$code"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
git rev-parse HEAD > "$run_dir/code.txt"
git status --porcelain >> "$run_dir/code.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "$run_dir/started_utc.txt"
nvidia-smi > "$run_dir/nvidia-smi.txt"
# Deadline includes environment setup/download, with 30 seconds for termination.
timeout --signal=TERM --kill-after=30s 3600 bash -euo pipefail <<'STEPS'
bash scripts/bootstrap_gpu.sh
source .venv-train/bin/activate
export CUDA_VISIBLE_DEVICES=0
export HF_HOME="$PWD/models/hf-cache"
python scripts/fetch_model.py --confirm-download
python -m alignlab train configs/gpu_20261118/preflight_sft.yaml --execute
python -m alignlab export configs/gpu_20261118/preflight_export_sft.yaml --execute
python -m alignlab train configs/gpu_20261118/preflight_dpo.yaml --execute
STEPS
