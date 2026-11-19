#!/usr/bin/env bash
# Linux only. Downloads packages/upstream code, never launches a paid instance.
set -euo pipefail
cd "$(dirname "$0")/.."
test "$(uname -s)" = Linux || { echo 'Use a Linux GPU instance'; exit 2; }
mkdir -p vendor runs
if [ ! -d vendor/LlamaFactory ]; then
  git clone --depth 1 --branch v0.9.3 https://github.com/hiyouga/LlamaFactory.git vendor/LlamaFactory
fi
test "$(git -C vendor/LlamaFactory rev-parse HEAD)" = ca75f1edf3cb50343ed1c98605141c3e22075b5f
if [ ! -d .venv-train ]; then python3 -m venv .venv-train; fi
source .venv-train/bin/activate
python -m pip install -c requirements/train-constraints.txt torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -c requirements/train-constraints.txt -e './vendor/LlamaFactory[bitsandbytes]'
python -m pip install -e .
python -m pip check
python scripts/check_training_imports.py --output runs/training-imports.json
python -m pip freeze > runs/train-environment-resolved.txt
python -m alignlab doctor --output runs/gpu-doctor.json
python -c "from alignlab.training.configs import verify_sources; verify_sources('.')"
echo 'Environment ready. Check gpu-doctor.json before downloading weights or training.'
