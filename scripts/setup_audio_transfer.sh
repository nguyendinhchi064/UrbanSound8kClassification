#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/upstream
if [[ ! -d data/upstream/EfficientAT/.git ]]; then
  git clone https://github.com/fschmid56/EfficientAT.git data/upstream/EfficientAT
  git -C data/upstream/EfficientAT checkout a425fdce92572e602a1d5634799bd9f1f2efa806
fi
test "$(git -C data/upstream/EfficientAT rev-parse HEAD)" = a425fdce92572e602a1d5634799bd9f1f2efa806
.venv/bin/python -m src.audio_transfer_data
