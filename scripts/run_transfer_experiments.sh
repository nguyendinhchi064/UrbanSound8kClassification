#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
mkdir -p reports/member4
run() {
  local name="$1"; shift
  if [[ -f "reports/member4/$name/summary.json" ]]; then return; fi
  "$PY" -m scripts.train_transfer --name "$name" "$@" > "reports/member4/$name.log" 2>&1
  "$PY" -m scripts.compare_transfer
}
for arch in efficientnet_b0 resnet18 mobilenet_v2 densenet121; do
  run "${arch}_frozen" --arch "$arch" --stage frozen --epochs 12
  run "${arch}_partial" --arch "$arch" --stage partial --epochs 25 --head-lr .0003 --from-checkpoint "checkpoints/transfer/${arch}_frozen/best.pth"
done
winner=$("$PY" -c 'import pandas as pd; d=pd.read_csv("reports/member4/model_comparison.csv"); print(d[d.stage=="partial"].iloc[0].arch)')
run "${winner}_full" --arch "$winner" --stage full --epochs 30 --head-lr .0003 --backbone-lr .00005 --specaugment --from-checkpoint "checkpoints/transfer/${winner}_partial/best.pth"
"$PY" -m scripts.compare_transfer
