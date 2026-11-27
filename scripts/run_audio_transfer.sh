#!/bin/bash
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
mkdir -p reports/member4_audio
run() {
  local name="$1"; shift
  if [[ -f "reports/member4_audio/$name/summary.json" ]]; then return 0; fi
  if [[ -d "reports/member4_audio/$name" ]]; then echo "Incomplete run preserved: $name"; return 1; fi
  if "$PY" -m scripts.train_transfer --name "$name" "$@" > "reports/member4_audio/$name.log" 2>&1; then
    "$PY" -m scripts.compare_audio_transfer
  else
    echo "FAILED: $name; see its log"; return 1
  fi
}
for arch in mn10_as mn20_as; do
  run "${arch}_frozen" --arch "$arch" --stage frozen --epochs 12 --dropout .25 || continue
  run "${arch}_partial" --arch "$arch" --stage partial --epochs 20 --head-lr .0003 --backbone-lr .00004 --dropout .25 --specaugment --from-checkpoint "checkpoints/audio_transfer/${arch}_frozen/best.pth" || continue
  run "${arch}_full" --arch "$arch" --stage full --epochs 25 --head-lr .0002 --backbone-lr .00003 --dropout .25 --specaugment --smoothing .05 --from-checkpoint "checkpoints/audio_transfer/${arch}_partial/best.pth" || true
done
# Higher capacity variant only if its frozen run is successful.
arch=mn40_as
run "${arch}_frozen" --arch "$arch" --stage frozen --epochs 10 --batch-size 16 --dropout .25 &&
run "${arch}_partial" --arch "$arch" --stage partial --epochs 15 --batch-size 16 --head-lr .0002 --backbone-lr .00003 --dropout .25 --specaugment --from-checkpoint "checkpoints/audio_transfer/${arch}_frozen/best.pth" || true
winner=$("$PY" -c 'import pandas as pd; d=pd.read_csv("reports/member4_audio/model_comparison.csv"); print(d[(d.stage=="partial") | (d.stage=="full")].iloc[0].arch)') || exit 1
for variant in mixup higher_lr; do
  extra=()
  case "$variant" in
    mixup) extra=(--mixup .15 --backbone-lr .00003);;
    higher_lr) extra=(--smoothing .05 --backbone-lr .00008);;
  esac
  run "${winner}_opt_${variant}" --arch "$winner" --stage full --epochs 25 --batch-size 16 --head-lr .0003 --specaugment --dropout .25 --from-checkpoint "checkpoints/audio_transfer/${winner}_partial/best.pth" "${extra[@]}" || true
done
"$PY" -m scripts.analyze_audio_transfer
