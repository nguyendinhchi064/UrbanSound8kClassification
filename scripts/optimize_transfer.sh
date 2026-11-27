#!/bin/bash
# Bounded search, no test inference, and no Git network actions.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
mkdir -p reports/member4
run() {
  local name="$1"; shift
  if [[ -f "reports/member4/$name/summary.json" ]]; then return 0; fi
  if [[ -d "reports/member4/$name" ]]; then
    echo "Incomplete run preserved: $name" >&2
    return 1
  fi
  if "$PY" -m scripts.train_transfer --name "$name" "$@" > "reports/member4/$name.log" 2>&1; then
    "$PY" -m scripts.compare_transfer
  else
    echo "FAILED: $name (see log)" >&2
    return 1
  fi
}
# Rank only comparable, completed partial fine-tunes from the initial sweep.
leaders=$("$PY" -c 'import pandas as pd; d=pd.read_csv("reports/member4/model_comparison.csv"); print(" ".join(d[d.stage=="partial"].drop_duplicates("arch").arch.head(2)))') || exit 1
for arch in $leaders; do
  for variant in higher_lr regularized mixup; do
    extra=()
    case "$variant" in
      higher_lr) extra=(--backbone-lr .0002 --head-lr .0005);;
      regularized) extra=(--backbone-lr .0001 --head-lr .0003 --specaugment --smoothing .05);;
      mixup) extra=(--backbone-lr .0001 --head-lr .0003 --mixup .2 --specaugment);;
    esac
    run "${arch}_opt_${variant}" --arch "$arch" --stage full --epochs 30 --patience 8 --batch-size 32 --from-checkpoint "checkpoints/transfer/${arch}_partial/best.pth" "${extra[@]}" || true
  done
done
# Select a single recipe on validation, then repeat from original ImageNet weights.
# These repeated seeds are sensitivity analysis, not a new held-out evaluation.
winner=$("$PY" -c 'import pandas as pd; d=pd.read_csv("reports/member4/model_comparison.csv"); print(d[(d.stage=="full") | (d.stage=="partial")].iloc[0]["name"])') || exit 1
for seed in 123 2024; do
  arch=$("$PY" -c 'import json,sys; print(json.load(open("reports/member4/"+sys.argv[1]+"/config.json"))["arch"])' "$winner")
  run "${arch}_repeat_frozen_${seed}" --arch "$arch" --stage frozen --epochs 12 --seed "$seed" || continue
  run "${arch}_repeat_partial_${seed}" --arch "$arch" --stage partial --epochs 25 --head-lr .0003 --seed "$seed" --from-checkpoint "checkpoints/transfer/${arch}_repeat_frozen_${seed}/best.pth" || continue
  # Apply the selected full fine-tuning recipe if the winning run used it.
  "$PY" -m scripts.repeat_transfer "$winner" "$seed" || true
  "$PY" -m scripts.compare_transfer
done
"$PY" -m scripts.analyze_transfer
