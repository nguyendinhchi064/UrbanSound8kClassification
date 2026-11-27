"""Repeat the selected full fine-tuning recipe from a seed-specific checkpoint."""

import json
import subprocess
import sys
from pathlib import Path

winner, seed = sys.argv[1:]
cfg = json.loads((Path("reports/member4") / winner / "config.json").read_text())
if cfg["stage"] != "full":
    raise SystemExit(0)
arch = cfg["arch"]
name = f"{arch}_repeat_full_{seed}"
out = Path("reports/member4") / name
if out.exists():
    raise SystemExit("Existing run preserved")
cmd = [
    sys.executable,
    "-m",
    "scripts.train_transfer",
    "--name",
    name,
    "--arch",
    arch,
    "--stage",
    "full",
    "--seed",
    seed,
    "--from-checkpoint",
    f"checkpoints/transfer/{arch}_repeat_partial_{seed}/best.pth",
]
for key in (
    "epochs",
    "batch_size",
    "head_lr",
    "backbone_lr",
    "patience",
    "dropout",
    "mixup",
    "smoothing",
    "weight_decay",
):
    if key in cfg:
        cmd.extend(["--" + key.replace("_", "-"), str(cfg[key])])
if cfg["specaugment"]:
    cmd.append("--specaugment")
with open(out.with_suffix(".log"), "w") as log:
    subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
