"""Repeat the winning audio recipe using independent training seeds."""

import json
from pathlib import Path
import subprocess
import sys
import pandas as pd

ROOT = Path("reports/member4_audio")
df = pd.read_csv(ROOT / "model_comparison.csv")
winner = df[(df.seed == 42) & df.stage.isin(["partial", "full"])].iloc[0]
name = winner["name"]


def chain(run):
    cfg = json.loads((ROOT / run / "config.json").read_text())
    parent = cfg.get("from_checkpoint")
    return (chain(Path(parent).parent.name) if parent else []) + [cfg]


recipe = chain(name)
summary = [dict(name=name, seed=42, accuracy=winner.accuracy, macro_f1=winner.macro_f1)]
for seed in (123, 2024):
    parent = None
    for i, cfg in enumerate(recipe):
        new = f"{cfg['arch']}_confirm_{seed}_{i}_{cfg['stage']}"
        out = ROOT / new
        if not (out / "summary.json").exists():
            if out.exists():
                print("Incomplete run preserved:", new)
                break
            cmd = [
                sys.executable,
                "-m",
                "scripts.train_transfer",
                "--name",
                new,
                "--arch",
                cfg["arch"],
                "--stage",
                cfg["stage"],
                "--seed",
                str(seed),
            ]
            for k in (
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
                if k in cfg:
                    cmd.extend(["--" + k.replace("_", "-"), str(cfg[k])])
            if cfg.get("specaugment"):
                cmd.append("--specaugment")
            if parent:
                cmd.extend(["--from-checkpoint", str(parent)])
            with open(out.with_suffix(".log"), "w") as log:
                result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                print("Failed:", new)
                break
        parent = Path("checkpoints/audio_transfer") / new / "best.pth"
    else:
        result = json.loads((out / "summary.json").read_text())
        summary.append(
            dict(
                name=new,
                seed=seed,
                accuracy=result["accuracy"],
                macro_f1=result["macro_f1"],
            )
        )
stats = pd.DataFrame(summary)
stats.to_csv(ROOT / "seed_comparison.csv", index=False)
stats[["accuracy", "macro_f1"]].agg(["mean", "std"]).to_csv(
    ROOT / "seed_statistics.csv"
)
print(stats.to_string(index=False))
subprocess.run([sys.executable, "-m", "scripts.compare_audio_transfer"], check=True)
