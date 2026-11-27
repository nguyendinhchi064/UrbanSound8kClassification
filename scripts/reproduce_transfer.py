"""Replay the recorded CNN configurations without altering published evidence.

Example: python -m scripts.reproduce_transfer resnet18_repeat_full_2024
The full parent chain is reconstructed from the saved configuration JSON.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOTS = (Path("reports/member4"), Path("reports/member4_audio"))


def find_config(name):
    if Path(name).name != name or name in (".", ".."):
        raise ValueError("Invalid run name")
    paths = [r / name / "config.json" for r in ROOTS]
    found = [p for p in paths if p.is_file()]
    if len(found) != 1:
        raise ValueError(f"Unknown or ambiguous run: {name}")
    return found[0], json.loads(found[0].read_text())


def lineage(name, seen=None):
    seen = set() if seen is None else seen
    if name in seen:
        raise ValueError("Cycle in recorded lineage")
    seen.add(name)
    _, cfg = find_config(name)
    parent = cfg.get("from_checkpoint")
    return (lineage(Path(parent).parent.name, seen) if parent else []) + [cfg]


def command(cfg, root, parent=None):
    cmd = [
        sys.executable,
        "-m",
        "scripts.train_transfer",
        "--name",
        cfg["name"],
        "--arch",
        cfg["arch"],
        "--stage",
        cfg["stage"],
        "--report-root",
        str(root / "reports"),
        "--checkpoint-root",
        str(root / "checkpoints"),
    ]
    for k in (
        "epochs",
        "batch_size",
        "head_lr",
        "backbone_lr",
        "seed",
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
    return cmd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", help="Recorded run names")
    parser.add_argument("--root", type=Path, default=Path("runs/member4/replay"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    # Keep historical output paths immutable.
    reserved = [Path("reports").resolve(), Path("checkpoints").resolve()]
    if any(root == p or p in root.parents for p in reserved):
        parser.error("Use an independent output root outside reports/ and checkpoints/")
    names = list(args.runs)
    if names == ["all-image"]:
        names = [p.parent.name for p in sorted(ROOTS[0].glob("*/config.json"))]
    elif names == ["all-audio"]:
        names = [p.parent.name for p in sorted(ROOTS[1].glob("*/config.json"))]
    done = set()
    for name in names:
        parent = None
        for cfg in lineage(name):
            run = cfg["name"]
            dest = root / "checkpoints" / run / "best.pth"
            report = root / "reports" / run
            if run in done:
                parent = dest
                continue
            if dest.is_file() and (report / "summary.json").is_file():
                print("Verified output paths exist; skipping", run)
            elif report.exists() and not args.dry_run:
                raise RuntimeError(
                    f"Incomplete replay at {report}; choose a new --root."
                )
            else:
                cmd = command(cfg, root, parent)
                print(" ".join(cmd), flush=True)
                if not args.dry_run:
                    subprocess.run(cmd, check=True)
            parent = dest
            done.add(run)


if __name__ == "__main__":
    main()
