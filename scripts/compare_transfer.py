"""Aggregate completed experiments; no final test evaluation."""

import json
from pathlib import Path
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path("reports/member4")
figdir = Path("figures/transfer")
figdir.mkdir(parents=True, exist_ok=True)
rows = []
for path in sorted(root.glob("*/summary.json")):
    row = json.loads(path.read_text())
    rows.append(row)
    hist = pd.read_csv(path.parent / "training_history.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric in zip(axes, ["accuracy", "loss"]):
        for split in ["train", "val"]:
            ax.plot(hist.epoch, hist[f"{split}_{metric}"], label=split)
        ax.set(xlabel="Epoch", ylabel=metric)
        ax.legend()
    fig.suptitle(row["name"])
    fig.tight_layout()
    fig.savefig(figdir / f"{row['name']}.png", dpi=150)
    plt.close(fig)
if not rows:
    raise SystemExit("No completed experiments")
df = pd.DataFrame(rows).sort_values(["accuracy", "macro_f1"], ascending=False)
df.to_csv(root / "model_comparison.csv", index=False)
cols = [
    "name",
    "arch",
    "stage",
    "seed",
    "accuracy",
    "macro_f1",
    "balanced_accuracy",
    "training_seconds",
    "total_parameters",
    "trainable_parameters",
]
table = df[cols]
fig, ax = plt.subplots(figsize=(10, max(4, len(df) * 0.35)))
ax.barh(df.name[::-1], df.accuracy[::-1])
ax.axvline(86.88725490196079, ls="--", color="black", label="Custom CNN validation")
ax.set(xlabel="Validation accuracy (%)", xlim=(0, 100))
ax.legend()
fig.tight_layout()
fig.savefig(figdir / "comparison.png", dpi=180)
plt.close(fig)
lines = [
    "# Member 4 — transfer learning",
    "",
    "Fixed split: 7,000 training, 816 validation; test fold reserved.",
    "All metrics below are validation metrics, not final test scores.",
    "",
    "| " + " | ".join(cols) + " |",
    "| " + " | ".join(["---"] * len(cols)) + " |",
]
for row in table.itertuples(index=False, name=None):
    lines.append(
        "| "
        + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in row)
        + " |"
    )
lines.extend(
    [
        "",
        f"Best observed run: {df.iloc[0]['name']}.",
        "Custom CNN reference: 86.89% validation accuracy (Member 3).",
        "Single-seed runs are preliminary. Fine-tuning times exclude frozen-stage training; compare cumulative time for deployment decisions.",
    ]
)
(root / "member4_report.md").write_text("\n".join(lines) + "\n")
print(table.to_string(index=False))
