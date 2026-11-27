"""Summarize audio-pretrained runs separately; validation-only."""

import json
from pathlib import Path
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("reports/member4_audio")
FIG = Path("figures/audio_transfer")
FIG.mkdir(parents=True, exist_ok=True)
rows = []
for f in sorted(ROOT.glob("*/summary.json")):
    row = json.loads(f.read_text())
    rows.append(row)
    hist = pd.read_csv(f.parent / "training_history.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric in zip(axes, ("accuracy", "loss")):
        ax.plot(hist.epoch, hist["train_" + metric], label="Train")
        ax.plot(hist.epoch, hist["val_" + metric], label="Validation")
        ax.set(xlabel="Epoch", ylabel=metric.title())
        ax.legend()
    fig.suptitle(row["name"])
    fig.tight_layout()
    fig.savefig(FIG / (row["name"] + ".png"), dpi=150)
    plt.close(fig)
if not rows:
    raise SystemExit("No completed audio runs")
df = pd.DataFrame(rows).sort_values(["accuracy", "macro_f1"], ascending=False)
df.to_csv(ROOT / "model_comparison.csv", index=False)
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
fig, ax = plt.subplots(figsize=(11, max(4, len(df) * 0.36)))
ax.barh(df.name[::-1], df.accuracy[::-1])
ax.axvline(86.88725490196079, ls="--", color="black", label="Custom CNN")
ax.axvline(87.5, ls=":", color="gray", label="Best ImageNet run")
ax.set(xlabel="Validation accuracy (%)", xlim=(0, 100))
ax.legend()
fig.tight_layout()
fig.savefig(FIG / "comparison.png", dpi=180)
plt.close(fig)
lines = [
    "# Audio-pretrained transfer learning",
    "",
    "Exploratory extension using additional AudioSet pretraining; not directly equivalent to ImageNet-only transfer.",
    "Fixed team holdout; no test evaluation. External pretraining overlap has not been ruled out.",
    "",
    "| " + " | ".join(cols) + " |",
    "| " + " | ".join(["---"] * len(cols)) + " |",
]
for row in df[cols].itertuples(index=False, name=None):
    lines.append(
        "| "
        + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in row)
        + " |"
    )
(ROOT / "report.md").write_text("\n".join(lines) + "\n")
print(df[cols].to_string(index=False))
