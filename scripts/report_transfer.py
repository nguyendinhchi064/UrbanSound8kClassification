"""Render concise submission figures from recorded validation metrics."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("reports/member4")
FIG = Path("figures/transfer")
FIG.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(ROOT / "model_comparison.csv")
bench = pd.read_csv(ROOT / "final_benchmark.csv")
baseline = json.loads((ROOT / "baseline_reference.json").read_text())
archs = ["resnet18", "mobilenet_v2", "efficientnet_b0", "densenet121"]
labels = ["ResNet18", "MobileNetV2", "EfficientNet-B0", "DenseNet121"]
x = np.arange(4)
fig, ax = plt.subplots(figsize=(9, 4.8))
for i, (stage, color) in enumerate([("frozen", "#a8b7c8"), ("partial", "#146d86")]):
    values = [
        float(df.loc[df.name == f"{a}_{stage}", "accuracy"].iloc[0]) for a in archs
    ]
    bars = ax.bar(x + (i - 0.5) * 0.33, values, 0.33, label=stage.title(), color=color)
    ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
ax.axhline(baseline["accuracy"], ls="--", color="#ba693d", label="Custom CNN: 86.89%")
ax.set_xticks(x, labels)
ax.set(
    ylabel="Validation accuracy (%)",
    ylim=(0, 100),
    title="ImageNet transfer: frozen vs. selective fine-tuning",
)
ax.legend(loc="lower right", frameon=False)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(FIG / "submission_stage_comparison.png", dpi=180)
plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 5.2))
offsets = {
    "ResNet18": (8, 7),
    "MobileNetV2": (8, 7),
    "EfficientNet-B0": (8, -16),
    "DenseNet121": (-78, 9),
}
for a, label in zip(archs, labels):
    row = bench[
        (bench.architecture == a) & (bench.run != "resnet18_opt_regularized")
    ].iloc[0]
    acc = float(df.loc[df.name == row.run, "accuracy"].iloc[0])
    ax.scatter(row.estimated_macs / 1e6, acc, s=85, color="#146d86")
    ax.annotate(
        label,
        (row.estimated_macs / 1e6, acc),
        xytext=offsets[label],
        textcoords="offset points",
    )
ax.scatter(
    baseline["estimated_macs"] / 1e6,
    baseline["accuracy"],
    s=95,
    marker="D",
    color="#ba693d",
)
ax.annotate(
    "Custom CNN",
    (baseline["estimated_macs"] / 1e6, baseline["accuracy"]),
    xytext=(8, 8),
    textcoords="offset points",
)
ax.set(
    xlabel="Estimated MACs per 128 × 173 input (millions)",
    ylabel="Validation accuracy (%)",
    title="Accuracy–compute trade-off (ImageNet track)",
    ylim=(80, 90),
    xlim=(0, 1450),
)
ax.grid(alpha=0.2)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(FIG / "submission_efficiency.png", dpi=180)
plt.close(fig)
print("Submission figures generated")
