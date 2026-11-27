"""Fixed validation blends and final exploratory summary."""

from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from src.data_loader import CLASS_NAMES

plt.switch_backend("Agg")

A = Path("reports/member4_audio/advanced")
B = Path("reports/member4_ast")
F = Path("figures/audio_transfer")
F.mkdir(parents=True, exist_ok=True)
idx = pd.read_csv("data/audio_transfer/validation_index.csv")
astidx = pd.read_csv("data/ast_transfer/validation.csv")
assert idx.slice_file_name.tolist() == astidx.slice_file_name.tolist()
assert idx.classID.tolist() == astidx.classID.tolist()
y = idx.classID.to_numpy()
names = [
    "mn20_as_partial",
    "mn20_as_confirm_123_1_partial",
    "mn20_as_confirm_2024_1_partial",
    "mn20_as_opt_higher_lr",
    "mn20_as_confirm_123_2_full",
    "mn20_as_confirm_2024_2_full",
    "mn10_as_partial",
    "mn40_as_frozen",
]
cnn = np.mean([np.load(A / f"{n}_probabilities.npy").mean(0) for n in names], axis=0)
candidates = {"cnn_diverse_tta": cnn}
for n in ("frozen", "tail"):
    p = B / f"{n}_probabilities.npy"
    if p.exists():
        a = np.load(p)
        assert a.shape == (len(y), 10) and np.isfinite(a).all()
        candidates["ast_" + n] = a
        candidates["cnn_ast_" + n + "_equal"] = (cnn + a) / 2
if "ast_frozen" in candidates and "ast_tail" in candidates:
    candidates["cnn_ast_both"] = (
        0.5 * cnn + 0.25 * candidates["ast_frozen"] + 0.25 * candidates["ast_tail"]
    )
rows = []
for n, p in candidates.items():
    pred = p.argmax(1)
    rows.append(
        dict(
            name=n,
            accuracy=100 * accuracy_score(y, pred),
            macro_f1=100 * f1_score(y, pred, average="macro"),
            balanced_accuracy=100 * balanced_accuracy_score(y, pred),
        )
    )
df = pd.DataFrame(rows).sort_values(["accuracy", "macro_f1"], ascending=False)
df.to_csv(B / "final_ensemble_comparison.csv", index=False)
name = df.iloc[0]["name"]
pred = candidates[name].argmax(1)
pd.DataFrame(
    classification_report(
        y,
        pred,
        labels=list(range(10)),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
).T.to_csv(B / "selected_class_metrics.csv")
cm = confusion_matrix(y, pred, labels=list(range(10)))
pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
    B / "selected_confusion.csv"
)

fig, ax = plt.subplots(figsize=(10, 5))
ax.barh(df.name[::-1], df.accuracy[::-1])
ax.axvline(86.88725490196079, ls="--", color="black", label="Custom CNN")
ax.set(xlabel="Validation accuracy (%)", xlim=(0, 100))
ax.legend()
fig.tight_layout()
fig.savefig(F / "ast_ensemble_comparison.png", dpi=180)
plt.close(fig)
fig, ax = plt.subplots(figsize=(9, 7))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(10), CLASS_NAMES, rotation=45, ha="right")
ax.set_yticks(range(10), CLASS_NAMES)
ax.set(xlabel="Predicted", ylabel="Actual", title=f"Validation — {name}")
for i in range(10):
    for j in range(10):
        ax.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center",
            color="white" if cm[i, j] > cm.max() / 2 else "black",
        )
fig.colorbar(im, ax=ax)
fig.tight_layout()
fig.savefig(F / "ast_selected_confusion.png", dpi=180)
plt.close(fig)
lines = [
    "# Optional AST and ensemble extension",
    "",
    "Validation only. The final test fold was not evaluated.",
    "AST is an AudioSet-pretrained transformer, outside the core CNN assignment. The input is shortened to 400 frames using the corresponding leading temporal position embeddings. Prefix activations are cached in float16. Frozen SVC decision and probability rules can differ; this table uses probability argmax for every blend.",
    "",
    "| Model | Accuracy (%) | Macro F1 (%) | Balanced accuracy (%) |",
    "| --- | ---: | ---: | ---: |",
]
for r in df.itertuples():
    lines.append(
        f"| {r.name} | {r.accuracy:.2f} | {r.macro_f1:.2f} | {r.balanced_accuracy:.2f} |"
    )
for file in ("frozen_summary.json", "tail_summary.json"):
    path = B / file
    if path.exists():
        data = json.loads(path.read_text())
        lines.append(
            f"\n{file}: validation accuracy {data['accuracy']:.2f}%, macro F1 {data['macro_f1']:.2f}%."
        )
lines.extend(
    [
        "",
        "The ensemble comparison is exploratory and validation-selected. No external pretraining overlap guarantee is made. Additional models increase memory and inference cost; a single CNN remains the practical core deliverable.",
    ]
)
(B / "report.md").write_text("\n".join(lines) + "\n")
print(df.to_string(index=False), flush=True)
