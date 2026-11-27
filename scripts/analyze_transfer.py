"""Validation-only error analysis, complexity benchmark and exploratory ensemble."""

import json
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)
from torch.utils.data import DataLoader
from src.datasets import SpectrogramDataset
from src.data_loader import CLASS_NAMES
from src.transfer_models import TransferCNN
from src.training import get_device

plt.switch_backend("Agg")

ROOT = Path("reports/member4")
FIG = Path("figures/transfer")
device = get_device()
dataset = SpectrogramDataset("validation")
assert len(dataset) == 816
loader = DataLoader(dataset, batch_size=32, shuffle=False)
df = pd.read_csv(ROOT / "model_comparison.csv")
ranked = df[df.seed == 42].sort_values(["accuracy", "macro_f1"], ascending=False)
leaders = ranked.drop_duplicates("arch").head(3)
probs = []
benchmarks = []


def sync():
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


for row in leaders.itertuples():
    path = Path("checkpoints/transfer") / row.name / "best.pth"
    ck = torch.load(path, map_location="cpu", weights_only=True)
    model = TransferCNN(row.arch, pretrained=False, dropout=ck["config"]["dropout"])
    model.load_state_dict(ck["model_state_dict"])
    model.configure(ck["config"]["stage"])
    model.to(device).eval()
    pred = []
    labels = []
    with torch.inference_mode():
        for x, y in loader:
            pred.append(model(x.to(device)).softmax(1).cpu().numpy())
            labels.extend(y.tolist())
        x = torch.zeros(1, 1, 128, 173, device=device)
        for _ in range(10):
            model(x)
        sync()
        t = time.perf_counter()
        for _ in range(50):
            model(x)
            sync()
        latency = (time.perf_counter() - t) * 1000 / 50
    probs.append(np.concatenate(pred))
    benchmarks.append(
        dict(
            run=row.name,
            arch=row.arch,
            device=str(device),
            batch1_latency_ms=latency,
            total_parameters=row.total_parameters,
            checkpoint_mib=path.stat().st_size / 2**20,
        )
    )
    try:
        from thop import profile

        macs, _ = profile(
            model.cpu(), inputs=(torch.zeros(1, 1, 128, 173),), verbose=False
        )
        benchmarks[-1]["estimated_macs"] = int(macs)
    except Exception as exc:
        benchmarks[-1]["mac_error"] = str(exc)
pd.DataFrame(benchmarks).to_csv(ROOT / "benchmark.csv", index=False)

results = []


def analyze(name, probability):
    pred = probability.argmax(1)
    results.append(
        dict(
            name=name,
            accuracy=100 * accuracy_score(labels, pred),
            macro_f1=100 * f1_score(labels, pred, average="macro"),
        )
    )
    report = classification_report(
        labels,
        pred,
        labels=list(range(10)),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).T.to_csv(ROOT / f"{name}_class_metrics.csv")
    cm = confusion_matrix(labels, pred, labels=list(range(10)))
    pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
        ROOT / f"{name}_confusion.csv"
    )
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(10), CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticks(range(10), CLASS_NAMES)
    ax.set(
        xlabel="Predicted",
        ylabel="Actual",
        title=f"Validation confusion matrix — {name}",
    )
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
    fig.savefig(FIG / f"{name}_confusion.png", dpi=160)
    plt.close(fig)


analyze("best_single", probs[0])
if len(probs) > 1:
    analyze("ensemble_top_architectures", np.mean(probs, axis=0))
pd.DataFrame(results).to_csv(ROOT / "validation_analysis.csv", index=False)
(ROOT / "ensemble_members.json").write_text(json.dumps(leaders.name.tolist(), indent=2))
# Repeat recipe comparison uses seed 42 winner and matching stage from independent runs.
winner = ranked[ranked.stage.isin(["partial", "full"])].iloc[0]
stage = winner.stage
arch = winner.arch
repeats = df[(df.arch == arch) & (df.stage == stage) & df.name.str.contains("_repeat_")]
if len(repeats):
    selected = pd.concat([winner.to_frame().T, repeats]).drop_duplicates("seed")
    selected[["name", "seed", "accuracy", "macro_f1"]].to_csv(
        ROOT / "seed_comparison.csv", index=False
    )
    metrics = selected[["accuracy", "macro_f1"]].astype(float)
    metrics.agg(["mean", "std"]).to_csv(ROOT / "seed_statistics.csv")
print(pd.DataFrame(results).to_string(index=False))
