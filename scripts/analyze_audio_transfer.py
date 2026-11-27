"""Validation-only evaluation of audio CNNs and mixed-pretraining ensembles."""

import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from src.audio_transfer_data import AudioDataset
from src.audio_transfer_models import AudioTransferCNN
from src.datasets import SpectrogramDataset
from src.transfer_models import TransferCNN
from src.features import load_feature_cache
from src.data_loader import split_dataframe, CLASS_NAMES
from src.training import get_device

plt.switch_backend("Agg")

ROOT = Path("reports/member4_audio")
FIG = Path("figures/audio_transfer")
FIG.mkdir(parents=True, exist_ok=True)
device = get_device()
audio = AudioDataset("validation")
image = SpectrogramDataset("validation")
# Confirm the exact ordered identity of each sample before blending.

_, idx = load_feature_cache()
idx = split_dataframe(idx, "validation").reset_index(drop=True)
assert idx.slice_file_name.tolist() == audio.index.slice_file_name.tolist()
assert idx.classID.tolist() == audio.index.classID.tolist()
df = pd.read_csv(ROOT / "model_comparison.csv")
leaders = df[df.seed == 42].drop_duplicates("arch").head(3)
probs = []
benchmarks = []


def predict(row, is_audio):
    folder = "audio_transfer" if is_audio else "transfer"
    path = Path("checkpoints") / folder / row["name"] / "best.pth"
    ck = torch.load(path, map_location="cpu", weights_only=True)
    m = (AudioTransferCNN if is_audio else TransferCNN)(
        row["arch"], pretrained=False, dropout=ck["config"]["dropout"]
    )
    m.load_state_dict(ck["model_state_dict"])
    m.configure(ck["config"]["stage"])
    m.to(device).eval()
    pred = []
    ys = []
    data = audio if is_audio else image
    with torch.inference_mode():
        for x, y in DataLoader(data, batch_size=32, shuffle=False):
            pred.append(m(x.to(device)).softmax(1).cpu().numpy())
            ys.extend(y.tolist())
        x = data[0][0].unsqueeze(0).to(device)
        for _ in range(5):
            m(x)
        if device.type == "mps":
            torch.mps.synchronize()
        t = time.perf_counter()
        for _ in range(30):
            m(x)
            if device.type == "mps":
                torch.mps.synchronize()
        latency = (time.perf_counter() - t) * 1000 / 30
    b = dict(
        name=row["name"],
        arch=row["arch"],
        parameters=sum(p.numel() for p in m.parameters()),
        checkpoint_mib=path.stat().st_size / 2**20,
        device=str(device),
        batch1_latency_ms=latency,
    )
    try:
        from thop import profile

        macs, _ = profile(m.cpu(), inputs=(x.cpu(),), verbose=False)
        b["estimated_macs"] = int(macs)
    except Exception as exc:
        b["mac_error"] = str(exc)
    benchmarks.append(b)
    assert ys == audio.y.tolist()
    return np.concatenate(pred)


for _, row in leaders.iterrows():
    probs.append(predict(row, True))
if not probs:
    raise SystemExit("No completed audio models")
labels = audio.y
results = []


def evaluate(name, p, members):
    y = p.argmax(1)
    results.append(
        dict(
            name=name,
            accuracy=100 * accuracy_score(labels, y),
            macro_f1=100 * f1_score(labels, y, average="macro"),
            members=";".join(members),
        )
    )
    pd.DataFrame(
        classification_report(
            labels,
            y,
            labels=list(range(10)),
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        )
    ).T.to_csv(ROOT / f"{name}_class_metrics.csv")
    cm = confusion_matrix(labels, y, labels=list(range(10)))
    pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
        ROOT / f"{name}_confusion.csv"
    )
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
    fig.savefig(FIG / f"{name}_confusion.png", dpi=160)
    plt.close(fig)


names = leaders.name.tolist()
evaluate("best_audio_seed42", probs[0], names[:1])
if len(probs) > 1:
    evaluate("audio_ensemble", np.mean(probs, axis=0), names)
base = pd.read_csv("reports/member4/model_comparison.csv")
base = base[base.seed == 42].iloc[0]
p = predict(base, False)
evaluate(
    "mixed_pretraining_ensemble",
    0.75 * np.mean(probs, axis=0) + 0.25 * p,
    names + [base["name"]],
)
pd.DataFrame(results).to_csv(ROOT / "validation_analysis.csv", index=False)
pd.DataFrame(benchmarks).to_csv(ROOT / "benchmark.csv", index=False)
print(pd.DataFrame(results).to_string(index=False))
