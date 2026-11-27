"""Fixed validation-only multi-seed ensemble and conservative time-shift TTA."""

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from src.audio_transfer_data import AudioDataset
from src.audio_transfer_models import AudioTransferCNN
from src.training import get_device

ROOT = Path("reports/member4_audio")
OUT = ROOT / "advanced"
OUT.mkdir(exist_ok=True)
data = AudioDataset("validation")
assert len(data) == 816
loader = DataLoader(data, batch_size=16, shuffle=False)
device = get_device()
# Fixed recipe: three independently trained partial MN20 checkpoints.
names = [
    "mn20_as_partial",
    "mn20_as_confirm_123_1_partial",
    "mn20_as_confirm_2024_1_partial",
]
# Add the independently trained full-stage recipe and smaller model for diversity.
names += [
    "mn20_as_opt_higher_lr",
    "mn20_as_confirm_123_2_full",
    "mn20_as_confirm_2024_2_full",
    "mn10_as_partial",
    "mn40_as_frozen",
]
labels = data.y


def shift(x, frames):
    if not frames:
        return x
    # Shift with silence padding, never wrap the end into the beginning.
    pad = (np.log(1e-5) + 4.5) / 5
    y = torch.full_like(x, pad)
    if frames > 0:
        y[..., frames:] = x[..., :-frames]
    else:
        y[..., :frames] = x[..., -frames:]
    return y


preds = {}
for name in names:
    ck = torch.load(
        Path("checkpoints/audio_transfer") / name / "best.pth",
        map_location="cpu",
        weights_only=True,
    )
    cfg = ck["config"]
    model = AudioTransferCNN(cfg["arch"], pretrained=False, dropout=cfg["dropout"])
    model.load_state_dict(ck["model_state_dict"])
    model.configure(cfg["stage"])
    model.to(device).eval()
    outputs = [[], [], []]
    with torch.inference_mode():
        for x, y in loader:
            x = x.to(device)
            for i, offset in enumerate((0, -8, 8)):
                outputs[i].append(model(shift(x, offset)).softmax(1).cpu().numpy())
    p = np.stack([np.concatenate(a) for a in outputs])
    preds[name] = p
    np.save(OUT / f"{name}_probabilities.npy", p)
    print(
        name,
        100 * accuracy_score(labels, p[0].argmax(1)),
        100 * accuracy_score(labels, p.mean(0).argmax(1)),
        flush=True,
    )
rows = []


def score(name, p, members):
    y = p.argmax(1)
    rows.append(
        dict(
            name=name,
            accuracy=100 * accuracy_score(labels, y),
            macro_f1=100 * f1_score(labels, y, average="macro"),
            balanced_accuracy=100 * balanced_accuracy_score(labels, y),
            members=";".join(members),
        )
    )


for n in names:
    score(n + "_tta", preds[n].mean(0), [n])
for label, members in (
    ("partial_three_seed", names[:3]),
    ("full_three_seed", names[3:6]),
    ("mn20_all", names[:6]),
    ("diverse_all", names),
):
    score(label, np.mean([preds[n][0] for n in members], axis=0), members)
    score(label + "_tta", np.mean([preds[n].mean(0) for n in members], axis=0), members)
pd.DataFrame(rows).sort_values("accuracy", ascending=False).to_csv(
    OUT / "ensemble_comparison.csv", index=False
)
print(
    pd.DataFrame(rows).sort_values("accuracy", ascending=False).to_string(index=False),
    flush=True,
)
