"""Validate one trusted transfer checkpoint on fold 9 only."""

import argparse
import json
from pathlib import Path
import torch
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from torch.utils.data import DataLoader
from scripts.predict_transfer import load_transfer_checkpoint
from src.audio_transfer_models import VARIANTS
from src.datasets import SpectrogramDataset
from src.training import get_device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    device = get_device()
    model, cfg = load_transfer_checkpoint(args.checkpoint, device)
    if cfg["arch"] in VARIANTS:
        from src.audio_transfer_data import AudioDataset

        data = AudioDataset("validation")
    else:
        data = SpectrogramDataset("validation")
    assert len(data) == 816
    y, pred = [], []
    with torch.inference_mode():
        for x, labels in DataLoader(data, batch_size=args.batch_size):
            pred.extend(model(x.to(device)).argmax(1).cpu().tolist())
            y.extend(labels.tolist())
    print(
        json.dumps(
            {
                "split": "validation",
                "clips": len(y),
                "architecture": cfg["arch"],
                "accuracy": 100 * accuracy_score(y, pred),
                "macro_f1": 100 * f1_score(y, pred, average="macro"),
                "balanced_accuracy": 100 * balanced_accuracy_score(y, pred),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
