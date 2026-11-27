"""Checkpoint handoff API for the team's final evaluator; CLI predicts one WAV.

No split is evaluated by this module. The final evaluation owner decides when
to run fold 10. Only load trusted, locally supplied checkpoint files.
"""

import argparse
from pathlib import Path
import json
import numpy as np
import torch
from src.data_loader import CLASS_NAMES, generate_log_mel
from src.audio_preprocessing import preprocess_audio
from src.transfer_models import TransferCNN
from src.audio_transfer_models import AudioTransferCNN, VARIANTS
from src.training import get_device


def load_transfer_checkpoint(path, device="cpu"):
    ck = torch.load(Path(path), map_location="cpu", weights_only=True)
    cfg = ck["config"]
    cls = AudioTransferCNN if cfg["arch"] in VARIANTS else TransferCNN
    model = cls(cfg["arch"], pretrained=False, dropout=cfg["dropout"])
    model.load_state_dict(ck["model_state_dict"], strict=True)
    model.configure(cfg["stage"])
    return model.to(device).eval(), cfg


def prepare_clip(path, arch):
    if arch in VARIANTS:
        import librosa
        from src.audio_transfer_data import frontend

        a, _ = librosa.load(path, sr=32000, mono=True)
        a = np.pad(a[:128000], (0, max(0, 128000 - len(a))))
        return frontend(torch.from_numpy(a).unsqueeze(0)).unsqueeze(1)
    return (
        torch.from_numpy(generate_log_mel(preprocess_audio(str(path))))
        .permute(2, 0, 1)
        .unsqueeze(0)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--audio", required=True, type=Path)
    a = parser.parse_args()
    device = get_device()
    model, cfg = load_transfer_checkpoint(a.checkpoint, device)
    x = prepare_clip(a.audio, cfg["arch"]).to(device)
    with torch.inference_mode():
        probs = model(x).softmax(1).cpu()[0]
    idx = int(probs.argmax())
    print(
        json.dumps(
            {
                "class_id": idx,
                "class_name": CLASS_NAMES[idx],
                "probability": float(probs[idx]),
                "probabilities": dict(zip(CLASS_NAMES, probs.tolist())),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
