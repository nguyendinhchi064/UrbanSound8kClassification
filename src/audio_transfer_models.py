"""AudioSet-pretrained EfficientAT transfer, pinned upstream and strict weights."""

from pathlib import Path
import sys
import os
import subprocess
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "data/upstream/EfficientAT"
REV = "a425fdce92572e602a1d5634799bd9f1f2efa806"
VARIANTS = {"mn10_as": 1.0, "mn20_as": 2.0, "mn40_as": 4.0}


def upstream():
    if not UPSTREAM.exists():
        raise FileNotFoundError("Run scripts/setup_audio_transfer.sh")
    sha = subprocess.check_output(
        ["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True
    ).strip()
    if sha != REV:
        raise RuntimeError(f"Unexpected upstream revision {sha}")
    sys.path.insert(0, str(UPSTREAM))
    cwd = Path.cwd()
    try:
        os.chdir(UPSTREAM)
        from models.mn import model as module
    finally:
        os.chdir(cwd)
    return module


class AudioTransferCNN(nn.Module):
    def __init__(self, variant, pretrained=True, dropout=0.25):
        super().__init__()
        module = upstream()
        base = module.get_model(width_mult=VARIANTS[variant], pretrained_name=None)
        if pretrained:
            state = torch.hub.load_state_dict_from_url(
                module.pretrained_models[variant],
                model_dir=str(ROOT / "data/upstream/weights"),
                map_location="cpu",
                weights_only=True,
            )
            base.load_state_dict(state, strict=True)
        dim = base.classifier[-1].in_features
        self.backbone = nn.Sequential(
            base.features, *list(base.classifier.children())[:-2]
        )
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(dim, 10))
        self.stage = "frozen"
        self.configure("frozen")

    def configure(self, stage):
        if stage not in ("frozen", "partial", "full"):
            raise ValueError(stage)
        self.stage = stage
        self.backbone.requires_grad_(stage == "full")
        if stage == "partial":
            self.backbone[0][-5:].requires_grad_(True)
            for layer in list(self.backbone.children())[1:]:
                layer.requires_grad_(True)
        self.head.requires_grad_(True)
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        if mode and self.stage == "frozen":
            self.backbone.eval()
        elif mode and self.stage == "partial":
            for m in self.backbone.modules():
                if not any(p.requires_grad for p in m.parameters()):
                    m.eval()
        return self

    def forward(self, x):
        if self.stage == "frozen":
            with torch.no_grad():
                x = self.backbone(x)
        else:
            x = self.backbone(x)
        return self.head(x)
