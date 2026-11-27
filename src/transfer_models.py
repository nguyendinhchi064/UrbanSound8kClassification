"""ImageNet transfer learning on unresized UrbanSound8K log-mel inputs."""

import torch
from torch import nn
from torchvision import models

ARCHS = ("resnet18", "mobilenet_v2", "efficientnet_b0", "densenet121")


class TransferCNN(nn.Module):
    def __init__(self, architecture, pretrained=True, dropout=0.3):
        super().__init__()
        self.architecture = architecture
        weights = {
            "resnet18": models.ResNet18_Weights.DEFAULT,
            "mobilenet_v2": models.MobileNet_V2_Weights.DEFAULT,
            "efficientnet_b0": models.EfficientNet_B0_Weights.DEFAULT,
            "densenet121": models.DenseNet121_Weights.DEFAULT,
        }
        self.backbone = getattr(models, architecture)(
            weights=weights[architecture] if pretrained else None
        )
        if architecture == "resnet18":
            n = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif architecture == "densenet121":
            n = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
        else:
            n = self.backbone.classifier[-1].in_features
            self.backbone.classifier = nn.Identity()
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(n, 10))
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )
        self.stage = "frozen"
        self.configure("frozen")

    def configure(self, stage):
        if stage not in ("frozen", "partial", "full"):
            raise ValueError(stage)
        self.stage = stage
        self.backbone.requires_grad_(stage == "full")
        if stage == "partial":
            if self.architecture == "resnet18":
                self.backbone.layer4.requires_grad_(True)
            elif self.architecture == "densenet121":
                self.backbone.features.denseblock4.requires_grad_(True)
                self.backbone.features.norm5.requires_grad_(True)
            else:
                self.backbone.features[-3:].requires_grad_(True)
        self.head.requires_grad_(True)
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        if mode and self.stage == "frozen":
            self.backbone.eval()
        elif mode and self.stage == "partial":
            # Keep frozen modules' running statistics and dropout fixed.
            for module in self.backbone.modules():
                if not any(p.requires_grad for p in module.parameters()):
                    module.eval()
        return self

    def forward(self, x):
        x = (x.expand(-1, 3, -1, -1) - self.mean) / self.std
        if self.stage == "frozen":
            with torch.no_grad():
                x = self.backbone(x)
        else:
            x = self.backbone(x)
        return self.head(x)
