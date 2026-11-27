import pytest
import torch
from src.transfer_models import TransferCNN, ARCHS


@pytest.mark.parametrize("arch", ARCHS)
def test_transfer(arch):
    model = TransferCNN(arch, pretrained=False)
    model.train()
    assert not model.backbone.training
    x = torch.rand(2, 1, 128, 173)
    y = model(x)
    assert y.shape == (2, 10)
    y.sum().backward()
    assert all(p.grad is None for p in model.backbone.parameters())
    assert all(p.grad is not None for p in model.head.parameters())
    model.configure("partial")
    model.train()
    assert any(p.requires_grad for p in model.backbone.parameters())
    assert any(not p.requires_grad for p in model.backbone.parameters())
    assert all(
        not m.training
        for m in model.backbone.modules()
        if isinstance(m, torch.nn.BatchNorm2d)
        and not any(p.requires_grad for p in m.parameters())
    )
    model.configure("full")
    assert all(p.requires_grad for p in model.parameters())
