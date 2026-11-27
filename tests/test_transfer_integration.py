"""Integration tests for the Member 4 training and inference interfaces."""

import torch
import pytest
from src.transfer_models import TransferCNN
from scripts.predict_transfer import load_transfer_checkpoint
from scripts.reproduce_transfer import lineage, command
from scripts.train_transfer import augment, epoch
from torch.utils.data import DataLoader, TensorDataset


def test_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(42)
    model = TransferCNN("resnet18", pretrained=False).eval()
    x = torch.rand(2, 1, 128, 173)
    checkpoint = tmp_path / "model.pth"
    cfg = {"arch": "resnet18", "stage": "frozen", "dropout": 0.3}
    torch.save({"config": cfg, "model_state_dict": model.state_dict()}, checkpoint)
    loaded, restored = load_transfer_checkpoint(checkpoint)
    assert restored == cfg
    with torch.inference_mode():
        torch.testing.assert_close(loaded(x), model(x))


def test_frozen_batchnorm_state_unchanged():
    model = TransferCNN("resnet18", pretrained=False).train()
    bn = model.backbone.bn1
    old = bn.running_mean.clone()
    out = model(torch.rand(2, 1, 128, 173))
    out.sum().backward()
    torch.testing.assert_close(bn.running_mean, old)
    assert not bn.training


def test_partial_backbone_gradients():
    model = TransferCNN("resnet18", pretrained=False)
    model.configure("partial")
    model.train()
    model(torch.rand(2, 1, 128, 173)).sum().backward()
    assert model.backbone.conv1.weight.grad is None
    assert model.backbone.layer4[0].conv1.weight.grad is not None


def test_augmentation_does_not_change_input():
    x = torch.ones(3, 1, 128, 173)
    before = x.clone()
    z = augment(x)
    torch.testing.assert_close(x, before)
    assert z.shape == x.shape


def test_validation_epoch_does_not_update_parameters():
    model = TransferCNN("resnet18", pretrained=False)
    p = model.head[-1].weight.detach().clone()
    loader = DataLoader(
        TensorDataset(torch.rand(2, 1, 128, 173), torch.tensor([0, 1])), batch_size=2
    )
    result = epoch(model, loader, torch.device("cpu"))
    assert set(result) == {"loss", "accuracy", "macro_f1", "balanced_accuracy"}
    torch.testing.assert_close(model.head[-1].weight, p)


def test_reproduction_lineage(tmp_path):
    configs = lineage("resnet18_repeat_full_2024")
    assert [c["stage"] for c in configs] == ["frozen", "partial", "full"]
    assert all(c["seed"] == 2024 for c in configs)
    cmd = command(configs[-1], tmp_path, tmp_path / "parent.pth")
    assert "--specaugment" in cmd and "--from-checkpoint" in cmd
    assert str(tmp_path / "reports") in cmd


def test_unknown_replay_rejected():
    with pytest.raises(ValueError):
        lineage("../outside")
