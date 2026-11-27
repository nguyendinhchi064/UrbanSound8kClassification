import torch
from src.audio_transfer_data import frontend


def test_frontend_shape_and_finite():
    x = torch.zeros(2, 128000)
    out = frontend(x)
    assert out.shape == (2, 128, 400)
    assert torch.isfinite(out).all()
    assert torch.equal(out, frontend(x))


def test_frontend_has_signal_information():
    x = torch.zeros(2, 128000)
    t = torch.arange(128000) / 32000
    x[1] = 0.5 * torch.sin(2 * torch.pi * 1000 * t)
    out = frontend(x)
    assert not torch.equal(out[0], out[1])
