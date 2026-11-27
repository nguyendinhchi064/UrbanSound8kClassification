"""EfficientAT-compatible inference frontend, without torchaudio dependency."""

from pathlib import Path
import json
import numpy as np
import torch
import librosa
from torch.utils.data import Dataset
from src.data_loader import load_metadata, split_dataframe, audio_file

ROOT = Path("data/audio_transfer")
SR = 32000
FRAMES = 400


def frontend(wave):
    # Kaldi triangular mel banks: 0–15000 Hz, 512 FFT bins plus zero Nyquist.
    def mel(hz):
        return 1127.0 * torch.log1p(torch.as_tensor(hz) / 700.0)

    edges = torch.linspace(mel(0.0), mel(15000.0), 130)
    bins = mel(torch.arange(512, dtype=torch.float32) * SR / 1024)
    left = (bins[None, :] - edges[:-2, None]) / (edges[1:-1, None] - edges[:-2, None])
    right = (edges[2:, None] - bins[None, :]) / (edges[2:, None] - edges[1:-1, None])
    bank = torch.nn.functional.pad(torch.minimum(left, right).clamp_min(0), (0, 1))
    wave = wave[:, 1:] - 0.97 * wave[:, :-1]
    stft = torch.stft(
        wave,
        1024,
        hop_length=320,
        win_length=800,
        window=torch.hann_window(800, periodic=False),
        center=True,
        return_complex=True,
    )
    return ((bank @ stft.abs().square() + 1e-5).log() + 4.5) / 5.0


def build():
    torch.set_num_threads(4)
    ROOT.mkdir(parents=True, exist_ok=True)
    meta = load_metadata()
    for split in ("train", "validation"):
        rows = split_dataframe(meta, split).reset_index(drop=True)
        dst = ROOT / f"{split}.npy"
        marker = ROOT / f"{split}.json"
        if marker.exists() and dst.exists():
            continue
        temp = dst.with_suffix(".tmp.npy")
        mm = np.lib.format.open_memmap(
            temp, mode="w+", dtype=np.float32, shape=(len(rows), 128, FRAMES)
        )
        for i in range(0, len(rows), 32):
            waves = []
            for row in rows.iloc[i : i + 32].to_dict("records"):
                a, _ = librosa.load(audio_file(row), sr=SR, mono=True)
                a = np.pad(a[: 4 * SR], (0, max(0, 4 * SR - len(a))))
                waves.append(a)
            x = frontend(torch.from_numpy(np.stack(waves))).numpy()
            assert x.shape[1:] == (128, FRAMES) and np.isfinite(x).all()
            mm[i : i + len(waves)] = x
            if i % 320 == 0:
                print(split, i, len(rows), flush=True)
        mm.flush()
        del mm
        temp.replace(dst)
        rows[["slice_file_name", "fsID", "fold", "classID"]].to_csv(
            ROOT / f"{split}_index.csv", index=False
        )
        marker.write_text(
            json.dumps(
                {
                    "clips": len(rows),
                    "sr": SR,
                    "frames": FRAMES,
                    "frontend": "EfficientAT eval parameters; fixed zero padding to 4 s",
                }
            )
        )


class AudioDataset(Dataset):
    def __init__(self, split):
        import pandas as pd

        assert split in ("train", "validation")
        assert (ROOT / f"{split}.json").exists()
        self.x = np.load(ROOT / f"{split}.npy", mmap_mode="r")
        self.index = pd.read_csv(ROOT / f"{split}_index.csv")
        assert len(self.index) == len(self.x)
        self.y = self.index.classID.to_numpy()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return torch.from_numpy(np.array(self.x[i])).unsqueeze(0), torch.tensor(
            self.y[i], dtype=torch.long
        )


if __name__ == "__main__":
    build()
