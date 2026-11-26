from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .data_loader import split_dataframe
from .features import FEATURE_FILE, load_feature_cache
from .paths import FEATURES_DIR


class SpectrogramDataset(Dataset):
    """One split of the feature cache. Items are (tensor of shape (1, 128, 173), label)."""

    def __init__(self, split, features_dir=FEATURES_DIR, transform=None):
        _, index = load_feature_cache(features_dir)
        rows = split_dataframe(index, split)
        self.split = split
        self.features_dir = Path(features_dir)
        self.rows = rows.index.to_numpy()
        self.labels = rows["classID"].to_numpy(dtype=np.int64)
        self.transform = transform
        self._features = None

    @property
    def features(self):
        # Opened lazily so DataLoader workers each map the file instead of copying the array.
        if self._features is None:
            self._features = np.load(self.features_dir / FEATURE_FILE, mmap_mode="r")
        return self._features

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_features"] = None
        return state

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        x = torch.from_numpy(np.array(self.features[self.rows[i]], dtype=np.float32)).unsqueeze(0)
        if self.transform is not None:
            x = self.transform(x)
        return x, torch.tensor(self.labels[i])


def make_dataloaders(batch_size=32, num_workers=0, features_dir=FEATURES_DIR, train_transform=None):
    """Train (shuffled), validation and test DataLoaders over the feature cache."""
    pin = torch.cuda.is_available()
    loaders = {}
    for split in ("train", "validation", "test"):
        dataset = SpectrogramDataset(split, features_dir, train_transform if split == "train" else None)
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=num_workers,
            pin_memory=pin,
            persistent_workers=num_workers > 0,
        )
    return loaders