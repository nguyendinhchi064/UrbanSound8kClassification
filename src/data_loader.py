
import os
from pathlib import Path

import pandas as pd
import numpy as np

from .audio_preprocessing import preprocess_audio, SAMPLE_RATE, DURATION, TARGET_LENGTH
from .spectrograms import generate_log_mel as log_mel_db, N_MELS, N_FFT, HOP_LENGTH

TRAIN_FOLDS = tuple(range(1, 9))
VALIDATION_FOLDS = (9,)
TEST_FOLDS = (10,)
INPUT_SHAPE = (128, 173, 1)

CLASS_NAMES = [
    "air_conditioner",
    "car_horn",
    "children_playing",
    "dog_bark",
    "drilling",
    "engine_idling",
    "gun_shot",
    "jackhammer",
    "siren",
    "street_music"
]


def get_dataset_path(dataset_path=None):
    """Find the UrbanSound8K dataset folder."""

    path = dataset_path or os.environ.get("URBANSOUND8K_PATH")

    if not path:
        raise ValueError(
            "Dataset path not configured. Set URBANSOUND8K_PATH "
            "or provide dataset_path."
        )

    path = Path(path).expanduser().resolve()

    if not ((path / "UrbanSound8K.csv").exists() or (path / "metadata/UrbanSound8K.csv").exists()):
        raise FileNotFoundError(
            f"UrbanSound8K.csv not found in: {path}"
        )

    return path


def load_metadata(dataset_path=None):
    """Load dataset metadata."""
    path = get_dataset_path(dataset_path)
    csv = path / "UrbanSound8K.csv"
    return pd.read_csv(csv if csv.is_file() else path / "metadata/UrbanSound8K.csv")


def generate_log_mel(audio):
    """Generate a normalized Log-Mel spectrogram."""

    mel_db = log_mel_db(audio)
    mel_db = np.clip(mel_db, -80, 0)
    mel_db = (mel_db + 80) / 80

    mel_db = mel_db[..., np.newaxis]

    return mel_db.astype(np.float32)


def split_metadata(split, dataset_path=None):
    """Use official folds, removing shared source IDs from earlier partitions."""
    df = load_metadata(dataset_path)
    train = df[df.fold.isin(TRAIN_FOLDS)]
    val = df[df.fold.isin(VALIDATION_FOLDS)]
    test = df[df.fold.isin(TEST_FOLDS)]
    val = val[~val.fsID.isin(test.fsID)]
    train = train[~train.fsID.isin(pd.concat([val, test]).fsID)]
    return {"train": train, "validation": val, "test": test}[split]


def load_split(split, dataset_path=None):
    root = get_dataset_path(dataset_path)
    samples = [load_sample(row, root) for _, row in split_metadata(split, root).iterrows()]
    X, y = zip(*samples)
    return np.stack(X), np.asarray(y, dtype=np.int64)


def load_sample(row, dataset_path=None):
    """Load one audio file and its class label."""

    path = get_dataset_path(dataset_path)

    fold = int(row["fold"])
    filename = row["slice_file_name"]

    file_path = path / f"fold{fold}" / filename

    if not file_path.exists():
        file_path = path / "audio" / f"fold{fold}" / filename
    if not file_path.exists():
        raise FileNotFoundError(file_path)

    audio = preprocess_audio(str(file_path))
    spectrogram = generate_log_mel(audio)

    label = int(row["classID"])

    return spectrogram, label


def iter_samples(folds, dataset_path=None):
    folds = tuple(folds)
    if not folds or not set(folds).issubset(range(1, 11)):
        raise ValueError("Select folds from 1 to 10.")
    path = get_dataset_path(dataset_path)
    df = load_metadata(path)
    for _, row in df[df["fold"].isin(folds)].iterrows():
        yield load_sample(row, path)


def load_data(folds=(1,), dataset_path=None):
    samples = list(iter_samples(folds, dataset_path))
    if not samples:
        raise ValueError("No recordings in selected folds.")
    X, y = zip(*samples)
    return np.stack(X), np.asarray(y, dtype=np.int64)


if __name__ == "__main__":

    print("Testing data loader...")

    metadata = load_metadata()

    print("Total audio files:", len(metadata))

    sample, label = load_sample(metadata.iloc[0])

    print("Spectrogram shape:", sample.shape)
    print("Class ID:", label)
    print("Class name:", CLASS_NAMES[label])
    print("Minimum value:", sample.min())
    print("Maximum value:", sample.max())

    print("\nData loader works successfully!")
