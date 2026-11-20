
import os
from pathlib import Path

import pandas as pd
import numpy as np
import librosa

# Dataset configuration
SAMPLE_RATE = 22050
DURATION = 4
TARGET_LENGTH = SAMPLE_RATE * DURATION

N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512

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

    path = Path(path)

    if not (path / "UrbanSound8K.csv").exists():
        raise FileNotFoundError(
            f"UrbanSound8K.csv not found in: {path}"
        )

    return path


def load_metadata(dataset_path=None):
    """Load dataset metadata."""
    path = get_dataset_path(dataset_path)
    return pd.read_csv(path / "UrbanSound8K.csv")


def preprocess_audio(file_path):
    """Load and standardize one audio file."""

    audio, sr = librosa.load(
        file_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    if len(audio) > TARGET_LENGTH:
        audio = audio[:TARGET_LENGTH]
    elif len(audio) < TARGET_LENGTH:
        audio = np.pad(
            audio,
            (0, TARGET_LENGTH - len(audio))
        )

    max_amplitude = np.max(np.abs(audio))

    if max_amplitude > 0:
        audio = audio / max_amplitude

    return audio


def generate_log_mel(audio):
    """Generate a normalized Log-Mel spectrogram."""

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )

    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = np.clip(mel_db, -80, 0)
    mel_db = (mel_db + 80) / 80

    mel_db = mel_db[..., np.newaxis]

    return mel_db.astype(np.float32)


def load_sample(row, dataset_path=None):
    """Load one audio file and its class label."""

    path = get_dataset_path(dataset_path)

    fold = int(row["fold"])
    filename = row["slice_file_name"]

    file_path = path / f"fold{fold}" / filename

    if not file_path.exists():
        raise FileNotFoundError(file_path)

    audio = preprocess_audio(str(file_path))
    spectrogram = generate_log_mel(audio)

    label = int(row["classID"])

    return spectrogram, label


def load_data(folds=(1,), dataset_path=None):
    """Load selected folds and return spectrograms and labels."""

    df = load_metadata(dataset_path)
    df = df[df["fold"].isin(folds)]

    X = []
    y = []

    for _, row in df.iterrows():
        spectrogram, label = load_sample(row, dataset_path)

        X.append(spectrogram)
        y.append(label)

    return (
        np.array(X, dtype=np.float32),
        np.array(y, dtype=np.int64)
    )


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
