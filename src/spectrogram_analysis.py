"""Helpers for the spectrogram analysis notebook: compare STFT and Log-Mel, inspect input dimensions."""

import time

import numpy as np
import pandas as pd

from .audio_preprocessing import DURATION, SAMPLE_RATE, preprocess_audio
from .data_loader import CLASS_NAMES, audio_file, split_dataframe
from .paths import get_dataset_path
from .spectrograms import HOP_LENGTH, N_FFT, N_MELS, generate_log_mel, generate_stft

REPRESENTATIONS = ("STFT", "Log-Mel")
TOTAL_CLIPS = 8732
TOTAL_FRAMES = 1 + DURATION * SAMPLE_RATE // HOP_LENGTH  # 173 frames for 4 s


def to_unit_range(db):
    """Map decibels relative to the clip maximum from [-80, 0] to [0, 1], as the data loader does."""
    return ((np.clip(db, -80, 0) + 80) / 80).astype(np.float32)


def representations(audio):
    """Both representations of one preprocessed clip, scaled to [0, 1]."""
    return {"STFT": to_unit_range(generate_stft(audio)), "Log-Mel": to_unit_range(generate_log_mel(audio))}


def representation_table(specs):
    """Size and frequency resolution of each representation (specs as returned by representations)."""
    rows = []
    for name, spec in specs.items():
        bins, frames = spec.shape
        resolution = f"{SAMPLE_RATE / N_FFT:.1f} Hz per bin (linear)" if name == "STFT" else f"{N_MELS} mel bands"
        rows.append({
            "representation": name,
            "frequency bins": bins,
            "time frames": frames,
            "frequency axis": resolution,
            "values per clip": bins * frames,
            "KB per clip (float32)": round(bins * frames * 4 / 1024, 1),
            "GB for 8,732 clips": round(bins * frames * 4 * TOTAL_CLIPS / 1024 ** 3, 2),
        })
    table = pd.DataFrame(rows).set_index("representation")
    table["size vs Log-Mel"] = (table["values per clip"] / table.loc["Log-Mel", "values per clip"]).round(1)
    return table


def class_examples(index, per_class=1, seed=0):
    """A few random rows of each class, ordered by classID."""
    return index.groupby("classID", group_keys=False).sample(per_class, random_state=seed).sort_values("classID")


def class_means(features, index, split="train"):
    """Average spectrogram of each class over one split: array of shape (10, mel bands, frames)."""
    rows = split_dataframe(index, split)
    return np.stack([features[rows.index[rows.classID == c].to_numpy()].mean(axis=0) for c in range(len(CLASS_NAMES))])


def content_frames(index):
    """Frames that hold real audio per clip; the rest of the 173 frames is zero padding."""
    seconds = (index["end"] - index["start"]).clip(upper=DURATION)
    return (1 + np.floor(seconds * SAMPLE_RATE / HOP_LENGTH)).astype(int).clip(upper=TOTAL_FRAMES)


def duration_table(index):
    """Per class: clips, mean duration and share of clips shorter than 4 s."""
    seconds = (index["end"] - index["start"]).clip(upper=DURATION)
    table = pd.DataFrame({"class": index["class"], "seconds": seconds, "short": seconds < DURATION - 0.01})
    return table.groupby("class").agg(clips=("seconds", "size"), mean_seconds=("seconds", "mean"),
                                      share_shorter_than_4s=("short", "mean")).round(3)


def summary_vector(spec):
    """Mean and standard deviation over time of every frequency bin."""
    return np.concatenate([spec.mean(axis=1), spec.std(axis=1)])


def sample_rows(index, split, per_class=None, seed=0):
    rows = split_dataframe(index, split)
    if per_class:
        rows = rows.sample(frac=1, random_state=seed).groupby("classID").head(per_class).sort_index()
    return rows


def summary_features(rows, dataset_path=None, progress=None):
    """Summary vectors of both representations for each row, decoding every clip once."""
    root = get_dataset_path(dataset_path)
    vectors = {name: [] for name in REPRESENTATIONS}
    iterator = rows.to_dict("records")
    if progress:
        iterator = progress(iterator, total=len(rows))
    for row in iterator:
        specs = representations(preprocess_audio(str(audio_file(row, root))))
        for name in REPRESENTATIONS:
            vectors[name].append(summary_vector(specs[name]))
    return {name: np.stack(v) for name, v in vectors.items()}, rows["classID"].to_numpy()


def compare_representations(train, y_train, validation, y_validation):
    """Fit the same linear baseline on each representation and score it on the validation fold."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rows = []
    for name in train:
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
        start = time.perf_counter()
        model.fit(train[name], y_train)
        seconds = time.perf_counter() - start
        predicted = model.predict(validation[name])
        rows.append({
            "representation": name,
            "features": train[name].shape[1],
            "validation accuracy": round(accuracy_score(y_validation, predicted), 3),
            "validation macro F1": round(f1_score(y_validation, predicted, average="macro"), 3),
            "fit seconds": round(seconds, 2),
        })
    return pd.DataFrame(rows).set_index("representation")
