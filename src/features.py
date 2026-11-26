import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .audio_preprocessing import SAMPLE_RATE, DURATION
from .data_loader import load_metadata, load_sample, INPUT_SHAPE
from .paths import FEATURES_DIR, get_dataset_path
from .spectrograms import N_MELS, N_FFT, HOP_LENGTH

FEATURE_FILE = "logmel.npy"
INDEX_FILE = "index.csv"
INFO_FILE = "info.json"
INDEX_COLUMNS = ["slice_file_name", "fsID", "start", "end", "fold", "classID", "class"]


def _settings():
    return {
        "representation": "log-mel",
        "sample_rate": SAMPLE_RATE,
        "duration": DURATION,
        "n_fft": N_FFT,
        "hop_length": HOP_LENGTH,
        "n_mels": N_MELS,
        "shape": list(INPUT_SHAPE[:2]),
        "value_range": [0.0, 1.0],
    }


def _compute(job):
    row, root = job
    spectrogram, _ = load_sample(row, root)
    return spectrogram[..., 0]


def build_feature_cache(dataset_path=None, out_dir=FEATURES_DIR, workers=None, overwrite=False):
    """Write every clip's Log-Mel spectrogram to out_dir/logmel.npy, in metadata order."""
    root = get_dataset_path(dataset_path)
    out_dir = Path(out_dir)
    info_path = out_dir / INFO_FILE
    if info_path.exists() and not overwrite:
        raise FileExistsError(f"Feature cache already exists in {out_dir}. Use overwrite=True to rebuild.")
    out_dir.mkdir(parents=True, exist_ok=True)
    info_path.unlink(missing_ok=True)  # info.json is written last and marks a complete cache

    df = load_metadata(root).reset_index(drop=True)
    jobs = [(row, root) for row in df[["slice_file_name", "fold", "classID"]].to_dict("records")]
    features = np.lib.format.open_memmap(
        out_dir / FEATURE_FILE, mode="w+", dtype=np.float32, shape=(len(df), *INPUT_SHAPE[:2])
    )

    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    progress = dict(total=len(jobs), desc="Log-Mel", unit="clip")
    if workers == 1:
        for i, spectrogram in enumerate(tqdm(map(_compute, jobs), **progress)):
            features[i] = spectrogram
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = executor.map(_compute, jobs, chunksize=16)
            for i, spectrogram in enumerate(tqdm(results, **progress)):
                features[i] = spectrogram
    features.flush()
    del features

    df[INDEX_COLUMNS].to_csv(out_dir / INDEX_FILE, index=False)
    info = {**_settings(), "clips": len(df), "dataset_path": str(root)}
    info_path.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    return out_dir


def load_feature_cache(features_dir=FEATURES_DIR, mmap=True):
    """Return (features, index): features has shape (clips, 128, 173), row i matches index row i."""
    import pandas as pd

    features_dir = Path(features_dir)
    info_path = features_dir / INFO_FILE
    if not info_path.is_file():
        raise FileNotFoundError(
            f"No complete feature cache in {features_dir}. Run `python -m scripts.prepare_features` first."
        )
    info = json.loads(info_path.read_text(encoding="utf-8"))
    current = _settings()
    changed = {k: (info.get(k), v) for k, v in current.items() if info.get(k) != v}
    if changed:
        raise ValueError(
            f"Feature cache was built with different settings {changed}. "
            "Rebuild it with `python -m scripts.prepare_features --overwrite`."
        )
    features = np.load(features_dir / FEATURE_FILE, mmap_mode="r" if mmap else None)
    index = pd.read_csv(features_dir / INDEX_FILE)
    if len(features) != len(index):
        raise ValueError("Feature cache is inconsistent: index and features have different lengths.")
    return features, index