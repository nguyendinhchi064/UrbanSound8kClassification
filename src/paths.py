"""Locate the UrbanSound8K dataset without heavy dependencies."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH_FILE = PROJECT_ROOT / ".dataset_path"
KAGGLE_HANDLE = "chrisfilo/urbansound8k"


def has_metadata(path):
    path = Path(path)
    return (path / "UrbanSound8K.csv").is_file() or (path / "metadata" / "UrbanSound8K.csv").is_file()


def save_dataset_path(path):
    DATASET_PATH_FILE.write_text(str(Path(path).resolve()), encoding="utf-8")


def saved_dataset_path():
    if not DATASET_PATH_FILE.is_file():
        return None
    text = DATASET_PATH_FILE.read_text(encoding="utf-8").strip()
    return Path(text) if text else None


def kagglehub_cache_versions():
    """Downloaded Kaggle versions in the kagglehub cache, newest first."""
    cache_root = Path(os.environ.get("KAGGLEHUB_CACHE") or Path.home() / ".cache" / "kagglehub")
    versions_dir = cache_root / "datasets" / KAGGLE_HANDLE / "versions"
    if not versions_dir.is_dir():
        return []
    versions = [p for p in versions_dir.iterdir() if p.is_dir() and p.name.isdigit()]
    return sorted(versions, key=lambda p: int(p.name), reverse=True)


def get_dataset_path(dataset_path=None):
    """Find the UrbanSound8K dataset folder.

    Order: dataset_path argument, URBANSOUND8K_PATH, the path saved by
    scripts.download_dataset, then the kagglehub cache.
    """
    explicit = dataset_path or os.environ.get("URBANSOUND8K_PATH")
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not has_metadata(path):
            raise FileNotFoundError(f"UrbanSound8K.csv not found in: {path}")
        return path

    saved = saved_dataset_path()
    candidates = ([saved] if saved else []) + kagglehub_cache_versions()
    for candidate in candidates:
        path = candidate.expanduser().resolve()
        if has_metadata(path):
            return path

    raise ValueError(
        "Dataset not found. Run `python -m scripts.download_dataset`, "
        "or set URBANSOUND8K_PATH / provide dataset_path."
    )
