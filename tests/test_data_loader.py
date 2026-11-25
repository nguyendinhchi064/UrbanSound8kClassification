import pytest

from src import paths


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """No env var, no saved path file, empty kagglehub cache."""
    monkeypatch.delenv("URBANSOUND8K_PATH", raising=False)
    monkeypatch.setenv("KAGGLEHUB_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(paths, "DATASET_PATH_FILE", tmp_path / ".dataset_path")
    return tmp_path


def make_dataset(folder, original_layout=False):
    csv_dir = folder / "metadata" if original_layout else folder
    csv_dir.mkdir(parents=True)
    (csv_dir / "UrbanSound8K.csv").write_text("fold,slice_file_name,classID\n")
    return folder


def test_missing_dataset_raises(isolated):
    with pytest.raises(ValueError, match="download_dataset"):
        paths.get_dataset_path()


def test_argument_and_env_var(isolated, monkeypatch):
    data = make_dataset(isolated / "data", original_layout=True)
    assert paths.get_dataset_path(data) == data.resolve()
    monkeypatch.setenv("URBANSOUND8K_PATH", str(data))
    assert paths.get_dataset_path() == data.resolve()


def test_wrong_explicit_path_is_not_ignored(isolated, monkeypatch):
    make_dataset(isolated / "cache/datasets/chrisfilo/urbansound8k/versions/1")
    monkeypatch.setenv("URBANSOUND8K_PATH", str(isolated / "nowhere"))
    with pytest.raises(FileNotFoundError):
        paths.get_dataset_path()


def test_saved_path(isolated):
    data = make_dataset(isolated / "data")
    paths.save_dataset_path(data)
    assert paths.get_dataset_path() == data.resolve()


def test_kagglehub_cache_newest_version(isolated):
    versions = isolated / "cache/datasets/chrisfilo/urbansound8k/versions"
    make_dataset(versions / "1")
    newest = make_dataset(versions / "2")
    assert paths.get_dataset_path() == newest.resolve()


def test_stale_saved_path_falls_back_to_cache(isolated):
    paths.save_dataset_path(isolated / "deleted")
    cached = make_dataset(isolated / "cache/datasets/chrisfilo/urbansound8k/versions/1")
    assert paths.get_dataset_path() == cached.resolve()


def test_empty_saved_file_is_ignored(isolated):
    paths.DATASET_PATH_FILE.write_text("  \n")
    with pytest.raises(ValueError):
        paths.get_dataset_path()
