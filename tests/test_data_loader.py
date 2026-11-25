from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from src.audio_preprocessing import preprocess_audio, TARGET_LENGTH
from src.spectrograms import generate_stft
from src.data_loader import split_metadata, generate_log_mel, load_data, TRAIN_FOLDS, VALIDATION_FOLDS, TEST_FOLDS, INPUT_SHAPE

@pytest.mark.parametrize("rate,duration,channels", [(44100, 1, 2), (16000, 5, 1), (22050, 4, 1)])
def test_preprocessing(tmp_path, rate, duration, channels):
    wave = .3 * np.sin(2 * np.pi * 440 * np.arange(rate * duration) / rate)
    if channels == 2:
        wave = np.column_stack([wave, wave])
    path = tmp_path / "sample.wav"
    sf.write(path, wave, rate)
    audio = preprocess_audio(path)
    assert audio.shape == (TARGET_LENGTH,)
    assert np.isclose(np.max(np.abs(audio)), 1)
    assert generate_stft(audio).shape == (1025, 173)
    mel = generate_log_mel(audio)
    assert mel.shape == INPUT_SHAPE
    assert mel.dtype == np.float32
    assert np.isfinite(mel).all() and 0 <= mel.min() <= mel.max() <= 1

def test_silence():
    assert np.all(generate_log_mel(np.zeros(TARGET_LENGTH, dtype=np.float32)) == 0)

def test_fold_split(tmp_path):
    df = pd.read_csv(Path(__file__).resolve().parents[1] / "metadata/UrbanSound8K.csv")
    groups = [set(f) for f in (TRAIN_FOLDS, VALIDATION_FOLDS, TEST_FOLDS)]
    assert set.union(*groups) == set(range(1, 11))
    assert sum(map(len, groups)) == 10
    df.to_csv(tmp_path / "UrbanSound8K.csv", index=False)
    parts = [split_metadata(s, tmp_path) for s in ("train", "validation", "test")]
    sources = [set(p.fsID) for p in parts]
    assert [len(p) for p in parts] == [7000, 816, 837]
    assert not (sources[0] & sources[1] or sources[0] & sources[2] or sources[1] & sources[2])
    assert len(df) == 8732
    assert set(df.classID) == set(range(10))

def test_loader_all_folds(tmp_path):
    rows = []
    for fold in range(1, 11):
        folder = tmp_path / f"fold{fold}"
        folder.mkdir()
        sf.write(folder / "sample.wav", np.zeros(22050), 22050)
        rows.append(dict(fold=fold, slice_file_name="sample.wav", classID=fold-1))
    pd.DataFrame(rows).to_csv(tmp_path / "UrbanSound8K.csv", index=False)
    X, y = load_data(range(1, 11), tmp_path)
    assert X.shape == (10, *INPUT_SHAPE)
    assert np.array_equal(y, np.arange(10))
    assert np.isfinite(X).all()
