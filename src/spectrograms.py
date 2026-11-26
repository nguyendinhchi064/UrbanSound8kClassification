from pathlib import Path

import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

from .audio_preprocessing import preprocess_audio, SAMPLE_RATE
from .paths import PROJECT_ROOT

N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128


def generate_stft(audio):
    """Generate an STFT spectrogram in decibels."""
    stft = librosa.stft(
        audio,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    if not np.any(stft):
        return np.full(stft.shape, -80, dtype=np.float32)
    stft_db = librosa.amplitude_to_db(
        np.abs(stft),
        ref=np.max
    )

    return stft_db


def generate_log_mel(audio):
    """Generate a Log-Mel spectrogram in decibels."""
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )

    if not np.any(mel):
        return np.full_like(mel, -80)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    return mel_db


def visualize_spectrograms(audio, save_path=PROJECT_ROOT / "figures" / "spectrogram_comparison.png"):
    """Display and save STFT and Log-Mel spectrograms."""

    stft_db = generate_stft(audio)
    mel_db = generate_log_mel(audio)

    fig, axes = plt.subplots(2, 1, figsize=(12, 9))

    librosa.display.specshow(
        stft_db,
        sr=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        x_axis="time",
        y_axis="hz",
        ax=axes[0]
    )
    axes[0].set_title("STFT Spectrogram")

    librosa.display.specshow(
        mel_db,
        sr=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        x_axis="time",
        y_axis="mel",
        ax=axes[1]
    )
    axes[1].set_title("Log-Mel Spectrogram")

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.show()

    return stft_db, mel_db


