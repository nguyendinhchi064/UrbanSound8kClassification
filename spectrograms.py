
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

from audio_preprocessing import preprocess_audio, SAMPLE_RATE

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

    mel_db = librosa.power_to_db(mel, ref=np.max)

    return mel_db


def visualize_spectrograms(audio, save_path="spectrogram_comparison.png"):
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
    plt.savefig(save_path)
    plt.show()

    return stft_db, mel_db


if __name__ == "__main__":
    file_path = r"C:\Users\TopTech\.cache\kagglehub\datasets\chrisfilo\urbansound8k\versions\1\fold1\101415-3-0-2.wav"

    audio = preprocess_audio(file_path)

    stft_db, mel_db = visualize_spectrograms(audio)

    print("STFT shape:", stft_db.shape)
    print("Log-Mel shape:", mel_db.shape)
    print("Both spectrograms generated successfully!")
