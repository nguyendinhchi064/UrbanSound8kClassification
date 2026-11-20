
import librosa
import numpy as np

# Standard audio settings
SAMPLE_RATE = 22050
DURATION = 4
TARGET_LENGTH = SAMPLE_RATE * DURATION


def preprocess_audio(file_path):
    # 1. Load audio and convert it to mono
    audio, sr = librosa.load(
        file_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    # 2. Make every audio file exactly 4 seconds
    if len(audio) > TARGET_LENGTH:
        audio = audio[:TARGET_LENGTH]
    else:
        audio = np.pad(
            audio,
            (0, TARGET_LENGTH - len(audio))
        )

    # 3. Normalize audio
    max_amplitude = np.max(np.abs(audio))

    if max_amplitude > 0:
        audio = audio / max_amplitude

    return audio


# Test the function
if __name__ == "__main__":

    file_path = r"C:\Users\TopTech\.cache\kagglehub\datasets\chrisfilo\urbansound8k\versions\1\fold1\101415-3-0-2.wav"

    processed_audio = preprocess_audio(file_path)

    print("PREPROCESSING RESULTS")
    print("---------------------")
    print("Sampling rate:", SAMPLE_RATE)
    print("Duration:", len(processed_audio) / SAMPLE_RATE, "seconds")
    print("Number of samples:", len(processed_audio))
    print("Maximum amplitude:", np.max(np.abs(processed_audio)))
