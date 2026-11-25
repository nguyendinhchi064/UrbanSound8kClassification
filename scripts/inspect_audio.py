import os
from src.data_loader import get_dataset_path, load_metadata

import soundfile as sf
import numpy as np

# Path to one audio file
audio_path = str(get_dataset_path() / "fold1" / "101415-3-0-2.wav")

# Read audio information
if not os.path.isfile(audio_path):
    audio_path = str(get_dataset_path() / "audio/fold1/101415-3-0-2.wav")
audio, sample_rate = sf.read(audio_path)

# Calculate audio properties
duration = len(audio) / sample_rate
channels = 1 if audio.ndim == 1 else audio.shape[1]
amplitude = np.max(np.abs(audio))

print("AUDIO INFORMATION")
print("-----------------")
print("Sampling rate:", sample_rate, "Hz")
print("Duration:", round(duration, 2), "seconds")
print("Number of channels:", channels)
print("Maximum amplitude:", round(amplitude, 4))
