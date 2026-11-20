
import os
import soundfile as sf
import numpy as np
import pandas as pd

dataset_path = r"C:\Users\TopTech\.cache\kagglehub\datasets\chrisfilo\urbansound8k\versions\1"

audio_properties = []

print("Analyzing all UrbanSound8K audio files...")

for i in range(1, 11):
    folder = os.path.join(dataset_path, f"fold{i}")

    for filename in os.listdir(folder):
        if filename.lower().endswith(".wav"):
            path = os.path.join(folder, filename)

            info = sf.info(path)

            audio_properties.append({
                "filename": filename,
                "fold": i,
                "sampling_rate": info.samplerate,
                "duration": info.duration,
                "channels": info.channels,
                "amplitude": np.max(np.abs(sf.read(path)[0]))
            })

audio_df = pd.DataFrame(audio_properties)

print("\nTotal audio files analyzed:", len(audio_df))

print("\nSampling rate distribution:")
print(audio_df["sampling_rate"].value_counts())

print("\nChannel distribution:")
print(audio_df["channels"].value_counts())

print("\nDuration statistics:")
print(audio_df["duration"].describe())

print("\nAmplitude statistics:")
print(audio_df["amplitude"].describe())

audio_df.to_csv("audio_properties.csv", index=False)

print("\nResults saved to audio_properties.csv")
