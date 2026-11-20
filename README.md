# UrbanSound8K Environmental Sound Classification

## Project Overview

This project prepares the UrbanSound8K dataset for environmental sound classification using a Convolutional Neural Network (CNN).

Member 1 is responsible for audio processing and dataset preparation.

## Dataset

- Dataset: UrbanSound8K
- Source: https://www.kaggle.com/datasets/chrisfilo/urbansound8k
- Total recordings: 8,732
- Number of sound classes: 10

Download and extract the dataset. The dataset folder should contain:
- UrbanSound8K.csv
- fold1/ through fold10/

The raw audio files are not included in this Git repository.

## Technologies

- Python
- NumPy
- Pandas
- Librosa
- SoundFile
- Matplotlib
- Jupyter Notebook

## Installation

Install the required libraries:

\\\ash
pip install numpy pandas librosa soundfile matplotlib jupyter
\\\

## Dataset Configuration

Set the dataset location using the URBANSOUND8K_PATH environment variable.

Example in Windows PowerShell:

\\\powershell
C:\Users\TopTech\.cache\kagglehub\datasets\chrisfilo\urbansound8k\versions\1 = "C:\path\to\UrbanSound8K"
\\\

Replace the example path with the location where you extracted the dataset.

## Member 1: Audio Processing and Dataset Preparation

The implemented tasks include:

1. Dataset exploration and class distribution analysis.
2. Audio properties analysis:
   - Sampling rates
   - Durations
   - Number of channels
   - Signal amplitudes
3. Audio preprocessing:
   - Resampling to 22,050 Hz
   - Conversion to mono
   - Padding or cropping to 4 seconds
   - Amplitude normalization
4. STFT spectrogram generation.
5. Log-Mel spectrogram generation.
6. Spectrogram visualization and comparison.
7. Reusable data loading functions for CNN input.

## Main Files

| File | Description |
|---|---|
| dataset_exploration.ipynb | Dataset exploration and visualizations |
| explore_dataset.py | Dataset metadata and class analysis |
| analyze_audio.py | Audio properties analysis |
| audio_preprocessing.py | Audio preprocessing functions |
| spectrograms.py | STFT and Log-Mel generation |
| data_loader.py | Reusable data loader |
| test_data_loader.py | Data loader tests |
| audio_properties.csv | Audio analysis results |

## Run the Tests

After configuring the dataset path, run:

\\\ash
python test_data_loader.py
\\\

To test the data loader with one sample:

\\\ash
python data_loader.py
\\\

## CNN Input Format

The data loader produces normalized Log-Mel spectrograms with the following shape:

\\\	ext
(128, 173, 1)
\\\

For a batch of 873 recordings:

\\\	ext
Input shape:  (873, 128, 173, 1)
Labels shape: (873,)
\\\

These features are prepared for the CNN training stage.

## Visualizations

Generated visual outputs include:

- class_distribution.png
- spectrogram_comparison.png
- stft_spectrogram.png
