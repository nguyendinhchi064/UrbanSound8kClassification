# UrbanSound8K Environmental Sound Classification

Environmental sound classification of ten urban sound categories using log-Mel spectrograms, a custom convolutional neural network and pretrained models. The repository contains the dataset preparation pipeline, executed analysis notebooks, model implementations, recorded experiments and selected checkpoints.

**Project highlights** · 8,732 audio clips · 10 classes · source-disjoint fixed holdout · PyTorch · CPU / CUDA / Apple MPS

## Project workflow

```mermaid
flowchart LR
    A[UrbanSound8K audio] --> B[Preprocessing and log-Mel features]
    B --> C[Custom CNN]
    B --> D[ImageNet CNN transfer learning]
    A --> E[Optional AudioSet-pretrained models]
    C --> F[Validation comparison]
    D --> F
    E --> F
    F --> G[Selected checkpoints]
```

## Results at a glance

| Track | Selected model | Validation accuracy |
|---|---|---:|
| Custom CNN | Tuned CNN | 86.89% |
| ImageNet transfer | ResNet18 | 87.50% |
| ImageNet efficiency alternative | MobileNetV2 | 86.40% |
| Optional AudioSet transfer | EfficientAT MN20 | 90.20% |
| Optional ensemble | EfficientAT + AST | 90.81% |

All results are from validation fold 9. Fold 10 is reserved for final evaluation. AudioSet results use additional pretraining information and are reported separately.

## Project deliverables

| Stage | Notebook / report |
|---|---|
| Dataset and audio exploration | [Dataset notebook](notebooks/dataset_exploration.ipynb) |
| Spectrogram analysis and custom CNN | [Spectrogram notebook](notebooks/spectrogram_analysis.ipynb) |
| CNN training and hyperparameter optimization | [Member 3 report](reports/member3_report.md) |
| Transfer learning | [Member 4 notebook](notebooks/member4_transfer_learning.ipynb) · [Report](reports/member4_report.md) |
| Optional audio-pretrained extension | [Audio transfer notebook](notebooks/member4_audio_transfer_extension.ipynb) |
| Reproduction and model handoff | [Member 4 guide](reports/member4/README.md) · [Repository map](docs/REPOSITORY_GUIDE.md) |


## Dataset

[UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html) contains 8,732 recordings from ten sound classes, organized into ten predefined folds.

![Class distribution](figures/class_distribution.png)

## Setup

Use Python 3.12 and run commands from the repository root.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m scripts.download_dataset
python -m scripts.prepare_features
```

`prepare_features` computes the Log-Mel spectrogram of every clip once (about 0.7 GB in `data/features/`, ignored by Git), so analysis and training do not decode audio every epoch. `pip install` installs the available PyTorch build for your platform. For NVIDIA CUDA, follow the platform-specific instructions from [pytorch.org](https://pytorch.org/get-started/locally/).

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`.

The download script saves the dataset location to `.dataset_path` (ignored by Git), and the scripts find it automatically. If the dataset is already in the kagglehub cache, it is found without downloading again. To use a copy stored elsewhere, set `URBANSOUND8K_PATH` (bash: `export URBANSOUND8K_PATH=/path/to/dataset`, PowerShell: `$env:URBANSOUND8K_PATH = "C:\path\to\dataset"`).

Both the Kaggle layout (CSV and fold directories at the root) and the original dataset layout (metadata/ and audio/) are supported.

## Audio representation

Recordings are resampled to 22,050 Hz, converted to mono, padded or cropped to four seconds and peak-normalized. STFT uses 2,048 FFT points and a 512-sample hop. Log-Mel features use 128 bands and are scaled to [0, 1]. Silent recordings map to zero-valued features.

![Spectrogram comparison](figures/spectrogram_comparison.png)

## Model inputs

```python
from src.data_loader import load_split

X_train, y_train = load_split("train")
X_val, y_val = load_split("validation")
```

For PyTorch, use the cached features instead:

```python
from src.datasets import make_dataloaders

loaders = make_dataloaders(batch_size=32)
x, y = next(iter(loaders["train"]))  # x: (32, 1, 128, 173), y: (32,)
```

Each `load_split` sample has shape (128, 173, 1), with integer labels 0–9. For channels-first models, transpose batches with X.transpose(0, 3, 1, 2). The iterator iter_samples also supports loading individual examples without retaining every spectrogram in memory.

Training uses folds 1–8, validation fold 9 and final testing fold 10. One source recording (fsID 180937) occurs in both training and validation folds. The split loader excludes its 79 training clips, retaining the original validation and test folds. The resulting partitions contain 7,000 training, 816 validation and 837 test clips; all 8,732 clips remain available for dataset analysis. Use load_split for model experiments, rather than the unfiltered fold loader. Use the same split for each model and reserve test results for final evaluation. This is a fixed holdout protocol, rather than ten-fold cross-validation.

## Custom CNN

`notebooks/spectrogram_analysis.ipynb` compares STFT and Log-Mel spectrograms and selects **Log-Mel**: it has 8 times fewer values per clip (128 × 173 instead of 1,025 × 173) and keeps fine frequency resolution at low frequencies, where much of the energy of urban sounds lies. The network takes the full 4 s spectrogram, **(1, 128, 173)**: cropping to 128 frames would drop about 24% of the real audio frames, and resizing would distort sound durations.

`src/model.py` defines `UrbanSoundCNN`:

| Stage | Layers | Output shape |
|---|---|---|
| Input | Log-Mel spectrogram | 1 × 128 × 173 |
| Block 1 | 2 × (Conv 3×3, BatchNorm, ReLU), MaxPool 2×2, Dropout2d 0.1 | 32 × 64 × 86 |
| Block 2 | same, 64 filters | 64 × 32 × 43 |
| Block 3 | same, 128 filters | 128 × 16 × 21 |
| Block 4 | same, 256 filters | 256 × 8 × 10 |
| Head | Global average pooling, Dropout 0.3, FC 256→128, ReLU, Dropout 0.3, FC 128→10 | 10 logits |

The default model has 1,206,826 parameters (4.6 MB) and needs about 1.1 G multiply-accumulates per spectrogram. It returns logits for `nn.CrossEntropyLoss`. The number of blocks, filters, convolutions per block, dropout rates and hidden units are constructor arguments for the hyperparameter experiments. Global average pooling keeps the classifier independent of the input width, so random time crops can be used for augmentation without changing the model.

```bash
python -m scripts.model_summary --real-batch   # prints the layer table and saves reports/custom_cnn_summary.txt
```

## Transfer learning — Member 4

The required ImageNet-pretrained CNN study uses the same fixed holdout and full 128 × 173 log-Mel representation as the custom CNN. ResNet18, MobileNetV2, EfficientNet-B0 and DenseNet121 were compared using frozen feature extraction and selective/full fine-tuning.

| Model | Validation accuracy | Role |
|---|---:|---|
| Custom CNN (Member 3) | 86.89% | Reference |
| ResNet18 | 87.50% | Best observed ImageNet transfer checkpoint |
| MobileNetV2 | 86.40% | Lower-MAC, smaller-checkpoint alternative |
| EfficientAT MN20 | 90.20% | Optional AudioSet-pretrained single CNN |
| EfficientAT + AST | 90.81% | Optional exploratory ensemble |

ImageNet ResNet18 averaged 86.68% over three seeds; its best run does not establish a robust improvement over the custom CNN. AudioSet results use extra pretraining and different preprocessing, so they are reported separately. Fold 10 remains reserved for the team's final evaluation.

- [Member 4 report](reports/member4_report.md): methodology, comparisons, limitations and handoff.
- [Executed ImageNet notebook](notebooks/member4_transfer_learning.ipynb) · [Executed optional extension notebook](notebooks/member4_audio_transfer_extension.ipynb).
- [Reproduction and checkpoint guide](reports/member4/README.md): setup, exact replay and validation.
- [ImageNet experiments](reports/member4/model_comparison.csv) · [Audio CNN experiments](reports/member4_audio/model_comparison.csv).

```bash
uv pip install -r requirements-transfer.txt
python -m scripts.validate_transfer_checkpoint --checkpoint checkpoints/best_transfer_resnet18.pth
```

## Repository layout

| Directory | Contents |
|---|---|
| report/ | Result of the different models |
| notebooks/ | Executed dataset, spectrogram and transfer-learning notebooks |
| src/ | Shared preprocessing, feature cache, dataset loaders and model implementations |
| scripts/ | Dataset preparation, training, analysis, inference and experiment reproduction |
| reports/ | Member reports, metrics, experiment configurations and histories |
| metadata/ | Metadata and audio properties |
| figures/ | Data visualizations, training curves and model comparisons |
| checkpoints/ | Selected custom and transfer CNN model weights |
| tests/ | Dataset, feature and model tests |
| docs/ | Repository navigation and handoff |

## Reproduce the analysis

```bash
python -m pytest -q
python -m scripts.explore_dataset
python -m scripts.analyze_audio
python -m scripts.inspect_audio
python -m scripts.validate_dataset
jupyter nbconvert --to notebook --execute --inplace notebooks/dataset_exploration.ipynb
python -m scripts.prepare_features
python -m scripts.model_summary --real-batch
jupyter nbconvert --to notebook --execute --inplace notebooks/spectrogram_analysis.ipynb
```