# Repository guide

The project follows the pipeline from dataset preparation to trained models. Existing team-owned files retain their original paths to keep imports, notebooks and training commands compatible.

| Project area | Entry point |
|---|---|
| Dataset | [Metadata](../metadata/UrbanSound8K.csv), [dataset exploration](../notebooks/dataset_exploration.ipynb) |
| Shared audio features | [Preprocessing](../src/audio_preprocessing.py), [spectrograms](../src/spectrograms.py), [feature cache](../src/features.py), [dataset loader](../src/datasets.py) |
| Custom CNN | [Architecture](../src/model.py), [training](../scripts/train.py), [Member 3 report](../reports/member3_report.md) |
| Required transfer learning | [Architecture](../src/transfer_models.py), [training](../scripts/train_transfer.py), [executed notebook](../notebooks/member4_transfer_learning.ipynb), [Member 4 report](../reports/member4_report.md) |
| Optional audio-pretrained transfer | [EfficientAT models](../src/audio_transfer_models.py), [audio frontend](../src/audio_transfer_data.py), [notebook](../notebooks/member4_audio_transfer_extension.ipynb), [optional reproduction notes](../reports/member4_audio/README.md) |
| Selected checkpoint interface | [Prediction](../scripts/predict_transfer.py), [validation](../scripts/validate_transfer_checkpoint.py), [handoff](../reports/member4/README.md) |
| Reproducibility | [Recorded run replay](../scripts/reproduce_transfer.py), [checkpoint manifest](../reports/member4/checkpoint_manifest.csv), [notebook builder](../scripts/build_transfer_notebooks.py) |

## Submission conventions

- Reports retain training configurations, metrics and histories; raw logs, datasets and intermediate model weights stay outside Git.
- The two selected ImageNet transfer checkpoints are included alongside the original custom CNN.
- Optional audio-pretrained weights are retained locally and reproduced from recorded configurations; the extra pretraining track is separate from the required CNN comparison.
- All submitted accuracy figures are validation results. No final test result is claimed.
- Figures and notebook outputs are rendered from recorded data rather than generated illustrations.
- Member 4 uses separate dependency files, keeping the original team requirements unchanged.

Start with the [main README](../README.md), then the relevant report or executed notebook.
