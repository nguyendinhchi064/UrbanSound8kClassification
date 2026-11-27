# Member 4 — reproduction and handoff

The [submission report](../member4_report.md) is the authoritative interpretation of the results. All recorded metrics use validation fold 9, not the held-out test fold. The original experiment directories contain the historical configs, histories and summaries; do not overwrite these during reproduction.

## Required ImageNet CNN track

From the repository root, use Python 3.12:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements-transfer.txt
python -m scripts.download_dataset
python -m scripts.prepare_features
python -m pytest -q
```

If using pip instead of uv, create a Python 3.12 virtual environment and use `pip install -r requirements-transfer.txt`. The additional requirements are separate so Member 1/3 installations are unaffected. Tested on macOS Apple Silicon, Python 3.12.14, PyTorch 2.15.0, torchvision 0.29.2, MPS. CPU/CUDA are selected by the team's existing device function; performance and exact training results may differ.

### Ready-to-use selected checkpoints

| Checkpoint | Run | Val. accuracy | Purpose |
|---|---|---:|---|
| `checkpoints/best_transfer_resnet18.pth` | `resnet18_repeat_full_2024` | 87.50% | Required comparison winner |
| `checkpoints/best_transfer_mobilenet_v2.pth` | `mobilenet_v2_opt_mixup` | 86.40% | Compute/size alternative |

Both contain the model state, configuration, selected epoch and validation metrics. SHA-256 digests are recorded in [submission_checkpoint_manifest.json](submission_checkpoint_manifest.json). The checkpoint files are exact copies of the original selected artifacts; no compression or quantization changes were applied.

```bash
python -m scripts.validate_transfer_checkpoint --checkpoint checkpoints/best_transfer_resnet18.pth
python -m scripts.validate_transfer_checkpoint --checkpoint checkpoints/best_transfer_mobilenet_v2.pth
```

The validation command is deliberately restricted to fold 9. To run single-clip prediction on a user-supplied WAV:

```bash
python -m scripts.predict_transfer --checkpoint checkpoints/best_transfer_resnet18.pth --audio /path/to/recording.wav
```

The final evaluation owner can integrate the model through:

```python
from scripts.predict_transfer import load_transfer_checkpoint
from src.training import get_device

device = get_device()
model, config = load_transfer_checkpoint(
    "checkpoints/best_transfer_resnet18.pth", device
)
# model.eval() is already set; input is the team's (B, 1, 128, 173) tensor.
# The responsible evaluation member controls when the untouched fold 10 is used.
```

### Reproduce recorded configurations

```bash
python -m scripts.reproduce_transfer resnet18_repeat_full_2024 --dry-run
python -m scripts.reproduce_transfer resnet18_repeat_full_2024 mobilenet_v2_opt_mixup
python -m scripts.reproduce_transfer all-image
```

The replay tool reads the exact configuration and reconstructs its frozen-to-fine-tuned checkpoint lineage. It writes new runs under ignored `runs/member4/replay/`, not the original report directories. Use `--root runs/member4/another_run` for a new independent repetition. A dry run prints the commands; incomplete replay folders are preserved. MPS and software version differences may affect numerical reproducibility.

For a new experiment, use `python -m scripts.train_transfer --help` and choose a unique name. The historical queue shell scripts are for the original experiment workspace, not a fresh clone containing published summaries but no intermediate weights. Use the replay tool instead.

### Results and experimental evidence

- `model_comparison.csv`: all 21 ImageNet runs.
- `<run>/config.json`, `training_history.csv`, `summary.json`: exact evidence.
- `seed_comparison.csv`, `seed_statistics.csv`: three-seed recipe.
- `split_audit.json`, `checkpoint_audit.csv`, `checkpoint_manifest.csv`: verified original split and checkpoint records.
- `final_benchmark.csv`, `baseline_reference.json`: measured selected-model metrics, including Member 3's existing checkpoint.
- `python -m scripts.report_transfer`: regenerate the two summary plots from the committed evidence.
- `python -m scripts.audit_transfer`: full historical audit, requires all 37 original local CNN checkpoint files and both cached datasets. Fresh-clone users should use the selected-checkpoint validation command unless supplied the full original archive.

## Executed notebooks

- [ImageNet CNN comparison](../../notebooks/member4_transfer_learning.ipynb)
- [Audio-pretrained extension](../../notebooks/member4_audio_transfer_extension.ipynb)

Both notebooks display recorded results, tables and plots. They can be regenerated with `python -m scripts.build_transfer_notebooks` from the repository root.

## Optional extensions

Install `requirements-transfer-optional.txt` only if running PANNs or AST; EfficientAT can use the required transfer environment. See the [audio-specific guide](../member4_audio/README.md) and [AST/ensemble notes](../member4_audio/advanced/README.md). AudioSet results use additional pretraining information and must remain distinct from the required ImageNet-only comparison.

Large optional model weights and all intermediate checkpoints are retained on the development machine, not committed. The local handoff package at `submission_local/member4_checkpoints/` contains the two selected core checkpoints, eight EfficientAT ensemble members (including the best MN20 and MN10 single models), PANNs classifier and AST frozen/tail artifacts, with SHA-256 manifest. Restore the package's project-relative `checkpoints/` paths only as needed and verify the checksums; use only trusted local model files. External pretrained backbones are fetched from the documented pinned sources. The full historical checkpoint trees remain under ignored `checkpoints/transfer/`, `checkpoints/audio_transfer/` and `checkpoints/ast_transfer/`.

No final test evaluation or GitHub publication is performed by these commands.
