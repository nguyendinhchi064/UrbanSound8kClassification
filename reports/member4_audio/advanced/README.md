# Optional research extension: AST and multi-seed ensembles

This extension belongs to Member 4 only and is **not required** for the CNN transfer-learning assignment. Keep the CNN comparison and its conclusions independent. The team split is unchanged (7,000 train / 816 validation / 837 reserved test).

## Experiments

- Eight already-trained EfficientAT checkpoints, including independent seeds, combined with fixed uniform weights.
- Inference-time shifts of ±8 spectrogram frames (80 ms), silence padding, no wrap-around. Three equal-weight predictions per model.
- Audio Spectrogram Transformer (AST): official AudioSet checkpoint `MIT/ast-finetuned-audioset-10-10-0.4593`, revision `f826b80d28226b62986cc218e5cec390b1096902`. Strict pretrained load from safetensors.
- AST features use the upstream Hugging Face extractor at 16 kHz, shortened to 400 frames; the original temporal position embeddings are cropped per frequency row. This is a project-specific efficiency adaptation, not the original 1024-frame evaluation recipe.
- Frozen representations (pooled embeddings, AudioSet logits, concatenation) with logistic and RBF classifiers fitted on train only.
- Cached-prefix final transformer block fine-tuning, with float16 cache and float32 training. Remaining backbone layers stay fixed. Eight epochs maximum.
- Equal-weight CNN/AST probability ensembles, with recording and label alignment verified. These are exploratory validation comparisons, not held-out test results.

## Reproduce

From the repository root with optional dependencies installed, the existing audio transfer cache and all eight original checkpoint files available (see the main reproduction guide and local handoff package):

```bash
python -m scripts.advanced_audio_ensemble
python -m scripts.ast_transfer --stage all
python -m scripts.final_transfer_ensemble
```

AST checkpoint and cache stay in ignored data directories. Model artifacts stay in ignored checkpoint directories. Reports record the actual scores, not an aspirational target.

## Limitations

The pretrained AudioSet corpus cannot be conclusively audited for overlap with UrbanSound8K. AST's shorter inputs and the cached activation precision alter its standard inference configuration. The Hugging Face NumPy frontend emits a zero-valued mel-filter warning at its native parameters; smoke tests verified finite inputs and outputs, but no numeric comparison with Torchaudio was possible in the installed environment. SVC probability argmax can differ from its decision-rule predictions. Repeated validation selection can produce optimistic results. Ensembles cost more to deploy than a single CNN. No fold-10 model selection or evaluation is performed. PaSST is deferred to avoid introducing another unverified preprocessing/dependency path.
