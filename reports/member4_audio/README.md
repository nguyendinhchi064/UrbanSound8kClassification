# Audio-pretrained extension — Member 4
## Objective and fair comparison
The core assignment remains the ImageNet-only CNN study in reports/member4. This extension adds external AudioSet supervision and may achieve stronger performance, but it is a separate comparison. Retain the team's exact 7,000/816/837 fixed split and never evaluate fold 10 while developing. The upstream AudioSet recordings are YouTube videos; UrbanSound8K is derived from Freesound. Shared or reuploaded audio cannot be ruled out without the original audio fingerprints and complete pretraining corpus, so no claim of verified absence of external pretraining overlap is made. Neither selected checkpoint is explicitly advertised as trained on UrbanSound8K. Do not equate this holdout with published official 10-fold CV scores.
## Sources and implementation
- EfficientAT: https://github.com/fschmid56/EfficientAT (MIT, revision a425fdce92572e602a1d5634799bd9f1f2efa806), mn10_as, mn20_as, mn40_as from official release v0.0.1. Audio frontend reproduces the upstream evaluation parameters (32 kHz, preemphasis, 1024 FFT, 800 Hann window, hop 320, Kaldi mel filters, log normalization). Four-second zero padding is a project-specific choice. Torchaudio 2.15 was unavailable, so the frontend is independently implemented using torch; no numerical equivalence to a live torchaudio reference is claimed.
- PANNs: https://github.com/qiuqiangkong/audioset_tagging_cnn (MIT, revision d2f4b8c18eab44737fcc0de1248ae21eb43f6aa4), official Cnn14 mAP=.431 checkpoint from Zenodo record 3987831. Frozen 2048-dimensional embeddings, training-only classifier fits. Frozen PANNs is a separate challenger, not a fine-tuned PANNs result.
## Reproduction (from project root)

Install the core transfer requirements. PANNs and AST additionally require the optional transfer requirements.

```bash
uv pip install -r requirements-transfer-optional.txt
bash scripts/setup_audio_transfer.sh
python -m scripts.reproduce_transfer mn20_as_confirm_123_1_partial
python -m scripts.reproduce_transfer all-audio
# Optional independent PANNs challenger, using pinned upstream code.
git clone https://github.com/qiuqiangkong/audioset_tagging_cnn.git data/upstream/PANNs
git -C data/upstream/PANNs checkout d2f4b8c18eab44737fcc0de1248ae21eb43f6aa4
python -m scripts.panns_challenger
```

Replays generate independent outputs in runs/member4/replay/. The original sweep and repetition scripts require the original historical checkpoint tree. To evaluate the exact optional selected model or recreate the original ensembles, restore the supplied local checkpoint bundle at the project-relative paths (or regenerate all required configurations and use their new checkpoints, acknowledging possible numerical differences). See the [main reproduction guide](../member4/README.md).

All downloads are local. Raw data and upstream repos stay ignored by Git. Checkpoints remain local pending approval. Each validation-selected run has a configuration, history, best checkpoint and summary. Repetition replays the full checkpoint lineage for each seed. Ensembles use fixed equal audio-model weights; mixed-pretraining uses 75% audio and 25% ImageNet, with verified sample alignment. Multiple trials on one validation fold induce selection optimism. 95% is an objective, not guaranteed. MPS determinism is not guaranteed across hardware and library versions.
