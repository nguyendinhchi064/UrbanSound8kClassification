"""AST AudioSet extension: frozen transfer, cached-prefix final block fine-tuning.
Only team training and validation clips are accessed.
"""

import argparse
import json
import random
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import joblib
import librosa
from transformers import ASTFeatureExtractor, ASTForAudioClassification
from src.data_loader import load_metadata, split_dataframe, audio_file
from src.training import get_device

SOURCE = Path("data/upstream/AST")
REPO = "MIT/ast-finetuned-audioset-10-10-0.4593"
REV = "f826b80d28226b62986cc218e5cec390b1096902"
CACHE = Path("data/ast_transfer")
OUT = Path("reports/member4_ast")
CHECK = Path("checkpoints/ast_transfer")
LENGTH = 400
SEED = 42


def setup():
    from huggingface_hub import snapshot_download

    if not (SOURCE / "model.safetensors").exists():
        snapshot_download(
            REPO,
            revision=REV,
            allow_patterns=[
                "config.json",
                "preprocessor_config.json",
                "model.safetensors",
                "README.md",
            ],
            local_dir=SOURCE,
        )
    (SOURCE / "revision.txt").write_text(REV)


def load_model():
    assert (SOURCE / "revision.txt").read_text().strip() == REV
    model, info = ASTForAudioClassification.from_pretrained(
        SOURCE, local_files_only=True, use_safetensors=True, output_loading_info=True
    )
    assert not any(
        info[k]
        for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    ), info
    backbone = model.audio_spectrogram_transformer
    config = backbone.config
    # Patch projection flattens frequency first and time second. Keep the first
    # 39 temporal positions of each frequency row; do not distort frequency.
    f = (config.num_mel_bins - config.patch_size) // config.frequency_stride + 1
    old_t = (config.max_length - config.patch_size) // config.time_stride + 1
    new_t = (LENGTH - config.patch_size) // config.time_stride + 1
    pos = backbone.embeddings.position_embeddings.detach()
    grid = pos[:, 2:].reshape(1, f, old_t, config.hidden_size)
    new = torch.cat(
        (pos[:, :2], grid[:, :, :new_t, :].reshape(1, f * new_t, config.hidden_size)),
        dim=1,
    )
    backbone.embeddings.position_embeddings = nn.Parameter(new)
    config.max_length = LENGTH
    return model.eval(), f * new_t + 2


def extract():
    setup()
    CACHE.mkdir(parents=True, exist_ok=True)
    model, tokens = load_model()
    model.requires_grad_(False)
    device = get_device()
    model.to(device)
    b = model.audio_spectrogram_transformer
    feature = ASTFeatureExtractor.from_pretrained(
        SOURCE, local_files_only=True, max_length=LENGTH
    )
    meta = load_metadata()
    for split in ("train", "validation"):
        rows = split_dataframe(meta, split).reset_index(drop=True)
        assert len(rows) == (7000 if split == "train" else 816)
        prefix = CACHE / f"{split}_prefix.npy"
        emb = CACHE / f"{split}_embedding.npy"
        log = CACHE / f"{split}_logits.npy"
        marker = CACHE / f"{split}.csv"
        if all(p.exists() for p in (prefix, emb, log, marker)):
            continue
        paths = [p.with_suffix(".tmp.npy") for p in (prefix, emb, log)]
        hmap = np.lib.format.open_memmap(
            paths[0], mode="w+", dtype=np.float16, shape=(len(rows), tokens, 768)
        )
        emap = np.lib.format.open_memmap(
            paths[1], mode="w+", dtype=np.float32, shape=(len(rows), 768)
        )
        lmap = np.lib.format.open_memmap(
            paths[2], mode="w+", dtype=np.float32, shape=(len(rows), 527)
        )
        start = time.perf_counter()
        with torch.inference_mode():
            for i in range(0, len(rows), 8):
                waves = []
                for row in rows.iloc[i : i + 8].to_dict("records"):
                    a, _ = librosa.load(audio_file(row), sr=16000, mono=True)
                    # Preserve short clips; feature extractor pads in log-mel domain.
                    a = a[:64000]
                    if len(a) < 400:
                        a = np.pad(a, (0, 400 - len(a)))
                    waves.append(a)
                x = feature(
                    waves, sampling_rate=16000, return_tensors="pt"
                ).input_values.to(device)
                h = b.embeddings(x)
                for layer in b.encoder.layer[:-1]:
                    h = layer(h)
                # Round-trip cache precision during extraction too.
                half = h.half().cpu().numpy()
                h = torch.from_numpy(half).to(device=device, dtype=torch.float32)
                z = b.layernorm(b.encoder.layer[-1](h))
                z = (z[:, 0] + z[:, 1]) / 2
                logits = model.classifier(z)
                assert np.isfinite(half).all()
                hmap[i : i + len(waves)] = half
                emap[i : i + len(waves)] = z.cpu().numpy()
                lmap[i : i + len(waves)] = logits.cpu().numpy()
                if i % 320 == 0:
                    print(
                        "AST extract",
                        split,
                        i,
                        len(rows),
                        "seconds",
                        round(time.perf_counter() - start, 1),
                        flush=True,
                    )
        for a in (hmap, emap, lmap):
            a.flush()
        del hmap, emap, lmap, a
        for tmp, dst in zip(paths, (prefix, emb, log)):
            tmp.replace(dst)
        rows[["slice_file_name", "fsID", "fold", "classID"]].to_csv(marker, index=False)
        (CACHE / f"{split}_time.json").write_text(
            json.dumps(time.perf_counter() - start)
        )
    del model
    if device.type == "mps":
        torch.mps.empty_cache()


def indices():
    train, val = [pd.read_csv(CACHE / f"{s}.csv") for s in ("train", "validation")]
    for s, idx in zip(("train", "validation"), (train, val)):
        ref = pd.read_csv(Path("data/audio_transfer") / f"{s}_index.csv")
        assert idx.slice_file_name.tolist() == ref.slice_file_name.tolist()
        assert idx.classID.tolist() == ref.classID.tolist()
    return train.classID.to_numpy(), val.classID.to_numpy()


def metrics(y, p):
    return dict(
        accuracy=100 * accuracy_score(y, p),
        macro_f1=100 * f1_score(y, p, average="macro"),
        balanced_accuracy=100 * balanced_accuracy_score(y, p),
    )


def fit_frozen():
    y, v = indices()
    OUT.mkdir(parents=True, exist_ok=True)
    CHECK.mkdir(parents=True, exist_ok=True)
    rows = []
    best = (-1.0, -1.0)
    for rep in ("embedding", "logits", "combined"):
        arrays = []
        for s in ("train", "validation"):
            if rep == "combined":
                arrays.append(
                    np.concatenate(
                        [
                            np.load(CACHE / f"{s}_embedding.npy"),
                            np.load(CACHE / f"{s}_logits.npy"),
                        ],
                        axis=1,
                    )
                )
            else:
                arrays.append(np.load(CACHE / f"{s}_{rep}.npy"))
        x, z = arrays
        for c in (0.1, 1.0, 10.0):
            for method in ("logistic", "rbf"):
                if method == "logistic":
                    m = make_pipeline(
                        StandardScaler(),
                        LogisticRegression(C=c, max_iter=1500, random_state=SEED),
                    )
                else:
                    m = make_pipeline(
                        Normalizer(),
                        SVC(
                            C=c,
                            kernel="rbf",
                            gamma="scale",
                            probability=True,
                            random_state=SEED,
                        ),
                    )
                tick = time.perf_counter()
                m.fit(x, y)
                pred = m.predict(z)
                score = metrics(v, pred)
                name = f"{rep}_{method}_C{c}"
                row = dict(
                    name=name,
                    representation=rep,
                    **score,
                    fit_seconds=time.perf_counter() - tick,
                )
                rows.append(row)
                pd.DataFrame(rows).sort_values(
                    ["accuracy", "macro_f1"], ascending=False
                ).to_csv(OUT / "frozen_comparison.csv", index=False)
                print(row, flush=True)
                rank = (score["accuracy"], score["macro_f1"])
                if rank > best:
                    best = rank
                    joblib.dump(
                        dict(model=m, representation=rep, recipe=name, upstream=REV),
                        CHECK / "best_frozen.joblib",
                    )
                    # SVC probability and decision-label rules can differ; disclose in blend.
                    np.save(OUT / "frozen_probabilities.npy", m.predict_proba(z))
                    (OUT / "frozen_summary.json").write_text(json.dumps(row, indent=2))


class PrefixData(Dataset):
    def __init__(self, split, y):
        self.x = np.load(CACHE / f"{split}_prefix.npy", mmap_mode="r")
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return torch.from_numpy(np.array(self.x[i])), int(self.y[i])


class Tail(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.block = base.encoder.layer[-1]
        self.norm = base.layernorm
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(768, 10))

    def forward(self, x):
        z = self.norm(self.block(x))
        return self.head((z[:, 0] + z[:, 1]) / 2)


def fit_tail():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    y, v = indices()
    base, _ = load_model()
    model = Tail(base.audio_spectrogram_transformer)
    del base
    device = get_device()
    model.to(device)
    # First fit a small head on cached pooled features, then unfreeze final block.
    e = torch.from_numpy(np.load(CACHE / "train_embedding.npy"))
    opt = torch.optim.AdamW(model.head.parameters(), lr=0.001, weight_decay=0.001)
    for _ in range(8):
        model.head.train()
        for x, t in DataLoader(
            TensorDataset(e, torch.from_numpy(y)), batch_size=128, shuffle=True
        ):
            opt.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model.head(x.to(device)), t.to(device))
            loss.backward()
            opt.step()
    opt = torch.optim.AdamW(
        [
            dict(params=model.block.parameters(), lr=1e-5),
            dict(params=model.norm.parameters(), lr=1e-5),
            dict(params=model.head.parameters(), lr=2e-4),
        ],
        weight_decay=0.001,
    )
    epochs = 8
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    loaders = [
        DataLoader(PrefixData("train", y), batch_size=16, shuffle=True),
        DataLoader(PrefixData("validation", v), batch_size=16),
    ]
    best = (-1.0, -1.0)
    history = []
    stale = 0
    for epoch in range(1, epochs + 1):
        tick = time.perf_counter()
        model.train()
        total = 0.0
        for x, t in loaders[0]:
            x, t = x.to(device, dtype=torch.float32), t.to(device)
            opt.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(x), t, label_smoothing=0.05)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(t)
        model.eval()
        probs = []
        with torch.inference_mode():
            for x, t in loaders[1]:
                probs.append(
                    model(x.to(device, dtype=torch.float32)).softmax(1).cpu().numpy()
                )
        p = np.concatenate(probs)
        row = dict(
            epoch=epoch,
            train_loss=total / len(y),
            **metrics(v, p.argmax(1)),
            epoch_seconds=time.perf_counter() - tick,
        )
        history.append(row)
        pd.DataFrame(history).to_csv(OUT / "tail_history.csv", index=False)
        print("AST tail", row, flush=True)
        rank = (row["accuracy"], row["macro_f1"])
        if rank > best:
            best = rank
            stale = 0
            torch.save(
                dict(
                    state_dict=model.state_dict(),
                    seed=SEED,
                    revision=REV,
                    frames=LENGTH,
                    metrics=row,
                ),
                CHECK / "best_tail.tmp",
            )
            (CHECK / "best_tail.tmp").replace(CHECK / "best_tail.pth")
            np.save(OUT / "tail_probabilities.npy", p)
            (OUT / "tail_summary.json").write_text(json.dumps(row, indent=2))
        else:
            stale += 1
        sched.step()
        if stale >= 3:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", choices=["extract", "frozen", "tail", "all"], default="all"
    )
    a = parser.parse_args()
    if a.stage in ("extract", "all"):
        extract()
    if a.stage in ("frozen", "all"):
        fit_frozen()
    if a.stage in ("tail", "all"):
        fit_tail()
