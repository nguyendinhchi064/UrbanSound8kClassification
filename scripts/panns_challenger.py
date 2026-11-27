"""Official PANNs CNN14 frozen embeddings plus train-only classifiers."""

import json
import sys
import subprocess
import time
from pathlib import Path
import numpy as np
import pandas as pd
import librosa
import torch
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
import joblib
from src.data_loader import load_metadata, split_dataframe, audio_file, CLASS_NAMES
from src.training import get_device

ROOT = Path("data/audio_transfer/panns")
OUT = Path("reports/member4_panns")
UPSTREAM = Path("data/upstream/PANNs").resolve()
REV = "d2f4b8c18eab44737fcc0de1248ae21eb43f6aa4"
URL = "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1"


def model():
    assert (
        subprocess.check_output(
            ["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True
        ).strip()
        == REV
    )
    sys.path.insert(0, str(UPSTREAM / "pytorch"))
    from models import Cnn14

    m = Cnn14(32000, 1024, 320, 64, 50, 14000, 527)
    ck = torch.hub.load_state_dict_from_url(
        URL,
        model_dir="data/upstream/weights",
        file_name="Cnn14_mAP_0431.pth",
        map_location="cpu",
        weights_only=True,
    )
    m.load_state_dict(ck["model"], strict=True)
    return m.requires_grad_(False).eval()


def extract():
    device = get_device()
    m = model().to(device)
    ROOT.mkdir(parents=True, exist_ok=True)
    meta = load_metadata()
    for split in ("train", "validation"):
        rows = split_dataframe(meta, split).reset_index(drop=True)
        dst = ROOT / f"{split}.npy"
        marker = ROOT / f"{split}.csv"
        if dst.exists() and marker.exists():
            continue
        chunks = []
        start = time.perf_counter()
        with torch.inference_mode():
            for i in range(0, len(rows), 8):
                waves = []
                for row in rows.iloc[i : i + 8].to_dict("records"):
                    a, _ = librosa.load(audio_file(row), sr=32000, mono=True)
                    a = np.pad(a[:128000], (0, max(0, 128000 - len(a))))
                    waves.append(a)
                x = torch.from_numpy(np.stack(waves)).to(device)
                z = m(x)["embedding"].cpu().numpy()
                assert z.shape == (len(waves), 2048) and np.isfinite(z).all()
                chunks.append(z)
                if i % 320 == 0:
                    print(split, i, len(rows), flush=True)
        tmp = dst.with_suffix(".tmp.npy")
        np.save(tmp, np.concatenate(chunks))
        tmp.replace(dst)
        rows[["slice_file_name", "fsID", "fold", "classID"]].to_csv(marker, index=False)
        (ROOT / f"{split}_seconds.json").write_text(
            json.dumps(time.perf_counter() - start)
        )


def fit():
    OUT.mkdir(parents=True, exist_ok=True)
    x, v = [np.load(ROOT / f"{s}.npy") for s in ("train", "validation")]
    indexes = [pd.read_csv(ROOT / f"{s}.csv") for s in ("train", "validation")]
    for s, idx in zip(("train", "validation"), indexes):
        ref = pd.read_csv(Path("data/audio_transfer") / f"{s}_index.csv")
        assert idx.slice_file_name.tolist() == ref.slice_file_name.tolist()
        assert idx.classID.tolist() == ref.classID.tolist()
    y, z = [idx.classID.to_numpy() for idx in indexes]
    x, v = np.log1p(np.maximum(x, 0)), np.log1p(np.maximum(v, 0))
    recipes = []
    for c in (0.1, 1.0, 10.0):
        recipes.append(
            (
                f"logistic_C{c}",
                make_pipeline(
                    StandardScaler(),
                    LogisticRegression(C=c, max_iter=1500, random_state=42),
                ),
            )
        )
    for c in (1.0, 10.0, 30.0):
        recipes.append(
            (
                f"rbf_C{c}",
                make_pipeline(Normalizer(), SVC(C=c, kernel="rbf", gamma="scale")),
            )
        )
    results = []
    best = (-1.0, -1.0)
    winner = None
    for name, est in recipes:
        start = time.perf_counter()
        est.fit(x, y)
        pred = est.predict(v)
        r = dict(
            name=name,
            accuracy=100 * accuracy_score(z, pred),
            macro_f1=100 * f1_score(z, pred, average="macro"),
            balanced_accuracy=100 * balanced_accuracy_score(z, pred),
            classifier_fit_seconds=time.perf_counter() - start,
        )
        results.append(r)
        print(r, flush=True)
        score = (r["accuracy"], r["macro_f1"])
        if score > best:
            best = score
            winner = (name, est, pred)
    pd.DataFrame(results).sort_values("accuracy", ascending=False).to_csv(
        OUT / "classifier_comparison.csv", index=False
    )
    name, est, pred = winner
    target = Path("checkpoints/audio_transfer/panns")
    target.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": est,
            "recipe": name,
            "input_transform": "log1p(max(embedding,0))",
            "upstream_rev": REV,
        },
        target / "classifier.joblib",
    )
    pd.DataFrame(
        classification_report(
            z,
            pred,
            target_names=CLASS_NAMES,
            labels=list(range(10)),
            zero_division=0,
            output_dict=True,
        )
    ).T.to_csv(OUT / "class_metrics.csv")
    pd.DataFrame(
        confusion_matrix(z, pred, labels=list(range(10))),
        index=CLASS_NAMES,
        columns=CLASS_NAMES,
    ).to_csv(OUT / "confusion.csv")
    (OUT / "summary.json").write_text(
        json.dumps(
            dict(
                best_recipe=name,
                accuracy=best[0],
                macro_f1=best[1],
                backbone_parameters=81837071,
                pretraining="AudioSet",
                split="validation",
                upstream=REV,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    extract()
    fit()
