"""Read-only validation audit of saved Member 4 experiments; no test inference."""

import hashlib
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from torch.utils.data import DataLoader
from src.data_loader import CLASS_NAMES, load_metadata, split_dataframe
from src.datasets import SpectrogramDataset
from src.audio_transfer_data import AudioDataset
from src.transfer_models import TransferCNN
from src.audio_transfer_models import AudioTransferCNN
from src.training import get_device

OUT = Path("reports/member4")
device = get_device()


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def sync():
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def main():
    meta = load_metadata()
    splits = {s: split_dataframe(meta, s) for s in ("train", "validation", "test")}
    assert [len(x) for x in splits.values()] == [7000, 816, 837]
    source_sets = [set(x.fsID) for x in splits.values()]
    assert not any(
        a & b for i, a in enumerate(source_sets) for b in source_sets[i + 1 :]
    )
    for s in ("train", "validation"):
        data = AudioDataset(s)
        assert data.index.slice_file_name.tolist() == splits[s].slice_file_name.tolist()
        assert data.index.classID.tolist() == splits[s].classID.tolist()
    (OUT / "split_audit.json").write_text(
        json.dumps(
            {
                s: {
                    "clips": len(x),
                    "source_ids": x.fsID.nunique(),
                    "folds": sorted(x.fold.unique().tolist()),
                }
                for s, x in splits.items()
            },
            indent=2,
        )
    )
    results = []
    manifest = []
    benchmarks = []
    selected = {
        "resnet18_repeat_full_2024",
        "resnet18_opt_regularized",
        "mobilenet_v2_opt_mixup",
        "densenet121_partial",
        "efficientnet_b0_partial",
        "mn20_as_confirm_123_1_partial",
        "mn10_as_partial",
    }
    for report, folder, cls, ds in (
        ("member4", "transfer", TransferCNN, SpectrogramDataset("validation")),
        (
            "member4_audio",
            "audio_transfer",
            AudioTransferCNN,
            AudioDataset("validation"),
        ),
    ):
        root = Path("reports") / report
        comp = pd.read_csv(root / "model_comparison.csv")
        assert len(comp) == (21 if folder == "transfer" else 16)
        for row in comp.itertuples():
            r = root / row.name
            cfg = json.loads((r / "config.json").read_text())
            summ = json.loads((r / "summary.json").read_text())
            hist = pd.read_csv(r / "training_history.csv")
            best = hist.sort_values(
                ["val_accuracy", "val_loss"], ascending=[False, True]
            ).iloc[0]
            assert int(best.epoch) == summ["best_epoch"] and np.isclose(
                best.val_accuracy, summ["accuracy"]
            )
            assert np.isclose(best.val_macro_f1, summ["macro_f1"]) and np.isclose(
                row.accuracy, summ["accuracy"]
            )
            path = Path("checkpoints") / folder / row.name / "best.pth"
            ck = torch.load(path, map_location="cpu", weights_only=True)
            assert ck["config"] == cfg and ck["epoch"] == summ["best_epoch"]
            assert all(
                np.isclose(ck["metrics"][k], summ[k])
                for k in ("accuracy", "macro_f1", "balanced_accuracy", "loss")
            )
            manifest.append(
                dict(
                    run=row.name,
                    path=str(path),
                    bytes=path.stat().st_size,
                    sha256=sha(path),
                )
            )
            result = dict(
                run=row.name,
                history_match=True,
                checkpoint_metadata_match=True,
                validation_replayed=False,
            )
            if row.name in selected:
                model = cls(cfg["arch"], pretrained=False, dropout=cfg["dropout"])
                model.load_state_dict(ck["model_state_dict"], strict=True)
                model.configure(cfg["stage"])
                model.to(device).eval()
                ys = []
                ps = []
                with torch.inference_mode():
                    for x, y in DataLoader(ds, batch_size=32, shuffle=False):
                        ps.extend(model(x.to(device)).argmax(1).cpu().tolist())
                        ys.extend(y.tolist())
                    x = ds[0][0].unsqueeze(0).to(device)
                    for _ in range(10):
                        model(x)
                    sync()
                    start = time.perf_counter()
                    for _ in range(50):
                        model(x)
                        sync()
                    ms = (time.perf_counter() - start) * 20
                accuracy = 100 * accuracy_score(ys, ps)
                macro = 100 * f1_score(ys, ps, average="macro")
                assert np.isclose(accuracy, summ["accuracy"], atol=1e-6), (
                    row.name,
                    accuracy,
                    summ["accuracy"],
                )
                assert np.isclose(macro, summ["macro_f1"], atol=1e-6)
                result.update(
                    validation_replayed=True,
                    replayed_accuracy=accuracy,
                    replayed_macro_f1=macro,
                )
                bench = dict(
                    run=row.name,
                    architecture=cfg["arch"],
                    input_shape="x".join(map(str, x.shape[1:])),
                    parameters=sum(p.numel() for p in model.parameters()),
                    trainable_parameters=sum(
                        p.numel() for p in model.parameters() if p.requires_grad
                    ),
                    checkpoint_mib=path.stat().st_size / 2**20,
                    latency_ms=ms,
                    device=str(device),
                )
                try:
                    from thop import profile

                    macs, _ = profile(model.cpu(), inputs=(x.cpu(),), verbose=False)
                    bench["estimated_macs"] = int(macs)
                except Exception as exc:
                    bench["mac_error"] = str(exc)
                benchmarks.append(bench)
                if row.name in (
                    "resnet18_repeat_full_2024",
                    "mn20_as_confirm_123_1_partial",
                    "mobilenet_v2_opt_mixup",
                ):
                    pd.DataFrame(
                        classification_report(
                            ys,
                            ps,
                            labels=list(range(10)),
                            target_names=CLASS_NAMES,
                            output_dict=True,
                            zero_division=0,
                        )
                    ).T.to_csv(root / f"{row.name}_class_metrics.csv")
                    cm = confusion_matrix(ys, ps, labels=list(range(10)))
                    pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
                        root / f"{row.name}_confusion.csv"
                    )
                del model
                print("Validation reproduced", row.name, accuracy, flush=True)
            results.append(result)
            del ck
    pd.DataFrame(results).to_csv(OUT / "checkpoint_audit.csv", index=False)
    pd.DataFrame(manifest).to_csv(OUT / "checkpoint_manifest.csv", index=False)
    pd.DataFrame(benchmarks).to_csv(OUT / "final_benchmark.csv", index=False)
    print(
        "Audit passed:",
        len(results),
        "checkpoints;",
        sum(x["validation_replayed"] for x in results),
        "validation replays",
        flush=True,
    )


if __name__ == "__main__":
    main()
