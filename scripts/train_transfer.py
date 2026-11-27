"""Train transfer CNNs using the team's fixed holdout; never evaluate test."""

import argparse
import json
import random
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from src.datasets import SpectrogramDataset
from src.transfer_models import TransferCNN, ARCHS
from src.training import get_device
from src.audio_transfer_models import AudioTransferCNN, VARIANTS


def augment(x):
    x = x.clone()
    for axis, width in ((2, 12), (3, 18)):
        for i in range(len(x)):
            w = random.randint(0, width)
            start = random.randint(0, x.shape[axis] - w)
            if axis == 2:
                x[i, :, start : start + w, :] = 0
            else:
                x[i, :, :, start : start + w] = 0
    return x


def epoch(
    model, loader, device, optimizer=None, specaugment=False, mixup=0.0, smoothing=0.0
):
    training = optimizer is not None
    model.train(training)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=smoothing if training else 0.0)
    total_loss, ys, ps = 0.0, [], []
    with torch.set_grad_enabled(training):
        for x, y in loader:
            if training and specaugment:
                x = augment(x)
            x, y = x.to(device), y.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            if training and mixup > 0:
                lam = float(np.random.beta(mixup, mixup))
                perm = torch.randperm(len(x)).to(device)
                out = model(lam * x + (1 - lam) * x[perm])
                loss = lam * loss_fn(out, y) + (1 - lam) * loss_fn(out, y[perm])
            else:
                out = model(x)
                loss = loss_fn(out, y)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
            ys.extend(y.detach().cpu().tolist())
            ps.extend(out.detach().argmax(1).cpu().tolist())
    return dict(
        loss=total_loss / len(ys),
        accuracy=100 * accuracy_score(ys, ps),
        macro_f1=100
        * f1_score(ys, ps, labels=list(range(10)), average="macro", zero_division=0),
        balanced_accuracy=100 * balanced_accuracy_score(ys, ps),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=(*ARCHS, *VARIANTS), required=True)
    parser.add_argument("--stage", choices=["frozen", "partial", "full"], required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--head-lr", type=float, default=0.001)
    parser.add_argument("--backbone-lr", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--specaugment", action="store_true")
    parser.add_argument("--mixup", type=float, default=0.0)
    parser.add_argument("--smoothing", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--from-checkpoint", type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument(
        "--report-root",
        type=Path,
        help="Override output directory for independent replay",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        help="Override checkpoint directory for independent replay",
    )
    args = parser.parse_args()
    if Path(args.name).name != args.name or args.name in ("", ".", ".."):
        parser.error("Name must be a single safe directory name")
    if (
        min(args.epochs, args.batch_size, args.patience) <= 0
        or min(args.head_lr, args.backbone_lr) <= 0
    ):
        parser.error("Epochs, batch size, patience and learning rates must be positive")
    if args.mixup < 0 or not 0 <= args.smoothing < 1:
        parser.error("Invalid augmentation or smoothing")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = get_device()
    audio = args.arch in VARIANTS
    if audio:
        from src.audio_transfer_data import AudioDataset

        train, val = AudioDataset("train"), AudioDataset("validation")
    else:
        train, val = SpectrogramDataset("train"), SpectrogramDataset("validation")
    assert len(train) == 7000 and len(val) == 816, "Unexpected split"
    loaders = [
        DataLoader(
            train,
            batch_size=args.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(args.seed),
        ),
        DataLoader(val, batch_size=args.batch_size),
    ]
    model = (AudioTransferCNN if audio else TransferCNN)(
        args.arch, pretrained=args.from_checkpoint is None, dropout=args.dropout
    )
    if args.from_checkpoint:
        ckpt = torch.load(args.from_checkpoint, map_location="cpu", weights_only=True)
        assert ckpt["config"]["arch"] == args.arch
        model.load_state_dict(ckpt["model_state_dict"])
    model.configure(args.stage)
    model.to(device)
    groups = [{"params": model.head.parameters(), "lr": args.head_lr}]
    backbone = [p for p in model.backbone.parameters() if p.requires_grad]
    if backbone:
        groups.append({"params": backbone, "lr": args.backbone_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    out = (
        args.report_root
        or Path("reports/member4_audio" if audio else "reports/member4")
    ) / args.name
    ckptdir = (
        args.checkpoint_root
        or Path("checkpoints/audio_transfer" if audio else "checkpoints/transfer")
    ) / args.name
    out.mkdir(parents=True, exist_ok=False)
    ckptdir.mkdir(parents=True, exist_ok=True)
    cfg = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    cfg.update(
        device=str(device),
        torch_version=str(torch.__version__),
        total_parameters=sum(p.numel() for p in model.parameters()),
        trainable_parameters=sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
    )
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    print(cfg, flush=True)
    history = []
    best = (-1.0, float("-inf"))
    stale = 0
    start = time.perf_counter()
    for e in range(1, args.epochs + 1):
        tick = time.perf_counter()
        tr = epoch(
            model,
            loaders[0],
            device,
            optimizer,
            args.specaugment,
            args.mixup,
            args.smoothing,
        )
        va = epoch(model, loaders[1], device)
        row = dict(
            epoch=e,
            **{"train_" + k: v for k, v in tr.items()},
            **{"val_" + k: v for k, v in va.items()},
            epoch_seconds=time.perf_counter() - tick,
        )
        history.append(row)
        pd.DataFrame(history).to_csv(out / "training_history.csv", index=False)
        score = (va["accuracy"], -va["loss"])
        if score > best:
            best = score
            stale = 0
            temp = ckptdir / "best.tmp"
            torch.save(
                dict(
                    model_state_dict=model.state_dict(), config=cfg, epoch=e, metrics=va
                ),
                temp,
            )
            temp.replace(ckptdir / "best.pth")
        else:
            stale += 1
        scheduler.step()
        print(json.dumps(row), flush=True)
        if stale >= args.patience:
            break
    chosen = torch.load(ckptdir / "best.pth", map_location="cpu", weights_only=True)
    (out / "summary.json").write_text(
        json.dumps(
            dict(
                **chosen["metrics"],
                best_epoch=chosen["epoch"],
                training_seconds=time.perf_counter() - start,
                **cfg,
            ),
            indent=2,
        )
    )
    print("Completed", args.name, flush=True)


if __name__ == "__main__":
    main()
