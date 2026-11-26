
import argparse
from html import parser
import random
import time
from pathlib import Path
from xml.parsers.expat import model

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.datasets import make_dataloaders
from src.model import UrbanSoundCNN
from src.training import get_device, train_one_epoch, validate


def main():

    # 1. Training arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", default="baseline")
    parser.add_argument("--n-blocks", type=int, default=4)
    parser.add_argument("--base-filters", type=int, default=32)
    parser.add_argument("--convs-per-block", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--conv-dropout", type=float, default=0.1)
    parser.add_argument(
        "--optimizer",
        choices=["adam", "adamw"],
        default="adam"
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0
    )
    parser.add_argument("--patience", type=int, default=0)



    args = parser.parse_args()

    # 2. Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 3. Select GPU or CPU
    device = get_device()

    print("\n===== TRAINING CONFIGURATION =====")
    print("Device:", device)
    print("Epochs:", args.epochs)
    print("Batch size:", args.batch_size)
    print("Learning rate:", args.lr)
    print("Random seed:", args.seed)

    # 4. Create DataLoaders
    loaders = make_dataloaders(
        batch_size=args.batch_size,
        num_workers=0
    )

    train_loader = loaders["train"]
    val_loader = loaders["validation"]

    # 5. Initialize CNN
    model = UrbanSoundCNN(
    num_classes=10,
    in_channels=1,
    n_blocks=args.n_blocks,
    base_filters=args.base_filters,
    convs_per_block=args.convs_per_block,
    dropout=args.dropout,
    conv_dropout=args.conv_dropout).to(device)

    # 6. Define loss function
    criterion = nn.CrossEntropyLoss()

    # 7. Define optimizer
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    else:

        optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    print("Actual optimizer:", type(optimizer).__name__)
    print("Actual weight decay:", optimizer.param_groups[0]["weight_decay"])
    print("Actual model configuration:", model.config)
    # 8. Create output directories
    checkpoint_dir = Path("checkpoints") / args.run_name
    report_dir = Path("reports") / args.run_name

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    # 9. Initialize training history
    history = []
    best_val_accuracy = -1.0
    epochs_without_improvement = 0

    start_time = time.time()

    # 10. Training loop
    for epoch in range(1, args.epochs + 1):

        epoch_start = time.time()

        print(f"\n===== EPOCH {epoch}/{args.epochs} =====")

        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        val_loss, val_accuracy = validate(
            model,
            val_loader,
            criterion,
            device
        )

        epoch_time = time.time() - epoch_start

        # 11. Store epoch results
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "epoch_time": epoch_time
        })

        # 12. Display epoch results
        print(f"Training Loss: {train_loss:.4f}")
        print(f"Training Accuracy: {train_accuracy:.2f}%")

        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation Accuracy: {val_accuracy:.2f}%")

        print(f"Epoch Time: {epoch_time:.2f} seconds")

        # 13. Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            epochs_without_improvement = 0

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "model_config": model.config,
                "training_config": vars(args),
                "val_accuracy": val_accuracy,
                "val_loss": val_loss
            }, checkpoint_dir / "best_cnn.pth")

            print("Best model saved!")

        else:
            epochs_without_improvement += 1

        # 14. Save history after every epoch
        pd.DataFrame(history).to_csv(
            report_dir / "training_history.csv",
            index=False
        )
        if args.patience > 0:
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

    # 15. Final training summary
    total_time = time.time() - start_time

    print("\n===== TRAINING COMPLETED =====")

    print(f"Best Validation Accuracy: {best_val_accuracy:.2f}%")
    print(f"Total Training Time: {total_time / 60:.2f} minutes")

    print("Model saved:", checkpoint_dir / "best_cnn.pth")
    print("History saved:", report_dir / "training_history.csv")


if __name__ == "__main__":
    main()
