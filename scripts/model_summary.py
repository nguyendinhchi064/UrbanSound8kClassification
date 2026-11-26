import argparse
from pathlib import Path

import torch
from torch import nn

from src.model import INPUT_SHAPE_CHW, NUM_CLASSES, UrbanSoundCNN, model_summary
from src.paths import PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-blocks", type=int, default=4)
    parser.add_argument("--base-filters", type=int, default=32)
    parser.add_argument("--convs-per-block", type=int, default=2)
    parser.add_argument("--conv-dropout", type=float, default=0.1)
    parser.add_argument("--fc-hidden", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--real-batch", action="store_true", help="also run one training step on cached spectrograms")
    parser.add_argument("--out", default=PROJECT_ROOT / "reports" / "custom_cnn_summary.txt")
    args = parser.parse_args()

    model = UrbanSoundCNN(
        n_blocks=args.n_blocks, base_filters=args.base_filters, convs_per_block=args.convs_per_block,
        conv_dropout=args.conv_dropout, fc_hidden=args.fc_hidden, dropout=args.dropout,
    )
    summary = model_summary(model)
    print(summary)

    batch = torch.randn(8, *INPUT_SHAPE_CHW)
    assert model(batch).shape == (8, NUM_CLASSES)
    print(f"\nRandom batch {tuple(batch.shape)} -> logits {(8, NUM_CLASSES)}: OK")

    if args.real_batch:
        from torch.utils.data import DataLoader
        from src.datasets import SpectrogramDataset

        x, y = next(iter(DataLoader(SpectrogramDataset("train"), batch_size=16, shuffle=True)))
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        model.train()
        loss = nn.CrossEntropyLoss()(model(x), y)
        loss.backward()
        optimizer.step()
        print(f"Real batch {tuple(x.shape)}, labels {tuple(y.shape)}: loss {loss.item():.3f} "
              f"(about ln(10) = 2.303 expected before training), one optimizer step OK")

    out = Path(args.out)
    out = out if out.is_absolute() else PROJECT_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(f"UrbanSoundCNN({model.config})\n\n{summary}\n", encoding="utf-8")
    print(f"Summary saved to {out}")


if __name__ == "__main__":
    main()