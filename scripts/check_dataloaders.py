
import torch
from src.datasets import make_dataloaders


def main():

    # 1. Configure batch size
    BATCH_SIZE = 32

    # 2. Create DataLoaders
    loaders = make_dataloaders(
        batch_size=BATCH_SIZE,
        num_workers=0
    )

    train_loader = loaders["train"]
    val_loader = loaders["validation"]
    test_loader = loaders["test"]

    # 3. Display dataset sizes
    print("\n===== DATASET SPLIT =====")

    print("Training samples:", len(train_loader.dataset))
    print("Validation samples:", len(val_loader.dataset))
    print("Testing samples:", len(test_loader.dataset))

    # 4. Display number of batches
    print("\n===== BATCH INFORMATION =====")

    print("Batch size:", BATCH_SIZE)
    print("Training batches:", len(train_loader))
    print("Validation batches:", len(val_loader))
    print("Testing batches:", len(test_loader))

    # 5. Load one training batch
    inputs, labels = next(iter(train_loader))

    print("\n===== INPUT VERIFICATION =====")

    print("Input shape:", inputs.shape)
    print("Label shape:", labels.shape)

    print("Input data type:", inputs.dtype)
    print("Label data type:", labels.dtype)

    print("Minimum value:", inputs.min().item())
    print("Maximum value:", inputs.max().item())

    # 6. Verify input and label properties
    assert inputs.shape == (32, 1, 128, 173)
    assert labels.shape == (32,)
    assert inputs.dtype == torch.float32
    assert labels.dtype == torch.int64

    assert torch.isfinite(inputs).all()
    assert inputs.min() >= 0
    assert inputs.max() <= 1

    assert labels.min() >= 0
    assert labels.max() <= 9

    # 7. Verify validation batch
    val_inputs, val_labels = next(iter(val_loader))

    assert val_inputs.shape == (32, 1, 128, 173)
    assert val_labels.shape == (32,)
    assert val_inputs.dtype == torch.float32
    assert val_labels.dtype == torch.int64
    assert torch.isfinite(val_inputs).all()

    # 8. Verify dataset sizes
    assert len(train_loader.dataset) == 7000
    assert len(val_loader.dataset) == 816
    assert len(test_loader.dataset) == 837

    print("\n===== VERIFICATION RESULT =====")
    print("All DataLoader checks passed successfully!")


if __name__ == "__main__":
    main()
