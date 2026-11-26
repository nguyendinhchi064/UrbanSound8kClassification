
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
FIGURES = ROOT / "figures"

FIGURES.mkdir(parents=True, exist_ok=True)

runs = [
    "baseline",
    "lr_low",
    "lr_high",
    "batch_16",
    "batch_64",
    "blocks_3",
    "filters_16",
    "convs_1",
    "dropout_05",
    "adamw",
    "weight_decay",
    "early_stopping"
]

results = []

for run in runs:
    df = pd.read_csv(
        REPORTS / run / "training_history.csv"
    )

    best = df.loc[df["val_accuracy"].idxmax()]

    results.append({
        "experiment": run,
        "best_epoch": int(best["epoch"]),
        "best_val_accuracy": best["val_accuracy"],
        "train_accuracy": best["train_accuracy"],
        "train_loss": best["train_loss"],
        "val_loss": best["val_loss"]
    })

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "best_val_accuracy",
    ascending=False
)

results_df.to_csv(
    REPORTS / "experiments.csv",
    index=False
)

best_run = results_df.iloc[0]["experiment"]

df = pd.read_csv(
    REPORTS / best_run / "training_history.csv"
)

# Graph 1: Accuracy

plt.figure(figsize=(10, 6))

plt.plot(
    df["epoch"],
    df["train_accuracy"],
    label="Training Accuracy"
)

plt.plot(
    df["epoch"],
    df["val_accuracy"],
    label="Validation Accuracy"
)

plt.axvline(
    df.loc[df["val_accuracy"].idxmax(), "epoch"],
    linestyle="--",
    label="Best Epoch"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("CNN Training and Validation Accuracy")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURES / "accuracy.png", dpi=300)
plt.close()

# Graph 2: Loss

plt.figure(figsize=(10, 6))

plt.plot(
    df["epoch"],
    df["train_loss"],
    label="Training Loss"
)

plt.plot(
    df["epoch"],
    df["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.title("CNN Training and Validation Loss")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURES / "loss.png", dpi=300)
plt.close()

# Graph 3: Hyperparameter comparison

plt.figure(figsize=(12, 7))

plt.barh(
    results_df["experiment"],
    results_df["best_val_accuracy"]
)

plt.gca().invert_yaxis()
plt.xlabel("Best Validation Accuracy (%)")
plt.title("Hyperparameter Experiment Comparison")
plt.xlim(0, 100)

for i, value in enumerate(
    results_df["best_val_accuracy"]
):
    plt.text(value + 0.3, i, f"{value:.2f}%")

plt.tight_layout()
plt.savefig(FIGURES / "experiments.png", dpi=300)
plt.close()

print(results_df.to_string(index=False))
print("\nBest experiment:", best_run)
print("All graphs generated successfully!")
