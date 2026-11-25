from src.data_loader import get_dataset_path, load_metadata

import pandas as pd
import matplotlib.pyplot as plt

# Read the metadata
dataset_path = get_dataset_path()
print("Dataset folder:", dataset_path)
df = load_metadata(dataset_path)

# Display basic information
print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

# Count recordings for each sound class
class_counts = df["class"].value_counts()

print("\nNumber of recordings per class:")
print(class_counts)

# Create a bar chart
class_counts.plot(kind="bar", figsize=(10, 5))

plt.title("UrbanSound8K - Number of Recordings per Sound Class")
plt.xlabel("Sound Class")
plt.ylabel("Number of Recordings")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig("figures/class_distribution.png")
plt.show()
