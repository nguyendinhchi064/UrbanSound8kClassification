
from data_loader import load_data, CLASS_NAMES

print("Testing multiple audio files...")

# Load only fold 1
X, y = load_data(folds=(1,))

print("Input shape:", X.shape)
print("Labels shape:", y.shape)
print("Number of samples:", len(X))

print("\nFirst 5 labels:")
for label in y[:5]:
    print(label, "=", CLASS_NAMES[label])

print("\nMinimum value:", X.min())
print("Maximum value:", X.max())

assert X.shape[0] == len(y)
assert X.shape[1:] == (128, 173, 1)
assert X.min() >= 0
assert X.max() <= 1

print("\nAll tests passed!")
