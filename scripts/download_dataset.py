import kagglehub

from src.paths import KAGGLE_HANDLE, DATASET_PATH_FILE, save_dataset_path

# Download UrbanSound8K (kagglehub reuses its cache if already downloaded)
path = kagglehub.dataset_download(KAGGLE_HANDLE)
save_dataset_path(path)

print("Dataset downloaded successfully!")
print("Path to dataset files:", path)
print("Saved dataset path to:", DATASET_PATH_FILE)
