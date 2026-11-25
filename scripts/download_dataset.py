
import kagglehub

# Download UrbanSound8K
path = kagglehub.dataset_download("chrisfilo/urbansound8k")

print("Dataset downloaded successfully!")
print("Path to dataset files:", path)
