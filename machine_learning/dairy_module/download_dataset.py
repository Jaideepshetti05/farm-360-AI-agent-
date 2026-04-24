import kagglehub
import os
import shutil

# Download dataset
path = kagglehub.dataset_download("madhuraatmarambhagat/milk-production-dataset")

print("Downloaded to:", path)

# Target folder
target_folder = "data/dairy"
os.makedirs(target_folder, exist_ok=True)

# Copy CSV files to data/dairy
for file in os.listdir(path):
    if file.endswith(".csv"):
        source_file = os.path.join(path, file)
        destination_file = os.path.join(target_folder, file)
        shutil.copy(source_file, destination_file)
        print("Copied:", file)

print("Dataset stored inside data/dairy/")
