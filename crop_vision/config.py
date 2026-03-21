import os

# Base directory (crop_vision folder)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Dataset path
DATA_DIR = os.path.join(
    BASE_DIR,
    "..",
    "data",
    "crop",
    "images",
    "Crop Diseases"
)

# Model save directory
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Results directory
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Training parameters
BATCH_SIZE = 32
NUM_EPOCHS = 8
LEARNING_RATE = 0.001
TEST_SPLIT = 0.2
RANDOM_SEED = 42