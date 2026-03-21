import os

# Base project path (auto-detect)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Dataset path (DO NOT change unless needed)
DATA_DIR = os.path.join(
    BASE_DIR,
    "..",
    "data",
    "animal_images",
    "animalsdata",
    "raw-img"
)

# Embedding save path
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "embeddings")

# Results path
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Training settings
BATCH_SIZE = 32
TEST_SIZE = 0.2
RANDOM_SEED = 42

# Model choice
CLASSIFIER_TYPE = "logistic"  # options: logistic, svm