import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import numpy as np
import os
from tqdm import tqdm
from config import DATA_DIR, EMBEDDINGS_DIR, BATCH_SIZE

# Create embeddings directory if not exists
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

device = torch.device("cpu")

print("Loading ResNet18...")
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = torch.nn.Identity()
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

print("Loading dataset...")
dataset = ImageFolder(DATA_DIR, transform=transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

all_features = []
all_labels = []

print("Extracting features...")
with torch.no_grad():
    for images, labels in tqdm(loader):
        images = images.to(device)
        features = model(images)
        all_features.append(features.cpu().numpy())
        all_labels.append(labels.numpy())

X = np.vstack(all_features)
y = np.hstack(all_labels)

np.save(os.path.join(EMBEDDINGS_DIR, "X_embeddings.npy"), X)
np.save(os.path.join(EMBEDDINGS_DIR, "y_labels.npy"), y)

print("Feature extraction complete.")
print("Embedding shape:", X.shape)