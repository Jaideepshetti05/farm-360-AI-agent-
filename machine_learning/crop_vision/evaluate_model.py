import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from config import DATA_DIR, MODEL_DIR, RESULTS_DIR, BATCH_SIZE, TEST_SPLIT

device = torch.device("cpu")

os.makedirs(RESULTS_DIR, exist_ok=True)

# Transform
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

train_size = int((1 - TEST_SPLIT) * len(dataset))
val_size = len(dataset) - train_size

_, val_dataset = random_split(dataset, [train_size, val_size])

val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print("Loading trained model...")

model = models.resnet18(weights=None)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(dataset.classes))

model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "crop_disease_model.pth")))
model = model.to(device)
model.eval()

all_preds = []
all_labels = []

print("Evaluating...")

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
print(f"Validation Accuracy: {accuracy:.4f}")

# Classification Report
report = classification_report(all_labels, all_preds, target_names=dataset.classes)
print(report)

with open(os.path.join(RESULTS_DIR, "classification_report.txt"), "w") as f:
    f.write(report)

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=False, cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"))
plt.close()

print("Evaluation complete. Results saved.")