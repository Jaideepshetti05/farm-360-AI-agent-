"""
Farm360 AI – EfficientNet-B3 Crop Disease Classifier
======================================================
Upgrade from ResNet18 (17 classes) to EfficientNet-B3 (42 classes).
Uses Albumentations for aggressive data augmentation.

Dataset: machine_learning/data/crop/images/Train/  (42 classes)
Output:  machine_learning/model_registry/models/crop_disease/v2.0/weights.pth

Usage:
    cd "ml models"
    python machine_learning/training/train_crop_disease_efficientnet.py

Requirements:
    pip install torch torchvision albumentations tqdm scikit-learn
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models, transforms

# Optional Albumentations (falls back to torchvision transforms)
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    import cv2
    _ALB = True
except ImportError:
    _ALB = False
    print("[WARN] Albumentations not installed — using torchvision transforms only")

from PIL import Image

# ── Config ─────────────────────────────────────────────────────────────────────
ROOT_DIR    = Path(__file__).resolve().parent.parent.parent  # ml models/
DATA_DIR    = ROOT_DIR / "machine_learning" / "data" / "crop" / "images" / "Train"
OUT_DIR     = ROOT_DIR / "machine_learning" / "model_registry" / "models" / "crop_disease" / "v2.0"
CLASSES_DIR = ROOT_DIR / "machine_learning" / "model_registry" / "classes"

BATCH_SIZE   = 32
NUM_EPOCHS   = 40
LR           = 1e-4
VAL_SPLIT    = 0.15
TEST_SPLIT   = 0.10
IMAGE_SIZE   = 224
NUM_WORKERS  = 0   # Set to 4 on Linux/Mac
PATIENCE     = 6   # Early stopping patience
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Augmented Dataset ──────────────────────────────────────────────────────────

class CropDiseaseDataset(Dataset):
    """ImageFolder-style dataset with Albumentations augmentation."""

    def __init__(self, samples: list, class_to_idx: dict, augment: bool = False):
        self.samples = samples
        self.class_to_idx = class_to_idx
        self.augment = augment

        if _ALB and augment:
            self.transform = A.Compose([
                A.RandomResizedCrop(IMAGE_SIZE, IMAGE_SIZE, scale=(0.7, 1.0)),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05, p=0.7),
                A.GaussianBlur(blur_limit=(3, 5), p=0.2),
                A.GaussNoise(p=0.2),
                A.RandomRotate90(p=0.2),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.4),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
            self.val_transform = A.Compose([
                A.Resize(IMAGE_SIZE, IMAGE_SIZE),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.3, 0.3, 0.3, 0.05),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            self.val_transform = transforms.Compose([
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")

        if _ALB and self.augment:
            img_np = np.array(img)
            transformed = self.transform(image=img_np)
            tensor = transformed["image"]
        elif _ALB and not self.augment:
            img_np = np.array(img)
            transformed = self.val_transform(image=img_np)
            tensor = transformed["image"]
        else:
            t = self.transform if self.augment else self.val_transform
            tensor = t(img)

        return tensor, label


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print(f"Device: {DEVICE}")
    print(f"Data dir: {DATA_DIR}")

    if not DATA_DIR.exists():
        sys.exit(f"ERROR: Data directory not found: {DATA_DIR}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Collect samples ────────────────────────────────────────────────────────
    classes = sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir()])
    class_to_idx = {c: i for i, c in enumerate(classes)}
    num_classes = len(classes)
    print(f"Classes: {num_classes}")

    all_samples = []
    for cls in classes:
        cls_dir = DATA_DIR / cls
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp"):
            for img_path in cls_dir.glob(ext):
                all_samples.append((str(img_path), class_to_idx[cls]))
            for img_path in cls_dir.glob(ext.upper()):
                all_samples.append((str(img_path), class_to_idx[cls]))

    np.random.shuffle(all_samples)
    total = len(all_samples)
    print(f"Total images: {total}")

    if total < 100:
        sys.exit("ERROR: Too few images found. Check DATA_DIR path.")

    # ── Split ──────────────────────────────────────────────────────────────────
    n_test = int(total * TEST_SPLIT)
    n_val  = int(total * VAL_SPLIT)
    n_train = total - n_val - n_test

    train_samples = all_samples[:n_train]
    val_samples   = all_samples[n_train:n_train + n_val]
    test_samples  = all_samples[n_train + n_val:]

    print(f"Train: {len(train_samples)} | Val: {len(val_samples)} | Test: {len(test_samples)}")

    train_ds = CropDiseaseDataset(train_samples, class_to_idx, augment=True)
    val_ds   = CropDiseaseDataset(val_samples,   class_to_idx, augment=False)
    test_ds  = CropDiseaseDataset(test_samples,  class_to_idx, augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # ── Model ──────────────────────────────────────────────────────────────────
    print("Loading EfficientNet-B3 (ImageNet pretrained)...")
    model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    model = model.to(DEVICE)

    # Loss + optimiser (AdamW with cosine decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    # ── Training loop ──────────────────────────────────────────────────────────
    best_val_acc = 0.0
    best_weights = None
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    print(f"\nStarting training for {NUM_EPOCHS} epochs (early stop patience={PATIENCE})...")
    for epoch in range(1, NUM_EPOCHS + 1):
        # Train
        model.train()
        running_loss = 0.0
        t0 = time.time()
        for X, y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)

        # Validate
        model.eval()
        val_loss, correct, total_v = 0.0, 0, 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                out = model(X)
                val_loss += criterion(out, y).item()
                pred = out.argmax(dim=1)
                correct += (pred == y).sum().item()
                total_v += y.size(0)

        val_loss /= len(val_loader)
        val_acc = correct / total_v

        scheduler.step()

        history["train_loss"].append(round(train_loss, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["val_acc"].append(round(val_acc, 4))

        elapsed = time.time() - t0
        print(
            f"[{epoch:02d}/{NUM_EPOCHS}] "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"val_acc={val_acc:.4f}  lr={scheduler.get_last_lr()[0]:.6f}  {elapsed:.0f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
                break

    # ── Save ───────────────────────────────────────────────────────────────────
    weights_path = OUT_DIR / "weights.pth"
    torch.save(best_weights, weights_path)
    print(f"\nBest model saved: {weights_path}")
    print(f"Best val accuracy: {best_val_acc:.4f}")

    # Save class labels
    classes_path = CLASSES_DIR / "crop_disease_v2.json"
    with open(classes_path, "w") as f:
        json.dump(classes, f, indent=2)
    print(f"Class labels saved: {classes_path}")

    # ── Test evaluation ────────────────────────────────────────────────────────
    model.load_state_dict(best_weights)
    model.eval()
    test_correct, test_total = 0, 0
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            pred = model(X).argmax(dim=1)
            test_correct += (pred == y).sum().item()
            test_total += y.size(0)

    test_acc = test_correct / test_total if test_total > 0 else 0
    print(f"Test accuracy: {test_acc:.4f}")

    # Save training report
    report = {
        "model": "EfficientNet-B3",
        "num_classes": num_classes,
        "classes": classes,
        "best_val_acc": round(best_val_acc, 4),
        "test_acc": round(test_acc, 4),
        "epochs_trained": epoch,
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "test_samples": len(test_samples),
        "history": history,
        "weights": str(weights_path),
    }
    report_path = OUT_DIR / "training_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Training report saved: {report_path}")
    print("\nDone! To deploy: update registry.json — set crop_disease.latest = 'v2.0'")


if __name__ == "__main__":
    main()
