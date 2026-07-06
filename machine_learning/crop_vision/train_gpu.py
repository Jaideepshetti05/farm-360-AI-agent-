import os
import sys
import time
import json
import random
import argparse
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import precision_recall_fscore_support

def set_seed(seed=42):
    """Set deterministic seeds for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class FocalLoss(nn.Module):
    """Focal Loss to address class imbalance."""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def create_experiment_dir(base_dir):
    os.makedirs(base_dir, exist_ok=True)
    runs = [d for d in os.listdir(base_dir) if d.startswith("run_") and os.path.isdir(os.path.join(base_dir, d))]
    next_run_num = max([int(r.split("_")[1]) for r in runs]) + 1 if runs else 1
    run_dir = os.path.join(base_dir, f"run_{next_run_num:03d}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir, f"run_{next_run_num:03d}"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=35, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="DataLoader batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--dataset-dir", type=str, default="", help="Path to clean dataset split parent folder")
    args = parser.parse_args()

    print("==================================================")
    print("🚀 Starting GPU-Accelerated Crop Disease V2 Training")
    print("==================================================")
    
    set_seed(42)
    print("✓ Set reproducible random seed = 42")

    # Resolve dataset paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not args.dataset_dir:
        dataset_base = os.path.abspath(os.path.join(script_dir, "..", "data", "crop_disease", "v2", "clean"))
    else:
        dataset_base = os.path.abspath(args.dataset_dir)
        
    train_dir = os.path.join(dataset_base, "train")
    val_dir = os.path.join(dataset_base, "validation")
    experiments_base = os.path.join(script_dir, "experiments")

    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        print(f"❌ Dataset splits not found at: {dataset_base}. Please run prepare_dataset.py first.")
        sys.exit(1)

    run_dir, run_name = create_experiment_dir(experiments_base)
    print(f"✓ Created experiment directory: {run_dir}")

    # Image transforms with RandAugment
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandAugment(num_ops=2, magnitude=9), # Advanced augmentations
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Datasets and Loaders
    train_dataset = ImageFolder(train_dir, transform=train_transform, allow_empty=True)
    val_dataset = ImageFolder(val_dir, transform=val_transform, allow_empty=True)
    classes = train_dataset.classes

    # Class balancing sampler
    class_counts = {}
    for _, label in train_dataset.samples:
        class_counts[label] = class_counts.get(label, 0) + 1
    
    class_weights = [1.0 / class_counts[i] if i in class_counts else 1.0 for i in range(len(classes))]
    sample_weights = [class_weights[label] for _, label in train_dataset.samples]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    g = torch.Generator()
    g.manual_seed(42)
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=sampler,
        num_workers=2, worker_init_fn=seed_worker, generator=g, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=2, worker_init_fn=seed_worker, generator=g, pin_memory=True
    )

    print(f"✓ Dataset loaded. Train: {len(train_dataset)} images, Val: {len(val_dataset)} images. Target classes: {len(classes)}")

    # Write class mapping JSON
    mapping_path = os.path.abspath(os.path.join(script_dir, "..", "model_registry", "classes", "crop_disease_v2.json"))
    os.makedirs(os.path.dirname(mapping_path), exist_ok=True)
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(classes, f, indent=2)
    print(f"✓ Saved class list crop_disease_v2.json to registry")

    # Rebuild Model
    model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, len(classes))

    # Device allocation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"✓ Allocated model to training device: {device}")

    # Loss, Optimizer, and Scheduler
    criterion = FocalLoss(gamma=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda')) # AMP Mixed Precision

    # Training Loop
    best_f1 = 0.0
    best_epoch = 1
    
    print("\nTraining loop started...")
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        
        t0 = time.time()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                outputs = model(images)
                loss = criterion(outputs, labels)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
            
        avg_train_loss = running_loss / len(train_loader)
        scheduler.step()
        epoch_time = time.time() - t0
        
        # Validation evaluation
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                    outputs = model(images)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
                
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
        top1_correct = sum([1 for p, l in zip(all_preds, all_labels) if p == l])
        val_acc = (top1_correct / len(all_labels)) * 100 if all_labels else 0.0
        
        print(f"Epoch [{epoch+1}/{args.epochs}] ({epoch_time:.1f}s) — Loss: {avg_train_loss:.4f} — Val Acc: {val_acc:.2f}% — Val F1: {f1:.4f}")
        
        # Save best model
        if f1 > best_f1:
            best_f1 = f1
            best_epoch = epoch + 1
            torch.save(model.state_dict(), os.path.join(run_dir, "weights.pth"))
            
    print(f"\n🎉 Training complete. Best Epoch: {best_epoch} with Val F1: {best_f1:.4f}")
    
    # Save final metrics configuration
    model_save_path = os.path.join(run_dir, "weights.pth")
    model_size_mb = os.path.getsize(model_save_path) / (1024 * 1024) if os.path.exists(model_save_path) else 0.0
    
    metrics = {
        "run_name": run_name,
        "epochs_run": args.epochs,
        "best_epoch": best_epoch,
        "val_accuracy_top1": val_acc,
        "macro_f1": best_f1,
        "model_size_mb": model_size_mb
    }
    
    with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    main()
