"""
Advanced Animal Detection Training - Multiple Object Detection Approaches
"""

import os
import cv2
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import json
import warnings
warnings.filterwarnings('ignore')

# PyTorch and related imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn, ssdlite320_mobilenet_v3_large
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.ssd import SSDClassificationHead

# For data augmentation
import albumentations as A
from albumentations.pytorch import ToTensorV2

class AnimalDetectionDatasetAdvanced(Dataset):
    """Advanced dataset class with bounding box support"""
    
    def __init__(self, data_dir, csv_file, xml_file, transform=None, target_size=(640, 640)):
        self.data_dir = data_dir
        self.annotations = pd.read_csv(os.path.join(data_dir, csv_file))
        self.xml_file = os.path.join(data_dir, xml_file)
        self.transform = transform
        self.target_size = target_size
        self.image_boxes = self._parse_annotations()
        
    def _parse_annotations(self):
        """Parse XML annotations with detailed bounding box information"""
        tree = ET.parse(self.xml_file)
        root = tree.getroot()
        
        image_boxes = {}
        for image_elem in root.findall('.//image'):
            image_id = int(image_elem.get('id'))
            image_name = image_elem.get('name')
            width = int(image_elem.get('width'))
            height = int(image_elem.get('height'))
            
            boxes = []
            for box_elem in image_elem.findall('box'):
                if box_elem.get('label') == 'cow':
                    xtl = float(box_elem.get('xtl'))
                    ytl = float(box_elem.get('ytl'))
                    xbr = float(box_elem.get('xbr'))
                    ybr = float(box_elem.get('ybr'))
                    is_visible = box_elem.find('attribute[@name="is_visible"]').text == 'true'
                    
                    # Convert to [x_min, y_min, x_max, y_max] format
                    boxes.append({
                        'bbox': [xtl, ytl, xbr, ybr],
                        'is_visible': is_visible,
                        'label': 1,  # cow class
                        'area': (xbr - xtl) * (ybr - ytl),
                        'iscrowd': 0
                    })
            
            image_boxes[image_id] = {
                'image_name': image_name,
                'boxes': boxes,
                'num_boxes': len(boxes),
                'width': width,
                'height': height
            }
        
        return image_boxes
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]
        image_id = row['image_id']
        image_path = os.path.join(self.data_dir, row['image_name'])
        
        # Load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_height, original_width = image.shape[:2]
        
        # Get annotations
        annotation = self.image_boxes.get(image_id, {})
        boxes = annotation.get('boxes', [])
        
        # Prepare targets for object detection
        if boxes:
            boxes_array = np.array([box['bbox'] for box in boxes], dtype=np.float32)
            labels_array = np.array([box['label'] for box in boxes], dtype=np.int64)
            areas = np.array([box['area'] for box in boxes], dtype=np.float32)
        else:
            # No boxes - create empty annotations
            boxes_array = np.zeros((0, 4), dtype=np.float32)
            labels_array = np.array([], dtype=np.int64)
            areas = np.array([], dtype=np.float32)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(
                image=image,
                bboxes=boxes_array,
                labels=labels_array
            )
            image = transformed['image']
            boxes_array = np.array(transformed['bboxes'])
            labels_array = np.array(transformed['labels'])
        
        # Convert to tensors
        boxes_tensor = torch.as_tensor(boxes_array, dtype=torch.float32)
        labels_tensor = torch.as_tensor(labels_array, dtype=torch.int64)
        areas_tensor = torch.as_tensor(areas, dtype=torch.float32)
        image_id_tensor = torch.tensor([image_id])
        
        target = {
            'boxes': boxes_tensor,
            'labels': labels_tensor,
            'image_id': image_id_tensor,
            'area': areas_tensor,
            'iscrowd': torch.zeros(len(boxes_tensor), dtype=torch.int64)
        }
        
        return image, target

class AdvancedAnimalDetectionPipeline:
    """Advanced pipeline for training multiple object detection models"""
    
    def __init__(self, data_dir="data/animal_detection"):
        self.data_dir = data_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.results = {}
        self.timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        
        # Create directories
        self.model_dir = f"models/animal_detection_advanced_{self.timestamp}"
        self.report_dir = f"reports/animal_detection_advanced_{self.timestamp}"
        self.vis_dir = f"visualizations/animal_detection_advanced_{self.timestamp}"
        
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.report_dir, exist_ok=True)
        os.makedirs(self.vis_dir, exist_ok=True)
    
    def create_transforms(self):
        """Create data augmentation transforms"""
        # Training transforms
        self.train_transform = A.Compose([
            A.Resize(640, 640),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.GaussNoise(p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
        
        # Validation transforms
        self.val_transform = A.Compose([
            A.Resize(640, 640),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
    
    def load_data(self):
        """Load and prepare the dataset"""
        print("Loading advanced animal detection dataset...")
        
        self.create_transforms()
        
        # Create dataset
        dataset = AnimalDetectionDatasetAdvanced(
            data_dir=self.data_dir,
            csv_file="cows.csv",
            xml_file="annotations.xml",
            transform=None  # Will apply transforms during training
        )
        
        # Split data
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        # Apply transforms
        train_dataset.dataset.transform = self.train_transform
        val_dataset.dataset.transform = self.val_transform
        
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=4,  # Small batch size for object detection
            shuffle=True, 
            collate_fn=self.collate_fn
        )
        self.val_loader = DataLoader(
            val_dataset, 
            batch_size=4, 
            shuffle=False, 
            collate_fn=self.collate_fn
        )
        
        print(f"Dataset loaded: {len(train_dataset)} training samples, {len(val_dataset)} validation samples")
        return dataset
    
    def collate_fn(self, batch):
        """Custom collate function for object detection"""
        return tuple(zip(*batch))
    
    def train_faster_rcnn(self, epochs=25):
        """Train Faster R-CNN model"""
        print("\n=== Training Faster R-CNN ===")
        
        # Load pretrained model
        model = fasterrcnn_resnet50_fpn(pretrained=True)
        
        # Replace the classifier with a new one for our classes
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)  # 2 classes: background + cow
        
        model = model.to(self.device)
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        
        train_losses = []
        
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            num_batches = 0
            
            for images, targets in self.train_loader:
                images = list(image.to(self.device) for image in images)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
                
                epoch_loss += losses.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            train_losses.append(avg_loss)
            
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')
            lr_scheduler.step()
        
        # Evaluation
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for images, targets in self.val_loader:
                images = list(image.to(self.device) for image in images)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                val_losses.append(losses.item())
        
        avg_val_loss = sum(val_losses) / len(val_losses)
        print(f"Validation Loss: {avg_val_loss:.4f}")
        
        # Save model
        model_path = os.path.join(self.model_dir, "faster_rcnn_animal_detector.pth")
        torch.save(model.state_dict(), model_path)
        
        self.models['faster_rcnn'] = model
        self.results['faster_rcnn'] = {
            'train_losses': train_losses,
            'val_loss': avg_val_loss,
            'model_path': model_path
        }
        
        return model
    
    def train_ssd(self, epochs=25):
        """Train SSD (Single Shot Detector) model"""
        print("\n=== Training SSD ===")
        
        # Load pretrained SSD model
        model = ssdlite320_mobilenet_v3_large(pretrained=True)
        
        # Modify the classification head for our classes
        in_channels = model.head.classification_head.in_channels
        num_anchors = model.head.classification_head.num_anchors
        model.head.classification_head = SSDClassificationHead(
            in_channels, num_anchors, 2  # 2 classes: background + cow
        )
        
        model = model.to(self.device)
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        
        train_losses = []
        
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            num_batches = 0
            
            for images, targets in self.train_loader:
                images = list(image.to(self.device) for image in images)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
                
                epoch_loss += losses.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            train_losses.append(avg_loss)
            
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')
            lr_scheduler.step()
        
        # Evaluation
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for images, targets in self.val_loader:
                images = list(image.to(self.device) for image in images)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                val_losses.append(losses.item())
        
        avg_val_loss = sum(val_losses) / len(val_losses)
        print(f"Validation Loss: {avg_val_loss:.4f}")
        
        # Save model
        model_path = os.path.join(self.model_dir, "ssd_animal_detector.pth")
        torch.save(model.state_dict(), model_path)
        
        self.models['ssd'] = model
        self.results['ssd'] = {
            'train_losses': train_losses,
            'val_loss': avg_val_loss,
            'model_path': model_path
        }
        
        return model
    
    def train_retinanet(self, epochs=25):
        """Train RetinaNet model"""
        print("\n=== Training RetinaNet ===")
        
        try:
            from torchvision.models.detection import retinanet_resnet50_fpn
            
            # Load pretrained RetinaNet
            model = retinanet_resnet50_fpn(pretrained=True)
            
            # Modify classifier for our classes
            in_channels = model.head.classification_head.conv[0].in_channels
            num_anchors = model.head.classification_head.num_anchors
            model.head.classification_head.num_classes = 2  # background + cow
            
            # Reinitialize the classification head
            model.head.classification_head.conv = nn.Conv2d(
                in_channels, num_anchors * 2, kernel_size=3, stride=1, padding=1
            )
            
            model = model.to(self.device)
            params = [p for p in model.parameters() if p.requires_grad]
            optimizer = optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
            
            train_losses = []
            
            for epoch in range(epochs):
                model.train()
                epoch_loss = 0
                num_batches = 0
                
                for images, targets in self.train_loader:
                    images = list(image.to(self.device) for image in images)
                    targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                    
                    loss_dict = model(images, targets)
                    if 'loss_classifier' in loss_dict:
                        losses = loss_dict['loss_classifier'] + loss_dict.get('loss_box_reg', 0)
                    else:
                        losses = sum(loss for loss in loss_dict.values())
                    
                    optimizer.zero_grad()
                    losses.backward()
                    optimizer.step()
                    
                    epoch_loss += losses.item()
                    num_batches += 1
                
                avg_loss = epoch_loss / num_batches
                train_losses.append(avg_loss)
                
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')
                lr_scheduler.step()
            
            # Evaluation
            model.eval()
            val_losses = []
            
            with torch.no_grad():
                for images, targets in self.val_loader:
                    images = list(image.to(self.device) for image in images)
                    targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                    
                    loss_dict = model(images, targets)
                    if 'loss_classifier' in loss_dict:
                        losses = loss_dict['loss_classifier'] + loss_dict.get('loss_box_reg', 0)
                    else:
                        losses = sum(loss for loss in loss_dict.values())
                    val_losses.append(losses.item())
            
            avg_val_loss = sum(val_losses) / len(val_losses)
            print(f"Validation Loss: {avg_val_loss:.4f}")
            
            # Save model
            model_path = os.path.join(self.model_dir, "retinanet_animal_detector.pth")
            torch.save(model.state_dict(), model_path)
            
            self.models['retinanet'] = model
            self.results['retinanet'] = {
                'train_losses': train_losses,
                'val_loss': avg_val_loss,
                'model_path': model_path
            }
            
            return model
            
        except ImportError:
            print("RetinaNet not available in this torchvision version")
            return None
    
    def generate_advanced_reports(self):
        """Generate detailed reports for object detection models"""
        print("\n=== Generating Advanced Reports ===")
        
        summary = {
            'timestamp': self.timestamp,
            'models_trained': list(self.models.keys()),
            'results': {}
        }
        
        for model_name, result in self.results.items():
            summary['results'][model_name] = {
                'final_train_loss': result['train_losses'][-1] if result['train_losses'] else None,
                'validation_loss': result['val_loss'],
                'model_path': result['model_path']
            }
        
        # Save summary
        summary_path = os.path.join(self.report_dir, "advanced_detection_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Generate individual model reports
        for model_name, result in self.results.items():
            report = f"""
# {model_name.upper()} Animal Detection Report

## Model Performance
- Final Training Loss: {result['train_losses'][-1]:.4f}
- Validation Loss: {result['val_loss']:.4f}
- Training Completed: {self.timestamp}

## Model Details
- Model Architecture: {model_name}
- Number of Classes: 2 (background + cow)
- Input Size: 640x640
- Model Path: {result['model_path']}

## Training Configuration
- Epochs: 25
- Batch Size: 4
- Optimizer: SGD with momentum
- Learning Rate Schedule: StepLR (gamma=0.1 every 5 epochs)
"""
            
            report_path = os.path.join(self.report_dir, f"{model_name}_detailed_report.txt")
            with open(report_path, 'w') as f:
                f.write(report)
        
        print(f"Advanced reports saved to: {self.report_dir}")
        return summary
    
    def generate_detection_visualizations(self):
        """Generate loss curves and model comparison visualizations"""
        print("\n=== Generating Detection Visualizations ===")
        
        # Training loss curves
        plt.figure(figsize=(12, 8))
        
        for model_name, result in self.results.items():
            if 'train_losses' in result and result['train_losses']:
                epochs = range(1, len(result['train_losses']) + 1)
                plt.plot(epochs, result['train_losses'], 
                        marker='o', label=f"{model_name.upper()}", linewidth=2)
        
        plt.title('Object Detection Training Loss Curves', fontsize=16, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        loss_curve_path = os.path.join(self.vis_dir, "detection_loss_curves.png")
        plt.savefig(loss_curve_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Model comparison bar chart
        model_names = []
        val_losses = []
        
        for model_name, result in self.results.items():
            if 'val_loss' in result:
                model_names.append(model_name.upper())
                val_losses.append(result['val_loss'])
        
        if model_names:
            plt.figure(figsize=(10, 6))
            bars = plt.bar(model_names, val_losses, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
            plt.title('Object Detection Model Comparison', fontsize=16, fontweight='bold')
            plt.ylabel('Validation Loss', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            
            # Add value labels on bars
            for bar, loss in zip(bars, val_losses):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{loss:.3f}', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            comparison_path = os.path.join(self.vis_dir, "model_comparison.png")
            plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"Detection visualizations saved to: {self.vis_dir}")
    
    def run_advanced_pipeline(self):
        """Run the complete advanced detection pipeline"""
        print("=" * 70)
        print("ADVANCED ANIMAL DETECTION TRAINING PIPELINE")
        print("=" * 70)
        
        # Load data
        self.load_data()
        
        # Train all detection models
        self.train_faster_rcnn(epochs=25)
        self.train_ssd(epochs=25)
        self.train_retinanet(epochs=25)
        
        # Generate reports and visualizations
        self.generate_advanced_reports()
        self.generate_detection_visualizations()
        
        # Summary
        print("\n" + "=" * 70)
        print("ADVANCED TRAINING COMPLETE!")
        print("=" * 70)
        print(f"Models saved to: {self.model_dir}")
        print(f"Reports saved to: {self.report_dir}")
        print(f"Visualizations saved to: {self.vis_dir}")
        print("\nModel Performance Summary:")
        
        for model_name, result in self.results.items():
            print(f"- {model_name.upper()}: Val Loss = {result['val_loss']:.4f}")

if __name__ == "__main__":
    # Run the advanced pipeline
    pipeline = AdvancedAnimalDetectionPipeline()
    pipeline.run_advanced_pipeline()