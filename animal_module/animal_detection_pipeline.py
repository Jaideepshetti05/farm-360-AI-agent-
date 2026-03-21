"""
Comprehensive Animal Detection Pipeline
Trains multiple ML models for cow detection using various approaches
"""

import os
import cv2
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# Deep learning imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import resnet50, mobilenet_v2
import torchvision

# YOLO imports (if available)
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("YOLO not available, will skip YOLO training")

class AnimalDetectionDataset(Dataset):
    """Dataset class for animal detection"""
    
    def __init__(self, data_dir, csv_file, xml_file, transform=None, target_size=(224, 224)):
        self.data_dir = data_dir
        self.annotations = pd.read_csv(os.path.join(data_dir, csv_file))
        self.xml_file = os.path.join(data_dir, xml_file)
        self.transform = transform
        self.target_size = target_size
        self.image_boxes = self._parse_annotations()
        
    def _parse_annotations(self):
        """Parse XML annotations to extract bounding boxes"""
        tree = ET.parse(self.xml_file)
        root = tree.getroot()
        
        image_boxes = {}
        for image_elem in root.findall('.//image'):
            image_id = int(image_elem.get('id'))
            image_name = image_elem.get('name')
            
            boxes = []
            for box_elem in image_elem.findall('box'):
                if box_elem.get('label') == 'cow':
                    xtl = float(box_elem.get('xtl'))
                    ytl = float(box_elem.get('ytl'))
                    xbr = float(box_elem.get('xbr'))
                    ybr = float(box_elem.get('ybr'))
                    is_visible = box_elem.find('attribute[@name="is_visible"]').text == 'true'
                    boxes.append({
                        'xtl': xtl, 'ytl': ytl, 'xbr': xbr, 'ybr': ybr,
                        'is_visible': is_visible, 'label': 'cow'
                    })
            
            image_boxes[image_id] = {
                'image_name': image_name,
                'boxes': boxes,
                'has_cows': len(boxes) > 0
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
        
        # Get bounding boxes
        annotation = self.image_boxes.get(image_id, {})
        boxes = annotation.get('boxes', [])
        has_cows = annotation.get('has_cows', False)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'image_id': image_id,
            'boxes': boxes,
            'has_cows': has_cows,
            'image_path': image_path
        }

class SimpleCNN(nn.Module):
    """Simple CNN for cow detection classification"""
    
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class AnimalDetectionPipeline:
    """Main pipeline for training multiple animal detection models"""
    
    def __init__(self, data_dir="data/animal_detection"):
        self.data_dir = data_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.results = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create directories
        self.model_dir = f"models/animal_detection_{self.timestamp}"
        self.report_dir = f"reports/animal_detection_{self.timestamp}"
        self.vis_dir = f"visualizations/animal_detection_{self.timestamp}"
        
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.report_dir, exist_ok=True)
        os.makedirs(self.vis_dir, exist_ok=True)
    
    def load_data(self):
        """Load and preprocess the animal detection dataset"""
        print("Loading animal detection dataset...")
        
        # Define transforms
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Create dataset
        dataset = AnimalDetectionDataset(
            data_dir=self.data_dir,
            csv_file="cows.csv",
            xml_file="annotations.xml",
            transform=transform
        )
        
        # Split data
        indices = list(range(len(dataset)))
        train_indices, test_indices = train_test_split(
            indices, test_size=0.2, random_state=42, stratify=[dataset[i]['has_cows'] for i in indices]
        )
        
        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        test_dataset = torch.utils.data.Subset(dataset, test_indices)
        
        self.train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
        
        print(f"Dataset loaded: {len(train_dataset)} training samples, {len(test_dataset)} test samples")
        return dataset
    
    def train_simple_cnn(self, epochs=20):
        """Train simple CNN classifier"""
        print("\n=== Training Simple CNN Classifier ===")
        
        model = SimpleCNN(num_classes=2).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        train_losses = []
        train_accuracies = []
        
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for batch in self.train_loader:
                images = batch['image'].to(self.device)
                labels = torch.tensor([int(x) for x in batch['has_cows']]).to(self.device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            
            epoch_loss = running_loss / len(self.train_loader)
            epoch_acc = 100. * correct / total
            train_losses.append(epoch_loss)
            train_accuracies.append(epoch_acc)
            
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')
        
        # Evaluate
        model.eval()
        test_correct = 0
        test_total = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in self.test_loader:
                images = batch['image'].to(self.device)
                labels = torch.tensor([int(x) for x in batch['has_cows']]).to(self.device)
                
                outputs = model(images)
                _, predicted = outputs.max(1)
                
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        test_accuracy = 100. * test_correct / test_total
        print(f"Test Accuracy: {test_accuracy:.2f}%")
        
        # Save model
        model_path = os.path.join(self.model_dir, "simple_cnn_animal_detector.pth")
        torch.save(model.state_dict(), model_path)
        
        # Store results
        self.models['simple_cnn'] = model
        self.results['simple_cnn'] = {
            'train_losses': train_losses,
            'train_accuracies': train_accuracies,
            'test_accuracy': test_accuracy,
            'predictions': all_predictions,
            'labels': all_labels,
            'model_path': model_path
        }
        
        return model
    
    def train_resnet50(self, epochs=15):
        """Train ResNet50 transfer learning model"""
        print("\n=== Training ResNet50 Classifier ===")
        
        # Load pretrained ResNet50
        model = resnet50(pretrained=True)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 2)  # 2 classes: cow/no cow
        model = model.to(self.device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        
        train_losses = []
        train_accuracies = []
        
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for batch in self.train_loader:
                images = batch['image'].to(self.device)
                labels = torch.tensor([int(x) for x in batch['has_cows']]).to(self.device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            
            epoch_loss = running_loss / len(self.train_loader)
            epoch_acc = 100. * correct / total
            train_losses.append(epoch_loss)
            train_accuracies.append(epoch_acc)
            
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')
        
        # Evaluate
        model.eval()
        test_correct = 0
        test_total = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in self.test_loader:
                images = batch['image'].to(self.device)
                labels = torch.tensor([int(x) for x in batch['has_cows']]).to(self.device)
                
                outputs = model(images)
                _, predicted = outputs.max(1)
                
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        test_accuracy = 100. * test_correct / test_total
        print(f"Test Accuracy: {test_accuracy:.2f}%")
        
        # Save model
        model_path = os.path.join(self.model_dir, "resnet50_animal_detector.pth")
        torch.save(model.state_dict(), model_path)
        
        # Store results
        self.models['resnet50'] = model
        self.results['resnet50'] = {
            'train_losses': train_losses,
            'train_accuracies': train_accuracies,
            'test_accuracy': test_accuracy,
            'predictions': all_predictions,
            'labels': all_labels,
            'model_path': model_path
        }
        
        return model
    
    def train_mobilenet(self, epochs=15):
        """Train MobileNetV2 transfer learning model"""
        print("\n=== Training MobileNetV2 Classifier ===")
        
        # Load pretrained MobileNetV2
        model = mobilenet_v2(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
        model = model.to(self.device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        
        train_losses = []
        train_accuracies = []
        
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for batch in self.train_loader:
                images = batch['image'].to(self.device)
                labels = torch.tensor([int(x) for x in batch['has_cows']]).to(self.device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            
            epoch_loss = running_loss / len(self.train_loader)
            epoch_acc = 100. * correct / total
            train_losses.append(epoch_loss)
            train_accuracies.append(epoch_acc)
            
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')
        
        # Evaluate
        model.eval()
        test_correct = 0
        test_total = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in self.test_loader:
                images = batch['image'].to(self.device)
                labels = torch.tensor([int(x) for x in batch['has_cows']]).to(self.device)
                
                outputs = model(images)
                _, predicted = outputs.max(1)
                
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        test_accuracy = 100. * test_correct / test_total
        print(f"Test Accuracy: {test_accuracy:.2f}%")
        
        # Save model
        model_path = os.path.join(self.model_dir, "mobilenet_animal_detector.pth")
        torch.save(model.state_dict(), model_path)
        
        # Store results
        self.models['mobilenet'] = model
        self.results['mobilenet'] = {
            'train_losses': train_losses,
            'train_accuracies': train_accuracies,
            'test_accuracy': test_accuracy,
            'predictions': all_predictions,
            'labels': all_labels,
            'model_path': model_path
        }
        
        return model

    def train_yolo(self):
        """Train YOLOv8 object detection model (if available)"""
        if not YOLO_AVAILABLE:
            print("YOLO not available, skipping YOLO training")
            return None
            
        print("\n=== Training YOLOv8 Object Detector ===")
        
        try:
            # Create YOLO dataset YAML
            yaml_content = f"""
path: {self.data_dir}
train: images
val: images
nc: 1
names: ['cow']
"""
            
            yaml_path = os.path.join(self.data_dir, "dataset.yaml")
            with open(yaml_path, 'w') as f:
                f.write(yaml_content)
            
            # Train YOLO model
            model = YOLO('yolov8n.pt')  # Load pretrained YOLOv8 nano model
            results = model.train(
                data=yaml_path,
                epochs=50,
                imgsz=640,
                batch=16,
                name=f'yolo_animal_detector_{self.timestamp}'
            )
            
            # Save model
            model_path = os.path.join(self.model_dir, "yolo_animal_detector.pt")
            model.save(model_path)
            
            # Store results
            self.models['yolo'] = model
            self.results['yolo'] = {
                'model_path': model_path,
                'results': results
            }
            
            print("YOLO training completed successfully")
            return model
            
        except Exception as e:
            print(f"YOLO training failed: {e}")
            return None
    
    def create_ensemble(self):
        """Create ensemble model combining all trained models"""
        print("\n=== Creating Ensemble Model ===")
        
        if len(self.models) < 2:
            print("Need at least 2 models for ensemble")
            return None
        
        # Simple voting ensemble
        all_test_predictions = []
        all_test_labels = []
        
        # Get predictions from all models
        for model_name in ['simple_cnn', 'resnet50', 'mobilenet']:
            if model_name in self.results:
                predictions = self.results[model_name]['predictions']
                labels = self.results[model_name]['labels']
                all_test_predictions.append(predictions)
                if len(all_test_labels) == 0:
                    all_test_labels = labels
        
        # Ensemble voting
        if all_test_predictions:
            ensemble_predictions = []
            for i in range(len(all_test_predictions[0])):
                votes = [pred[i] for pred in all_test_predictions]
                ensemble_pred = 1 if sum(votes) > len(votes) / 2 else 0
                ensemble_predictions.append(ensemble_pred)
            
            # Calculate ensemble accuracy
            ensemble_correct = sum(1 for i in range(len(ensemble_predictions)) 
                                 if ensemble_predictions[i] == all_test_labels[i])
            ensemble_accuracy = 100. * ensemble_correct / len(ensemble_predictions)
            
            print(f"Ensemble Accuracy: {ensemble_accuracy:.2f}%")
            
            self.results['ensemble'] = {
                'accuracy': ensemble_accuracy,
                'predictions': ensemble_predictions,
                'labels': all_test_labels
            }
            
            return ensemble_predictions
        
        return None
    
    def generate_reports(self):
        """Generate comprehensive evaluation reports"""
        print("\n=== Generating Reports ===")
        
        # Overall results summary
        summary = {
            'timestamp': self.timestamp,
            'models_trained': list(self.models.keys()),
            'results': {}
        }
        
        for model_name, result in self.results.items():
            if 'test_accuracy' in result:
                summary['results'][model_name] = {
                    'test_accuracy': result['test_accuracy']
                }
            elif 'accuracy' in result:  # ensemble
                summary['results'][model_name] = {
                    'accuracy': result['accuracy']
                }
        
        # Save summary
        summary_path = os.path.join(self.report_dir, "animal_detection_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Generate detailed reports for each model
        for model_name, result in self.results.items():
            if 'test_accuracy' in result:
                report = f"""
# {model_name.upper()} Animal Detection Report

## Model Performance
- Test Accuracy: {result['test_accuracy']:.2f}%
- Training Completed: {self.timestamp}

## Classification Report
{classification_report(result['labels'], result['predictions'], 
                       target_names=['No Cow', 'Cow'], output_dict=False)}

## Model Details
- Model Path: {result['model_path']}
- Training Timestamp: {self.timestamp}
"""
                
                report_path = os.path.join(self.report_dir, f"{model_name}_report.txt")
                with open(report_path, 'w') as f:
                    f.write(report)
        
        print(f"Reports saved to: {self.report_dir}")
        return summary
    
    def generate_visualizations(self):
        """Generate training curves and confusion matrices"""
        print("\n=== Generating Visualizations ===")
        
        # Training curves
        plt.figure(figsize=(15, 5))
        
        # Plot 1: Training Losses
        plt.subplot(1, 2, 1)
        for model_name, result in self.results.items():
            if 'train_losses' in result:
                plt.plot(result['train_losses'], label=f"{model_name} Loss", marker='o')
        plt.title('Training Loss Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Plot 2: Training Accuracies
        plt.subplot(1, 2, 2)
        for model_name, result in self.results.items():
            if 'train_accuracies' in result:
                plt.plot(result['train_accuracies'], label=f"{model_name} Accuracy", marker='s')
        plt.title('Training Accuracy Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        loss_curve_path = os.path.join(self.vis_dir, "training_curves.png")
        plt.savefig(loss_curve_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Confusion matrices
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        model_names = ['simple_cnn', 'resnet50', 'mobilenet']
        
        for i, model_name in enumerate(model_names):
            if model_name in self.results:
                result = self.results[model_name]
                cm = confusion_matrix(result['labels'], result['predictions'])
                
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                           xticklabels=['No Cow', 'Cow'], 
                           yticklabels=['No Cow', 'Cow'], ax=axes[i])
                axes[i].set_title(f'{model_name.upper()} Confusion Matrix')
                axes[i].set_xlabel('Predicted')
                axes[i].set_ylabel('Actual')
        
        plt.tight_layout()
        cm_path = os.path.join(self.vis_dir, "confusion_matrices.png")
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualizations saved to: {self.vis_dir}")
    
    def run_complete_pipeline(self):
        """Run the complete animal detection training pipeline"""
        print("=" * 60)
        print("ANIMAL DETECTION TRAINING PIPELINE")
        print("=" * 60)
        
        # Load data
        self.load_data()
        
        # Train all models
        self.train_simple_cnn(epochs=20)
        self.train_resnet50(epochs=15)
        self.train_mobilenet(epochs=15)
        self.train_yolo()  # Optional
        
        # Create ensemble
        self.create_ensemble()
        
        # Generate reports and visualizations
        self.generate_reports()
        self.generate_visualizations()
        
        # Summary
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE!")
        print("=" * 60)
        print(f"Models saved to: {self.model_dir}")
        print(f"Reports saved to: {self.report_dir}")
        print(f"Visualizations saved to: {self.vis_dir}")
        print("\nModel Performance Summary:")
        
        for model_name, result in self.results.items():
            if 'test_accuracy' in result:
                print(f"- {model_name}: {result['test_accuracy']:.2f}%")
            elif 'accuracy' in result:
                print(f"- {model_name}: {result['accuracy']:.2f}%")

if __name__ == "__main__":
    # Run the complete pipeline
    pipeline = AnimalDetectionPipeline()
    pipeline.run_complete_pipeline()