"""
Simple Animal Image Classification Training
===========================================

This script trains your animal dataset using basic Python libraries without external dependencies.
"""

import os
import cv2
import json
from pathlib import Path
from collections import Counter
import random

class SimpleAnimalClassifier:
    """Simple animal image classifier using basic features."""
    
    def __init__(self, data_dir: str = "data/animal_images/animalsdata/raw-img"):
        self.data_dir = Path(data_dir)
        self.animal_classes = [d.name for d in self.data_dir.iterdir() if d.is_dir()]
        self.num_classes = len(self.animal_classes)
        self.input_shape = (64, 64)  # Small size for fast processing
        
        # Create class label mapping
        self.translation_map = {
            "cane": "dog", "cavallo": "horse", "elefante": "elephant", 
            "farfalla": "butterfly", "gallina": "chicken", "gatto": "cat", 
            "mucca": "cow", "pecora": "sheep", "scoiattolo": "squirrel",
            "ragno": "spider"
        }
        
        self.class_labels = {}
        for i, class_name in enumerate(self.animal_classes):
            translated_name = self.translation_map.get(class_name, class_name)
            self.class_labels[i] = translated_name
        
        print(f"Found {self.num_classes} animal classes: {self.animal_classes}")
        print(f"Translated class labels: {self.class_labels}")
        
    def extract_basic_features(self, image_path):
        """Extract very basic features from image."""
        try:
            # Read image
            img = cv2.imread(str(image_path))
            if img is None:
                return None
                
            # Resize to small size
            img = cv2.resize(img, self.input_shape)
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Simple features: mean, std, and some basic statistics
            features = [
                float(gray.mean()),           # Mean intensity
                float(gray.std()),            # Standard deviation
                float(gray.min()),            # Min intensity
                float(gray.max()),            # Max intensity
                float(img.shape[0]),          # Height
                float(img.shape[1]),          # Width
                float(img.size),              # Total pixels
            ]
            
            # Add some color channel statistics
            for channel in range(3):
                features.extend([
                    float(img[:,:,channel].mean()),
                    float(img[:,:,channel].std())
                ])
            
            return features
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None
    
    def load_sample_dataset(self, sample_per_class=100):
        """Load a sample of the dataset."""
        print("Loading sample dataset...")
        
        images = []
        labels = []
        class_counts = {}
        
        for class_idx, class_name in enumerate(self.animal_classes):
            class_path = self.data_dir / class_name
            if not class_path.is_dir():
                continue
                
            image_files = list(class_path.glob('*'))
            # Take a sample of images
            sampled_files = image_files[:sample_per_class] if len(image_files) > sample_per_class else image_files
            
            print(f"Processing {len(sampled_files)} images from {class_name}...")
            class_counts[class_name] = len(sampled_files)
            
            for img_path in sampled_files:
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    features = self.extract_basic_features(img_path)
                    if features is not None:
                        images.append(features)
                        labels.append(class_idx)
        
        print(f"Loaded {len(images)} images with features")
        print(f"Class distribution: {class_counts}")
        return images, labels
    
    def simple_classifier(self, X_train, y_train, X_test):
        """Simple distance-based classifier."""
        predictions = []
        
        for test_features in X_test:
            # Find the closest training sample using Euclidean distance
            min_distance = float('inf')
            predicted_class = 0
            
            for i, train_features in enumerate(X_train):
                # Calculate Euclidean distance
                distance = sum((a - b) ** 2 for a, b in zip(test_features, train_features)) ** 0.5
                if distance < min_distance:
                    min_distance = distance
                    predicted_class = y_train[i]
            
            predictions.append(predicted_class)
        
        return predictions
    
    def train_and_evaluate(self):
        """Train and evaluate the simple classifier."""
        print("=" * 60)
        print("SIMPLE ANIMAL IMAGE CLASSIFICATION TRAINING")
        print("=" * 60)
        
        # Load dataset
        X, y = self.load_sample_dataset(sample_per_class=100)
        
        if len(X) == 0:
            print("No images found!")
            return
        
        # Simple train/test split
        # Take first 80% as training, last 20% as testing
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        
        # Train simple classifier
        print("\nTraining simple distance-based classifier...")
        predictions = self.simple_classifier(X_train, y_train, X_test)
        
        # Calculate accuracy
        correct = sum(1 for pred, true in zip(predictions, y_test) if pred == true)
        accuracy = correct / len(y_test)
        
        print(f"\nResults:")
        print(f"Accuracy: {accuracy:.4f} ({correct}/{len(y_test)} correct)")
        
        # Per-class accuracy
        class_correct = Counter()
        class_total = Counter()
        
        for pred, true in zip(predictions, y_test):
            class_total[true] += 1
            if pred == true:
                class_correct[true] += 1
        
        print(f"\nPer-class accuracy:")
        for class_idx in range(self.num_classes):
            class_name = self.class_labels[class_idx]
            total = class_total[class_idx]
            correct = class_correct[class_idx]
            if total > 0:
                acc = correct / total
                print(f"  {class_name}: {acc:.3f} ({correct}/{total})")
        
        # Save results
        results = {
            'dataset_info': {
                'total_images': len(X),
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'classes': list(self.class_labels.values())
            },
            'results': {
                'overall_accuracy': accuracy,
                'per_class_accuracy': {
                    self.class_labels[class_idx]: class_correct[class_idx] / class_total[class_idx] 
                    for class_idx in range(self.num_classes) 
                    if class_total[class_idx] > 0
                }
            }
        }
        
        # Save to file
        os.makedirs('models', exist_ok=True)
        with open('models/simple_animal_classifier_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to models/simple_animal_classifier_results.json")
        
        return results

def main():
    """Main function."""
    print("🐾 Starting Simple Animal Image Classification Training 🐾")
    
    classifier = SimpleAnimalClassifier()
    results = classifier.train_and_evaluate()
    
    print("\n🎉 Simple training completed!")
    print("Results have been saved to the models directory.")

if __name__ == "__main__":
    main()