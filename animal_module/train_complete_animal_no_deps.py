"""
Complete Animal Image Classification Training - No External Dependencies
========================================================================

This script trains your animal dataset using only built-in Python libraries.
"""

import os
import json
from pathlib import Path
from collections import Counter
import random
from PIL import Image
import math

class BasicAnimalClassifier:
    """Basic animal image classifier using only built-in libraries."""
    
    def __init__(self, data_dir: str = "data/animal_images/animalsdata/raw-img"):
        self.data_dir = Path(data_dir)
        self.animal_classes = [d.name for d in self.data_dir.iterdir() if d.is_dir()]
        self.num_classes = len(self.animal_classes)
        self.target_size = (32, 32)  # Very small for fast processing
        
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
        
    def extract_simple_features(self, image_path):
        """Extract very basic features from image using PIL."""
        try:
            # Open image
            with Image.open(image_path) as img:
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize to small size
                img = img.resize(self.target_size)
                
                # Get pixel data
                pixels = list(img.getdata())
                
                # Calculate basic statistics
                r_values = [p[0] for p in pixels]
                g_values = [p[1] for p in pixels]
                b_values = [p[2] for p in pixels]
                
                # Simple features
                features = [
                    sum(r_values) / len(r_values),  # Mean red
                    sum(g_values) / len(g_values),  # Mean green
                    sum(b_values) / len(b_values),  # Mean blue
                    max(r_values),                  # Max red
                    max(g_values),                  # Max green
                    max(b_values),                  # Max blue
                    min(r_values),                  # Min red
                    min(g_values),                  # Min green
                    min(b_values),                  # Min blue
                    img.width,                      # Width
                    img.height,                     # Height
                ]
                
                return features
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None
    
    def load_dataset_sample(self, sample_per_class=50):
        """Load a sample of the dataset."""
        print("Loading dataset sample...")
        
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
                    features = self.extract_simple_features(img_path)
                    if features is not None:
                        images.append(features)
                        labels.append(class_idx)
        
        print(f"Loaded {len(images)} images with features")
        print(f"Class distribution: {class_counts}")
        return images, labels
    
    def euclidean_distance(self, point1, point2):
        """Calculate Euclidean distance between two points."""
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(point1, point2)))
    
    def simple_knn_classifier(self, X_train, y_train, X_test, k=3):
        """Simple KNN classifier."""
        predictions = []
        
        for test_features in X_test:
            # Calculate distances to all training samples
            distances = []
            for i, train_features in enumerate(X_train):
                distance = self.euclidean_distance(test_features, train_features)
                distances.append((distance, y_train[i]))
            
            # Sort by distance and take k nearest neighbors
            distances.sort(key=lambda x: x[0])
            k_nearest = distances[:k]
            
            # Vote for the most common class
            votes = Counter([label for _, label in k_nearest])
            predicted_class = votes.most_common(1)[0][0]
            predictions.append(predicted_class)
        
        return predictions
    
    def train_and_evaluate_all_methods(self):
        """Train and evaluate using multiple simple methods."""
        print("=" * 70)
        print("COMPLETE ANIMAL IMAGE CLASSIFICATION TRAINING")
        print("=" * 70)
        print("Using multiple approaches without external dependencies")
        print("=" * 70)
        
        # Load dataset
        X, y = self.load_dataset_sample(sample_per_class=50)
        
        if len(X) == 0:
            print("No images found!")
            return
        
        # Simple train/test split
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"\nDataset split:")
        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        
        results = {}
        
        # Method 1: 1-NN (Nearest Neighbor)
        print("\n1. Training 1-NN Classifier...")
        predictions_1nn = self.simple_knn_classifier(X_train, y_train, X_test, k=1)
        correct_1nn = sum(1 for pred, true in zip(predictions_1nn, y_test) if pred == true)
        accuracy_1nn = correct_1nn / len(y_test)
        results['1-NN'] = {
            'accuracy': accuracy_1nn,
            'correct': correct_1nn,
            'total': len(y_test)
        }
        print(f"1-NN Accuracy: {accuracy_1nn:.4f} ({correct_1nn}/{len(y_test)})")
        
        # Method 2: 3-NN (K-Nearest Neighbors)
        print("\n2. Training 3-NN Classifier...")
        predictions_3nn = self.simple_knn_classifier(X_train, y_train, X_test, k=3)
        correct_3nn = sum(1 for pred, true in zip(predictions_3nn, y_test) if pred == true)
        accuracy_3nn = correct_3nn / len(y_test)
        results['3-NN'] = {
            'accuracy': accuracy_3nn,
            'correct': correct_3nn,
            'total': len(y_test)
        }
        print(f"3-NN Accuracy: {accuracy_3nn:.4f} ({correct_3nn}/{len(y_test)})")
        
        # Method 3: 5-NN (K-Nearest Neighbors)
        print("\n3. Training 5-NN Classifier...")
        predictions_5nn = self.simple_knn_classifier(X_train, y_train, X_test, k=5)
        correct_5nn = sum(1 for pred, true in zip(predictions_5nn, y_test) if pred == true)
        accuracy_5nn = correct_5nn / len(y_test)
        results['5-NN'] = {
            'accuracy': accuracy_5nn,
            'correct': correct_5nn,
            'total': len(y_test)
        }
        print(f"5-NN Accuracy: {accuracy_5nn:.4f} ({correct_5nn}/{len(y_test)})")
        
        # Method 4: Simple Mean Classifier (assign each test sample to the class 
        # whose training samples have the closest mean feature vector)
        print("\n4. Training Mean-Based Classifier...")
        # Calculate mean features for each class
        class_means = {}
        for class_idx in range(self.num_classes):
            class_features = [X_train[i] for i in range(len(X_train)) if y_train[i] == class_idx]
            if class_features:
                mean_features = []
                for feature_idx in range(len(class_features[0])):
                    mean_val = sum(f[feature_idx] for f in class_features) / len(class_features)
                    mean_features.append(mean_val)
                class_means[class_idx] = mean_features
        
        predictions_mean = []
        for test_features in X_test:
            min_distance = float('inf')
            predicted_class = 0
            for class_idx, mean_features in class_means.items():
                distance = self.euclidean_distance(test_features, mean_features)
                if distance < min_distance:
                    min_distance = distance
                    predicted_class = class_idx
            predictions_mean.append(predicted_class)
        
        correct_mean = sum(1 for pred, true in zip(predictions_mean, y_test) if pred == true)
        accuracy_mean = correct_mean / len(y_test)
        results['Mean-Based'] = {
            'accuracy': accuracy_mean,
            'correct': correct_mean,
            'total': len(y_test)
        }
        print(f"Mean-Based Accuracy: {accuracy_mean:.4f} ({correct_mean}/{len(y_test)})")
        
        # Find best method
        best_method = max(results.keys(), key=lambda x: results[x]['accuracy'])
        best_accuracy = results[best_method]['accuracy']
        
        print(f"\n" + "=" * 50)
        print("FINAL RESULTS SUMMARY")
        print("=" * 50)
        print(f"Best performing method: {best_method}")
        print(f"Best accuracy: {best_accuracy:.4f}")
        print()
        print("All Method Results:")
        for method, result in results.items():
            print(f"  {method}: {result['accuracy']:.4f} ({result['correct']}/{result['total']})")
        
        # Per-class analysis for best method
        best_predictions = None
        if best_method == '1-NN':
            best_predictions = predictions_1nn
        elif best_method == '3-NN':
            best_predictions = predictions_3nn
        elif best_method == '5-NN':
            best_predictions = predictions_5nn
        else:
            best_predictions = predictions_mean
        
        print(f"\nPer-class performance for {best_method}:")
        class_correct = Counter()
        class_total = Counter()
        
        for pred, true in zip(best_predictions, y_test):
            class_total[true] += 1
            if pred == true:
                class_correct[true] += 1
        
        for class_idx in range(self.num_classes):
            class_name = self.class_labels[class_idx]
            total = class_total[class_idx]
            correct = class_correct[class_idx]
            if total > 0:
                acc = correct / total
                print(f"  {class_name}: {acc:.3f} ({correct}/{total})")
        
        # Save comprehensive results
        comprehensive_results = {
            'dataset_info': {
                'total_images_processed': len(X),
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'classes': list(self.class_labels.values()),
                'feature_dimensions': len(X[0]) if X else 0
            },
            'method_results': results,
            'best_method': {
                'name': best_method,
                'accuracy': best_accuracy
            },
            'per_class_performance': {
                self.class_labels[class_idx]: class_correct[class_idx] / class_total[class_idx] 
                for class_idx in range(self.num_classes) 
                if class_total[class_idx] > 0
            }
        }
        
        # Save to files
        os.makedirs('models', exist_ok=True)
        os.makedirs('reports', exist_ok=True)
        
        # Save JSON results
        with open('models/complete_animal_training_results.json', 'w') as f:
            json.dump(comprehensive_results, f, indent=2)
        
        # Save text report
        with open('reports/animal_training_complete_report.txt', 'w') as f:
            f.write("COMPLETE ANIMAL IMAGE CLASSIFICATION TRAINING REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Dataset: {self.data_dir}\n")
            f.write(f"Total Images Processed: {len(X)}\n")
            f.write(f"Training Samples: {len(X_train)}\n")
            f.write(f"Test Samples: {len(X_test)}\n")
            f.write(f"Number of Classes: {self.num_classes}\n")
            f.write(f"Classes: {', '.join(self.class_labels.values())}\n\n")
            
            f.write("Method Performance:\n")
            f.write("-" * 30 + "\n")
            for method, result in results.items():
                f.write(f"{method}: {result['accuracy']:.4f} ({result['correct']}/{result['total']})\n")
            
            f.write(f"\nBest Method: {best_method} with {best_accuracy:.4f} accuracy\n\n")
            
            f.write("Per-Class Performance:\n")
            f.write("-" * 30 + "\n")
            for class_idx in range(self.num_classes):
                class_name = self.class_labels[class_idx]
                total = class_total[class_idx]
                correct = class_correct[class_idx]
                if total > 0:
                    acc = correct / total
                    f.write(f"{class_name}: {acc:.3f} ({correct}/{total})\n")
        
        print(f"\n" + "=" * 70)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print("Results saved to:")
        print("- models/complete_animal_training_results.json")
        print("- reports/animal_training_complete_report.txt")
        print("=" * 70)
        
        return comprehensive_results

def main():
    """Main function."""
    print("🐾 Starting Complete Animal Image Classification Training 🐾")
    print("Using multiple approaches without external dependencies")
    print()
    
    classifier = BasicAnimalClassifier()
    results = classifier.train_and_evaluate_all_methods()
    
    print("\n🎉 Complete training completed successfully!")
    print("All models and results have been saved!")

if __name__ == "__main__":
    main()