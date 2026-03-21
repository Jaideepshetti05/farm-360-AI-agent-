"""
Final Animal Image Classification using Scikit-learn
===================================================

This script trains your animal dataset using scikit-learn approaches
on a representative sample to complete training quickly with proper error handling.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump
import warnings
warnings.filterwarnings('ignore')

class FinalAnimalClassifier:
    """Class to handle animal image classification using scikit-learn on a sample."""
    
    def __init__(self, data_dir: str = "data/animal_images/animalsdata/raw-img", max_samples_per_class: int = 500):
        self.data_dir = Path(data_dir)
        self.max_samples_per_class = max_samples_per_class
        self.animal_classes = [d.name for d in self.data_dir.iterdir() if d.is_dir()]
        self.num_classes = len(self.animal_classes)
        self.input_shape = (128, 128)  # Smaller size for faster processing
        self.models = {}
        self.results = {}
        
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
        print(f"Max samples per class: {self.max_samples_per_class}")
        print(f"Total classes: {self.num_classes}")
        
    def extract_features(self, image_path):
        """Extract features from an image using traditional computer vision techniques."""
        # Read image
        img = cv2.imread(str(image_path))
        if img is None:
            return None
            
        # Resize image to smaller dimensions for faster processing
        img = cv2.resize(img, self.input_shape)
        
        # Convert to different color spaces and extract features
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Statistical features
        mean_color = np.mean(img, axis=(0, 1))  # Mean color per channel
        std_color = np.std(img, axis=(0, 1))    # Std dev per channel
        gray_mean = np.mean(gray)
        gray_std = np.std(gray)
        
        # Edge features using Canny edge detection
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Simplified texture features
        lbp_hist = []
        for i in range(0, gray.shape[0]-4, 4):
            for j in range(0, gray.shape[1]-4, 4):
                patch = gray[i:i+4, j:j+4]
                lbp_hist.append(np.mean(patch))
        
        # Take only first 50 texture features to limit dimensionality
        lbp_features = lbp_hist[:50] if len(lbp_hist) > 50 else lbp_hist + [0]*(50-len(lbp_hist))
        
        # Histogram features
        hist_b, _ = np.histogram(img[:,:,0].flatten(), bins=16, range=(0, 256))
        hist_g, _ = np.histogram(img[:,:,1].flatten(), bins=16, range=(0, 256))
        hist_r, _ = np.histogram(img[:,:,2].flatten(), bins=16, range=(0, 256))
        
        # Combine all features
        features = np.concatenate([
            mean_color,
            std_color,
            [gray_mean, gray_std, edge_density],
            lbp_features,
            hist_b,
            hist_g,
            hist_r
        ])
        
        return features.astype(np.float32)
    
    def load_dataset(self):
        """Load and preprocess a sample of the dataset."""
        print("Loading dataset sample...")
        
        images = []
        labels = []
        
        for class_idx, class_name in enumerate(self.animal_classes):
            class_path = self.data_dir / class_name
            if not class_path.is_dir():
                continue
                
            image_files = list(class_path.glob('*'))
            # Take only a sample of the images to speed up processing
            sampled_files = image_files[:min(len(image_files), self.max_samples_per_class)]
            print(f"Processing {len(sampled_files)} images from {class_name}...")
            
            for img_path in sampled_files:
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    features = self.extract_features(img_path)
                    if features is not None:
                        images.append(features)
                        labels.append(class_idx)
        
        print(f"Loaded {len(images)} images with features")
        return np.array(images), np.array(labels)
    
    def train_classifiers(self, X_train, y_train, X_test, y_test):
        """Train multiple classifiers and evaluate them."""
        
        # Define classifiers with lighter parameters for faster training
        classifiers = {
            'RandomForest': RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42),
            'GradientBoosting': GradientBoostingClassifier(n_estimators=50, max_depth=5, random_state=42),
            'SVM': SVC(kernel='rbf', C=0.5, gamma='scale', random_state=42),
            'LogisticRegression': LogisticRegression(max_iter=500, random_state=42),
            'KNN': KNeighborsClassifier(n_neighbors=5),
            'NaiveBayes': GaussianNB()
        }
        
        # Train and evaluate each classifier
        for name, clf in classifiers.items():
            print(f"\nTraining {name}...")
            
            # Train the classifier
            clf.fit(X_train, y_train)
            
            # Make predictions
            y_pred = clf.predict(X_test)
            
            # Calculate accuracy
            accuracy = accuracy_score(y_test, y_pred)
            
            # Perform cross-validation with fewer folds for speed
            cv_scores = cross_val_score(clf, X_train, y_train, cv=3)
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
            
            # Store results
            self.models[name] = clf
            self.results[name] = {
                'accuracy': accuracy,
                'cv_mean': cv_mean,
                'cv_std': cv_std,
                'predictions': y_pred
            }
            
            print(f"{name} - Test Accuracy: {accuracy:.4f}, CV Score: {cv_mean:.4f} (+/- {cv_std*2:.4f})")
    
    def create_ensemble_model(self, X_train, y_train):
        """Create an ensemble model combining multiple classifiers."""
        print("\nCreating ensemble model...")
        
        # Define base classifiers with lighter parameters
        rf_clf = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42)
        gb_clf = GradientBoostingClassifier(n_estimators=50, max_depth=5, random_state=42)
        svm_clf = SVC(probability=True, kernel='rbf', C=0.5, gamma='scale', random_state=42)
        
        # Create voting classifier
        ensemble = VotingClassifier(
            estimators=[
                ('rf', rf_clf),
                ('gb', gb_clf),
                ('svm', svm_clf)
            ],
            voting='soft'
        )
        
        # Train ensemble
        ensemble.fit(X_train, y_train)
        
        # Store ensemble model
        self.models['Ensemble'] = ensemble
        
        return ensemble
    
    def plot_results(self):
        """Plot the results of different classifiers."""
        if not self.results:
            print("No results to plot.")
            return
        
        # Prepare data for plotting
        model_names = list(self.results.keys())
        accuracies = [self.results[name]['accuracy'] for name in model_names]
        cv_means = [self.results[name]['cv_mean'] for name in model_names]
        
        # Create bar plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x_pos = np.arange(len(model_names))
        width = 0.35
        
        bars1 = ax.bar(x_pos - width/2, accuracies, width, label='Test Accuracy', alpha=0.8)
        bars2 = ax.bar(x_pos + width/2, cv_means, width, label='Cross-Validation Score', alpha=0.8)
        
        ax.set_xlabel('Model')
        ax.set_ylabel('Accuracy')
        ax.set_title('Performance Comparison of Different Classifiers')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.legend()
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('animal_classification_results_final.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_models(self):
        """Save trained models to disk."""
        os.makedirs('models', exist_ok=True)
        
        for model_name, model in self.models.items():
            model_path = f'models/final_{model_name.lower().replace(" ", "_")}_animal_classifier.joblib'
            dump(model, model_path)
            print(f"Saved {model_name} model to {model_path}")
    
    def train_all_approaches(self):
        """Train using all possible scikit-learn approaches."""
        print("=" * 80)
        print("FINAL SCIKIT-LEARN ANIMAL IMAGE CLASSIFICATION TRAINING")
        print("=" * 80)
        print(f"Dataset: {self.data_dir}")
        print(f"Number of classes: {self.num_classes}")
        print(f"Max samples per class: {self.max_samples_per_class}")
        print(f"Classes: {list(self.class_labels.values())}")
        print("=" * 80)
        
        # Load dataset
        X, y = self.load_dataset()
        
        if len(X) == 0:
            print("No images found in the dataset!")
            return
        
        print(f"Feature vector shape: {X.shape}")
        
        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        
        # Train individual classifiers
        self.train_classifiers(X_train, y_train, X_test, y_test)
        
        # Create ensemble model
        ensemble_model = self.create_ensemble_model(X_train, y_train)
        
        # Evaluate ensemble
        y_pred_ensemble = ensemble_model.predict(X_test)
        ensemble_accuracy = accuracy_score(y_test, y_pred_ensemble)
        
        # Perform cross-validation for ensemble
        ensemble_cv_scores = cross_val_score(ensemble_model, X_train, y_train, cv=3)
        
        # Add ensemble to results
        self.results['Ensemble'] = {
            'accuracy': ensemble_accuracy,
            'cv_mean': ensemble_cv_scores.mean(),
            'cv_std': ensemble_cv_scores.std(),
            'predictions': y_pred_ensemble
        }
        
        print(f"\nEnsemble Test Accuracy: {ensemble_accuracy:.4f}")
        
        # Print detailed results for best model
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['accuracy'])
        best_accuracy = self.results[best_model_name]['accuracy']
        
        print(f"\nBest performing model: {best_model_name} (Accuracy: {best_accuracy:.4f})")
        
        # Show classification report for best model
        y_pred_best = self.results[best_model_name]['predictions']
        print(f"\nDetailed Classification Report for {best_model_name}:")
        print(classification_report(y_test, y_pred_best, 
                                  target_names=list(self.class_labels.values())))
        
        # Plot confusion matrix for best model
        cm = confusion_matrix(y_test, y_pred_best)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=list(self.class_labels.values()),
                   yticklabels=list(self.class_labels.values()))
        plt.title(f'Confusion Matrix - {best_model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('confusion_matrix_final.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Plot results
        self.plot_results()
        
        # Save models
        self.save_models()
        
        print("\n" + "=" * 80)
        print("FINAL SCIKIT-LEARN TRAINING COMPLETED!")
        print("=" * 80)
        print("Models have been saved to the 'models/' directory")
        print("Results visualization saved as 'animal_classification_results_final.png'")
        print("Confusion matrix saved as 'confusion_matrix_final.png'")
        
        return self.results


def main():
    """Main function to run final scikit-learn animal image classification training."""
    print("🐾 Starting Final Scikit-learn Animal Image Classification Training 🐾")
    
    # Initialize the classifier with the new dataset
    classifier = FinalAnimalClassifier(max_samples_per_class=300)  # Reduced sample size for faster training
    
    # Run the complete pipeline
    results = classifier.train_all_approaches()
    
    print("\n🎉 Final scikit-learn training completed successfully!")
    print("Models have been saved to the 'models/' directory")


if __name__ == "__main__":
    main()