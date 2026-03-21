"""
Quick Animal Detection Training Demo
Demonstrates multiple training approaches on a small sample
"""

import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime

def extract_simple_features(image_path):
    """Extract simple features from image"""
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Simple features
    features = [
        np.mean(image),           # Mean intensity
        np.std(image),            # Standard deviation
        np.mean(gray),            # Mean grayscale
        np.std(gray),             # Grayscale std
        image.shape[0],           # Height
        image.shape[1],           # Width
        np.sum(gray > 128),       # Bright pixels
        np.sum(gray < 64),        # Dark pixels
    ]
    
    return features

def main():
    print("🐄 QUICK ANIMAL DETECTION TRAINING DEMO 🐄")
    print("=" * 50)
    
    # Setup directories
    data_dir = "data/animal_detection"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = f"reports/animal_demo_{timestamp}"
    vis_dir = f"visualizations/animal_demo_{timestamp}"
    
    os.makedirs(report_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    
    # Load data
    print("Loading dataset...")
    csv_path = os.path.join(data_dir, "cows.csv")
    df = pd.read_csv(csv_path)
    
    # Process small sample for demo (first 10 images)
    sample_df = df.head(10)
    print(f"Processing {len(sample_df)} sample images...")
    
    features = []
    labels = []
    
    for idx, row in sample_df.iterrows():
        image_path = os.path.join(data_dir, row['image_name'])
        if os.path.exists(image_path):
            feature_vector = extract_simple_features(image_path)
            if feature_vector is not None:
                features.append(feature_vector)
                labels.append(1)  # All images contain cows in this dataset
    
    X = np.array(features)
    y = np.array(labels)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    print(f"Feature dimensions: {X.shape[1]}")
    
    # Train multiple models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42),
        'SVM': SVC(kernel='rbf', random_state=42, probability=True),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
    }
    
    results = {}
    
    print("\nTraining models...")
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        
        # Predictions
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        # Metrics
        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)
        
        results[name] = {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'predictions': test_pred.tolist(),
            'model': model
        }
        
        print(f"  Train Accuracy: {train_acc:.4f}")
        print(f"  Test Accuracy: {test_acc:.4f}")
    
    # Create ensemble
    print("\nCreating ensemble...")
    ensemble_predictions = []
    for i in range(len(X_test)):
        votes = [results[name]['predictions'][i] for name in models.keys()]
        ensemble_pred = 1 if sum(votes) > len(votes) / 2 else 0
        ensemble_predictions.append(ensemble_pred)
    
    ensemble_accuracy = accuracy_score(y_test, ensemble_predictions)
    results['Ensemble'] = {
        'test_accuracy': ensemble_accuracy
    }
    print(f"Ensemble Accuracy: {ensemble_accuracy:.4f}")
    
    # Generate reports
    print("\nGenerating reports...")
    
    # Summary report
    summary = {
        'timestamp': timestamp,
        'dataset_size': len(X),
        'feature_dimensions': X.shape[1],
        'models_trained': list(models.keys()) + ['Ensemble'],
        'results': {name: {'test_accuracy': res['test_accuracy']} for name, res in results.items()}
    }
    
    summary_path = os.path.join(report_dir, "demo_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Detailed reports
    for name, result in results.items():
        if name != 'Ensemble':
            report = f"""
# {name} Animal Detection Demo Report

## Performance
- Training Accuracy: {result['train_accuracy']:.4f}
- Test Accuracy: {result['test_accuracy']:.4f}

## Model Details
- Model Type: {name}
- Features Used: 8 (mean, std, shape, intensity statistics)
- Training Samples: {len(X_train)}
- Test Samples: {len(X_test)}
- Timestamp: {timestamp}
"""
            
            report_path = os.path.join(report_dir, f"{name.lower().replace(' ', '_')}_demo_report.txt")
            with open(report_path, 'w') as f:
                f.write(report)
    
    # Generate visualizations
    print("Generating visualizations...")
    
    # Performance comparison
    model_names = list(results.keys())
    accuracies = [results[name]['test_accuracy'] for name in model_names]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, accuracies, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    plt.title('Animal Detection Model Performance Comparison (Demo)', fontsize=14, fontweight='bold')
    plt.ylabel('Test Accuracy', fontsize=12)
    plt.ylim(0, 1.1)
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "demo_performance_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n" + "=" * 50)
    print("DEMO TRAINING COMPLETE!")
    print("=" * 50)
    print(f"Reports saved to: {report_dir}")
    print(f"Visualizations saved to: {vis_dir}")
    print("\nModel Performance Summary:")
    for name, result in results.items():
        print(f"- {name}: {result['test_accuracy']:.4f}")
    
    # Find best model
    best_model = max(results.items(), key=lambda x: x[1]['test_accuracy'])
    print(f"\nBest Model: {best_model[0]} ({best_model[1]['test_accuracy']:.4f})")

if __name__ == "__main__":
    main()