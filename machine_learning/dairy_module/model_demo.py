"""
Agricultural AI System - Model Usage Demo
=========================================

Demonstration of how to load and use trained models from the Agricultural AI System.

This script shows:
1. How to load trained models
2. How to make predictions
3. How to access model metadata
"""

import pickle
import json
import pandas as pd
import numpy as np
from pathlib import Path

def load_trained_model(model_path: str, metadata_path: str):
    """
    Load a trained model and its metadata.
    
    Args:
        model_path (str): Path to the saved model .pkl file
        metadata_path (str): Path to the metadata .json file
        
    Returns:
        Tuple of (model, metadata)
    """
    # Load model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    return model, metadata

def demonstrate_milk_yield_prediction():
    """Demonstrate milk yield prediction using the trained model."""
    print("=" * 60)
    print("MILK YIELD PREDICTION DEMO")
    print("=" * 60)
    
    # Load the best performing model (Random Forest for milk yield)
    model_path = "models/agri_ai_regression_v1_20260217_215213.pkl"
    metadata_path = "models/metadata_regression_v1_20260217_215213.json"
    
    try:
        model, metadata = load_trained_model(model_path, metadata_path)
        
        print(f"Model: {metadata['model_name']}")
        print(f"Dataset: {metadata['dataset_name']}")
        print(f"Performance: CV R² = {metadata['metrics']['cv_score_mean']:.4f}")
        print(f"Features: {metadata['data_shape']['features']}")
        print()
        
        # Example prediction - using some sample feature values
        # Note: In real usage, you'd need to provide all 35 features
        print("Example prediction:")
        print("(Note: This is a simplified example with dummy data)")
        
        # For demonstration, let's create sample data with correct feature names
        sample_data = pd.DataFrame({
            'Age_Months': [48],
            'Weight_kg': [500],
            'Days_in_Milk': [100],
            'Feed_Quantity_kg': [15],
            'Water_Intake_L': [60],
            'Ambient_Temperature_C': [25],
            'Humidity_percent': [60],
            'Housing_Score': [0.7],
            'FMD_Vaccine': [1],
            'health_status': [3]  # Scaled value
        })
        
        # In practice, you'd need all 35 features. This is just for demonstration.
        print(f"Sample input shape: {sample_data.shape}")
        print("Prediction would require all 35 features from the training data")
        
    except FileNotFoundError:
        print("Model files not found. Please run agricultural_ai_system.py first.")
    except Exception as e:
        print(f"Error: {str(e)}")

def demonstrate_simple_milk_production():
    """Demonstrate simple milk production prediction using smaller dataset model."""
    print("\n" + "=" * 60)
    print("SIMPLE MILK PRODUCTION PREDICTION DEMO")
    print("=" * 60)
    
    # Load model trained on Mik_Pro.csv (yearly milk production)
    model_path = "models/agri_ai_regression_v1_20260217_215218.pkl"
    metadata_path = "models/metadata_regression_v1_20260217_215218.json"
    
    try:
        model, metadata = load_trained_model(model_path, metadata_path)
        
        print(f"Model: {metadata['model_name']}")
        print(f"Dataset: {metadata['dataset_name']}")
        print(f"Performance: CV R² = {metadata['metrics']['cv_score_mean']:.4f}")
        print(f"Features: {', '.join(metadata['feature_names'])}")
        print()
        
        # Make predictions for future years
        future_years = np.array([[2023], [2024], [2025], [2026]])
        
        predictions = model.predict(future_years)
        
        print("Future Milk Production Predictions:")
        print("-" * 40)
        for year, prediction in zip(future_years.flatten(), predictions):
            print(f"Year {year}: {prediction:.0f} units")
            
    except FileNotFoundError:
        print("Model files not found. Please run agricultural_ai_system.py first.")
    except Exception as e:
        print(f"Error: {str(e)}")

def list_available_models():
    """List all available trained models."""
    print("\n" + "=" * 60)
    print("AVAILABLE TRAINED MODELS")
    print("=" * 60)
    
    models_dir = Path("models")
    if not models_dir.exists():
        print("No models directory found.")
        return
    
    # Find all model files
    model_files = list(models_dir.glob("agri_ai_*.pkl"))
    metadata_files = list(models_dir.glob("metadata_*.json"))
    
    if not model_files:
        print("No trained models found.")
        return
    
    print(f"Found {len(model_files)} trained models:")
    print()
    
    for model_file in model_files:
        # Find corresponding metadata
        model_name = model_file.stem
        metadata_file = models_dir / f"metadata_{model_name.split('_', 2)[2]}.json"
        
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                print(f"Model: {metadata['model_name']}")
                print(f"  Dataset: {metadata['dataset_name']}")
                print(f"  Task: {metadata['task_type']}")
                print(f"  Performance: {metadata['metrics']['cv_score_mean']:.4f}")
                print(f"  Features: {metadata['data_shape']['features']}")
                print(f"  File: {model_file.name}")
                print()
                
            except Exception as e:
                print(f"  Error reading metadata for {model_file.name}: {str(e)}")
        else:
            print(f"  {model_file.name} (no metadata)")

if __name__ == "__main__":
    print("AGRICULTURAL AI SYSTEM - MODEL DEMONSTRATION")
    print("=" * 60)
    
    # List available models
    list_available_models()
    
    # Demonstrate predictions
    demonstrate_simple_milk_production()
    demonstrate_milk_yield_prediction()
    
    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)
    print("To train more models, run: python agricultural_ai_system.py")
