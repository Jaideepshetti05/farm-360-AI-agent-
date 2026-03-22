import os
import sys

def verify_setup():
    """Verify all models and dependencies before running"""
    print("="*60)
    print("Farm360 Pre-flight Verification")
    print("="*60)
    
    errors = []
    
    # Check Python imports
    print("\n1. Checking Python imports...")
    try:
        import google.generativeai as genai
        print("   ✓ google.generativeai imported successfully")
    except ImportError as e:
        errors.append(f"google.generativeai import failed: {e}")
        print(f"   ✗ {e}")
    
    try:
        import torch
        print("   ✓ torch imported successfully")
    except ImportError as e:
        errors.append(f"torch import failed: {e}")
        print(f"   ✗ {e}")
    
    try:
        from loguru import logger
        print("   ✓ loguru imported successfully")
    except ImportError as e:
        errors.append(f"loguru import failed: {e}")
        print(f"   ✗ {e}")
    
    # Check environment variables
    print("\n2. Checking environment variables...")
    google_key = os.getenv("GOOGLE_API_KEY", "NOT_SET")
    if google_key == "NOT_SET":
        print("   ⚠ GOOGLE_API_KEY not set (will use deterministic mode)")
    elif google_key == "your_actual_google_gemini_api_key_here":
        print("   ⚠ GOOGLE_API_KEY has default value (will use deterministic mode)")
    else:
        print("   ✓ GOOGLE_API_KEY configured")
    
    farm360_key = os.getenv("FARM360_API_KEY", "NOT_SET")
    if farm360_key == "NOT_SET":
        print("   ⚠ FARM360_API_KEY not set")
    else:
        print("   ✓ FARM360_API_KEY configured")
    
    # Check model files
    print("\n3. Checking model files...")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(base_dir)
    
    models_to_check = [
        ("Crop Regression", os.path.join(parent_dir, "crop_regression", "models", "production_model_log.pkl")),
        ("Dairy Intelligence", os.path.join(parent_dir, "models", "dairy_intelligence_v1_20260217_210257.pkl")),
        ("Animal Disease RF", os.path.join(parent_dir, "models", "animal_disease_20260218_215356", "RandomForest_Tuned.pkl")),
        ("Crop Vision", os.path.join(parent_dir, "crop_vision", "models", "crop_disease_model.pth")),
    ]
    
    for model_name, model_path in models_to_check:
        if os.path.exists(model_path):
            size_mb = os.path.getsize(model_path) / (1024 * 1024)
            print(f"   ✓ {model_name}: {size_mb:.2f} MB")
        else:
            errors.append(f"{model_name} not found at {model_path}")
            print(f"   ✗ {model_name}: NOT FOUND")
    
    # Summary
    print("\n" + "="*60)
    if errors:
        print("VERIFICATION FAILED")
        print("="*60)
        for error in errors:
            print(f"  • {error}")
        return False
    else:
        print("VERIFICATION PASSED ✓")
        print("="*60)
        print("All systems ready for Farm360 deployment!")
        return True

if __name__ == "__main__":
    success = verify_setup()
    sys.exit(0 if success else 1)
