import os
import sys

# Ensure backend can be imported
base_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(base_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from backend.config import settings

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
    openrouter_key = settings.openrouter_api_key
    if not openrouter_key:
        print("   ⚠ OPENROUTER_API_KEY not set (will use deterministic fallback mode)")
    else:
        print(f"   ✓ OPENROUTER_API_KEY configured (starts with: {openrouter_key[:12]}...)")
    
    farm360_key = settings.farm360_api_key
    if not farm360_key:
        print("   ⚠ FARM360_API_KEY not set")
    else:
        print(f"   ✓ FARM360_API_KEY configured")
    
    # Check model files
    print("\n3. Checking model files...")
    
    models_to_check = [
        ("Crop Regression", os.path.join(settings.model_base_path, settings.crop_model_path)),
        ("Dairy Intelligence", os.path.join(settings.model_base_path, settings.dairy_model_path)),
        ("Animal Disease RF", os.path.join(settings.model_base_path, settings.animal_model_dir, "RandomForest_Tuned.pkl")),
        ("Crop Vision", os.path.join(settings.model_base_path, settings.crop_vision_model_path)),
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
