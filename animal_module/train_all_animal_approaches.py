"""
Master Animal Detection Training Script
Runs all possible training approaches for the animal detection dataset
"""

import os
import sys
import subprocess
import time
from datetime import datetime

def run_training_script(script_name, description):
    """Run a training script and handle errors"""
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"Script: {script_name}")
    print(f"{'='*60}")
    
    try:
        start_time = time.time()
        result = subprocess.run([
            sys.executable, 
            os.path.join("animal_module", script_name)
        ], capture_output=True, text=True, cwd=".")
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        if result.returncode == 0:
            print(f"✅ SUCCESS: {description}")
            print(f"Execution time: {execution_time:.2f} seconds")
            if result.stdout:
                print("Output:")
                print(result.stdout[-500:])  # Show last 500 characters
        else:
            print(f"❌ FAILED: {description}")
            print(f"Error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ ERROR running {description}: {e}")
        return False
    
    return True

def main():
    """Run all animal detection training approaches"""
    print("🐄 ANIMAL DETECTION MASTER TRAINING PIPELINE 🐄")
    print("=" * 60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("This will train the animal detection dataset using ALL possible approaches")
    print("=" * 60)
    
    # List of all training scripts with descriptions
    training_scripts = [
        ("animal_detection_pipeline.py", "Basic CNN and Transfer Learning Models"),
        ("advanced_detection_pipeline.py", "Advanced Object Detection Models (Faster R-CNN, SSD, RetinaNet)"),
        ("comprehensive_animal_training.py", "Traditional ML + Ensemble Approaches")
    ]
    
    results = []
    
    # Run each training approach
    for script_name, description in training_scripts:
        success = run_training_script(script_name, description)
        results.append((description, success))
        print()  # Add spacing
    
    # Summary
    print("=" * 60)
    print("FINAL TRAINING SUMMARY")
    print("=" * 60)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    successful = sum(1 for _, success in results if success)
    total = len(results)
    
    print(f"Successfully completed: {successful}/{total} training approaches")
    print()
    
    print("Training Approaches Summary:")
    for description, success in results:
        status = "✅ COMPLETED" if success else "❌ FAILED"
        print(f"  {status}: {description}")
    
    print()
    if successful == total:
        print("🎉 ALL TRAINING APPROACHES COMPLETED SUCCESSFULLY! 🎉")
        print("\nTrained models are available in the 'models/' directory")
        print("Reports and visualizations are in 'reports/' and 'visualizations/' directories")
    elif successful > 0:
        print(f"⚠️  {successful}/{total} approaches completed. Some training failed.")
        print("Check the error messages above for details.")
    else:
        print("❌ ALL TRAINING APPROACHES FAILED!")
        print("Please check your environment and dependencies.")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()