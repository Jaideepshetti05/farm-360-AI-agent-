import os
import shutil
import json
import random
from datetime import datetime
from PIL import Image

# Directories
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW_TRAIN_DIR = os.path.join(PROJECT_ROOT, "machine_learning", "data", "crop", "images", "Train")
RAW_VAL_DIR = os.path.join(PROJECT_ROOT, "machine_learning", "data", "crop", "images", "Validation")

V2_BASE_DIR = os.path.join(PROJECT_ROOT, "machine_learning", "data", "crop_disease", "v2")
CLEAN_TRAIN_DIR = os.path.join(V2_BASE_DIR, "clean", "train")
CLEAN_VAL_DIR = os.path.join(V2_BASE_DIR, "clean", "validation")
CLEAN_TEST_DIR = os.path.join(V2_BASE_DIR, "clean", "test")
REPORTS_DIR = os.path.join(V2_BASE_DIR, "reports")

CLASS_MAP = {
    "American Bollworm on Cotton": "American Bollworm on Cotton",
    "Anthracnose on Cotton": "Anthracnose on Cotton",
    "Army worm": "Armyworm",
    "bacterial_blight in Cotton": "Bacterial Blight on Cotton",
    "Bacterial Blight in cotton": "Bacterial Blight on Cotton",
    "Becterial Blight in Rice": "Bacterial Blight in Rice",
    "bollrot on Cotton": "Boll Rot on Cotton",
    "bollworm on Cotton": "Bollworm on Cotton",
    "Brownspot": "Brown Spot in Rice",
    "Common_Rust": "Common Rust in Maize",
    "Cotton Aphid": "Cotton Aphid",
    "cotton mealy bug": "Cotton Mealybug",
    "cotton whitefly": "Cotton Whitefly",
    "Flag Smut": "Flag Smut in Wheat",
    "Gray_Leaf_Spot": "Gray Leaf Spot in Maize",
    "Healthy cotton": "Healthy Cotton",
    "Healthy Maize": "Healthy Maize",
    "Healthy Wheat": "Healthy Wheat",
    "Leaf Curl": "Leaf Curl on Cotton",
    "Leaf smut": "Leaf Smut in Rice",
    "maize ear rot": "Maize Ear Rot",
    "maize fall armyworm": "Maize Fall Armyworm",
    "maize stem borer": "Maize Stem Borer",
    "Mosaic sugarcane": "Mosaic on Sugarcane",
    "pink bollworm in cotton": "Pink Bollworm on Cotton",
    "red cotton bug": "Red Cotton Bug",
    "RedRot sugarcane": "Red Rot on Sugarcane",
    "RedRust sugarcane": "Red Rust on Sugarcane",
    "Rice Blast": "Rice Blast",
    "Sugarcane Healthy": "Healthy Sugarcane",
    "thirps on  cotton": "Thrips on Cotton",
    "Tungro": "Tungro in Rice",
    "Wheat aphid": "Wheat Aphid",
    "Wheat black rust": "Wheat Black Rust",
    "Wheat Brown leaf Rust": "Wheat Brown Leaf Rust",
    "Wheat Brown leaf rust": "Wheat Brown Leaf Rust",
    "Wheat leaf blight": "Wheat Leaf Blight",
    "Wheat mite": "Wheat Mite",
    "Wheat powdery mildew": "Wheat Powdery Mildew",
    "Wheat scab": "Wheat Scab",
    "Wheat Stem fly": "Wheat Stem Fly",
    "Wheat___Yellow_Rust": "Wheat Yellow Rust",
    "Wilt": "Wilt on Cotton",
    "Yellow Rust Sugarcane": "Yellow Rust on Sugarcane"
}

def calculate_ahash(image_path):
    """Calculate an 8x8 Average Hash (aHash) of the image using PIL."""
    try:
        with Image.open(image_path) as img:
            # Resize, grayscale, and get pixels
            img = img.resize((8, 8), Image.Resampling.BILINEAR).convert('L')
            pixels = list(img.getdata())
            avg = sum(pixels) / 64.0
            bits = "".join(["1" if p > avg else "0" for p in pixels])
            return hex(int(bits, 2))
    except Exception:
        return None

def main():
    print("==================================================")
    print("🧹 Phase 2.5: Dataset Reconstruction & Stratified Splitting")
    print("==================================================")
    
    # 1. Clean existing clean splits
    for clean_dir in [CLEAN_TRAIN_DIR, CLEAN_VAL_DIR, CLEAN_TEST_DIR]:
        if os.path.exists(clean_dir):
            print(f"Removing old split directory: {clean_dir}")
            shutil.rmtree(clean_dir)
            
    os.makedirs(REPORTS_DIR, exist_ok=True)
    
    # 2. Collect all raw images
    all_raw_files = [] # list of dicts: {"path": ..., "canonical": ...}
    
    for split_dir in [RAW_TRAIN_DIR, RAW_VAL_DIR]:
        if not os.path.exists(split_dir):
            print(f"❌ Raw split directory not found: {split_dir}")
            continue
        classes = [d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))]
        for c in classes:
            c_path = os.path.join(split_dir, c)
            canonical = CLASS_MAP.get(c, c)
            files = [f for f in os.listdir(c_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp'))]
            for f in files:
                all_raw_files.append({
                    "path": os.path.join(c_path, f),
                    "canonical": canonical
                })
                
    total_raw = len(all_raw_files)
    print(f"✓ Collected {total_raw} raw images from splits.")
    
    # 3. Perform global deduplication and corruption verification
    hashes_db = set()
    clean_images = {} # canonical_name -> list of file paths
    duplicates_removed = 0
    corrupted_removed = 0
    
    for item in all_raw_files:
        src_path = item["path"]
        canonical = item["canonical"]
        
        # Verify corruption
        try:
            with Image.open(src_path) as img:
                img.verify()
        except Exception:
            corrupted_removed += 1
            continue
            
        # Verify duplicates
        ahash = calculate_ahash(src_path)
        if ahash is None:
            corrupted_removed += 1
            continue
            
        if ahash in hashes_db:
            duplicates_removed += 1
            continue
            
        hashes_db.add(ahash)
        clean_images.setdefault(canonical, []).append(src_path)
        
    total_clean = sum(len(paths) for paths in clean_images.values())
    print(f"✓ Global deduplication complete.")
    print(f"  - Total clean: {total_clean}")
    print(f"  - Duplicates removed: {duplicates_removed}")
    print(f"  - Corrupt removed: {corrupted_removed}")
    
    # 4. Generate stratified splits (80/10/10) with seed 42
    rng = random.Random(42)
    
    split_info = {
        "train": {"total": 0, "classes": {}},
        "validation": {"total": 0, "classes": {}},
        "test": {"total": 0, "classes": {}}
    }
    
    for canonical, paths in clean_images.items():
        # Sort to ensure deterministic split order
        paths.sort()
        rng.shuffle(paths)
        
        total = len(paths)
        train_cnt = int(total * 0.8)
        val_cnt = int(total * 0.1)
        test_cnt = total - train_cnt - val_cnt
        
        # Ensure at least 1 image in each split if total >= 3
        if total >= 3:
            if train_cnt == 0: train_cnt = 1
            if val_cnt == 0: val_cnt = 1
            if test_cnt == 0: test_cnt = 1
            diff = total - (train_cnt + val_cnt + test_cnt)
            train_cnt += diff
            
        # Split lists
        train_paths = paths[:train_cnt]
        val_paths = paths[train_cnt:train_cnt + val_cnt]
        test_paths = paths[train_cnt + val_cnt:]
        
        # Write files using hardlinks
        for name, file_paths, dest_dir in [("train", train_paths, CLEAN_TRAIN_DIR), 
                                           ("validation", val_paths, CLEAN_VAL_DIR), 
                                           ("test", test_paths, CLEAN_TEST_DIR)]:
            class_dest_dir = os.path.join(dest_dir, canonical)
            os.makedirs(class_dest_dir, exist_ok=True)
            
            for src in file_paths:
                f_name = os.path.basename(src)
                dest_path = os.path.join(class_dest_dir, f_name)
                try:
                    if os.path.exists(dest_path):
                        os.remove(dest_path)
                    os.link(src, dest_path)
                except OSError:
                    shutil.copy2(src, dest_path)
                    
            split_info[name]["total"] += len(file_paths)
            split_info[name]["classes"][canonical] = len(file_paths)
            
    # 5. Create final manifest structure
    manifest = {
        "dataset_version": "v2.5",
        "cleaning_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "totals": {
            "raw_images_processed": total_raw,
            "clean_images_saved": total_clean,
            "duplicates_removed": duplicates_removed,
            "corrupted_removed": corrupted_removed
        },
        "split_statistics": {
            "train": {
                "total_images": split_info["train"]["total"],
                "class_distribution": split_info["train"]["classes"]
            },
            "validation": {
                "total_images": split_info["validation"]["total"],
                "class_distribution": split_info["validation"]["classes"]
            },
            "test": {
                "total_images": split_info["test"]["total"],
                "class_distribution": split_info["test"]["classes"]
            }
        }
    }
    
    # Write manifest files
    manifest_path = os.path.join(REPORTS_DIR, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        
    with open(os.path.join(V2_BASE_DIR, "clean", "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        
    print(f"\n🎉 Clean reconstructed dataset and manifest generated successfully!")
    print(f"  - Train: {split_info['train']['total']} images")
    print(f"  - Validation: {split_info['validation']['total']} images")
    print(f"  - Test: {split_info['test']['total']} images")
    print(f"  - Manifest written to: {manifest_path}")

if __name__ == "__main__":
    main()
