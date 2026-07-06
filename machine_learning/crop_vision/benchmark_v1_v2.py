import os
import time
import json
import argparse
import psutil
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

def get_memory_usage_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, default="", help="Path to the experiment run directory")
    args = parser.parse_args()

    experiments_base = os.path.abspath(os.path.join(os.path.dirname(__file__), "experiments"))
    if not args.run_dir:
        runs = [d for d in os.listdir(experiments_base) if d.startswith("run_") and os.path.isdir(os.path.join(experiments_base, d))]
        if not runs:
            print("❌ No runs found.")
            return
        runs.sort()
        args.run_dir = os.path.join(experiments_base, runs[-1])

    print("==================================================")
    print("📊 Benchmarking V1 (ResNet18) vs V2 (EfficientNetV2-S)")
    print("==================================================")

    # 1. Load class lists
    v1_classes = [
        "Corn___Common_Rust", "Corn___Gray_Leaf_Spot", "Corn___Healthy",
        "Corn___Northern_Leaf_Blight", "Potato___Early_Blight", "Potato___Healthy",
        "Potato___Late_Blight", "Rice___Brown_Spot", "Rice___Healthy",
        "Rice___Leaf_Blast", "Rice___Neck_Blast", "Sugarcane_Bacterial Blight",
        "Sugarcane_Healthy", "Sugarcane_Red Rot", "Wheat___Brown_Rust",
        "Wheat___Healthy", "Wheat___Yellow_Rust"
    ]
    
    classes_json_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model_registry", "classes", "crop_disease_v2.json"))
    with open(classes_json_path, "r", encoding="utf-8") as f:
        v2_classes = json.load(f)

    # 2. Overlapping validation directories mapping
    # Maps raw folder name -> (V1 class list index, V2 class list index)
    overlap_mapping = {
        "Common_Rust": (v1_classes.index("Corn___Common_Rust"), v2_classes.index("Common_Rust")),
        "Gray_Leaf_Spot": (v1_classes.index("Corn___Gray_Leaf_Spot"), v2_classes.index("Gray_Leaf_Spot")),
        "Wheat___Yellow_Rust": (v1_classes.index("Wheat___Yellow_Rust"), v2_classes.index("Wheat___Yellow_Rust")),
    }

    # 3. Collect validation images
    val_base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "crop_disease", "v2", "clean", "validation"))
    test_samples = []
    
    for folder, (v1_idx, v2_idx) in overlap_mapping.items():
        folder_path = os.path.join(val_base, folder)
        if os.path.exists(folder_path):
            imgs = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            # Take up to 20 samples per class for quick benchmark
            for img_path in imgs[:20]:
                test_samples.append((img_path, v1_idx, v2_idx))

    print(f"✓ Collected {len(test_samples)} overlapping validation samples for direct comparison.")
    if not test_samples:
        print("❌ No test samples found. Verify validation directories exist.")
        return

    # 4. Load Models
    v1_weights = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "crop_vision", "models", "crop_disease_model.pth"))
    v2_weights = os.path.join(args.run_dir, "weights.pth")

    # Load V1
    model_v1 = models.resnet18()
    model_v1.fc = nn.Linear(model_v1.fc.in_features, 17)
    if os.path.exists(v1_weights):
        model_v1.load_state_dict(torch.load(v1_weights, map_location=torch.device("cpu"), weights_only=True))
    model_v1.eval()
    
    # Load V2
    model_v2 = models.efficientnet_v2_s()
    num_features = model_v2.classifier[1].in_features
    model_v2.classifier[1] = nn.Linear(num_features, len(v2_classes))
    if os.path.exists(v2_weights):
        model_v2.load_state_dict(torch.load(v2_weights, map_location=torch.device("cpu")))
    model_v2.eval()

    # Preprocessing transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 5. Run Benchmark
    # V1 evaluation
    v1_correct = 0
    v1_latencies = []
    mem_before_v1 = get_memory_usage_mb()
    
    with torch.no_grad():
        for path, v1_label, _ in test_samples:
            img = Image.open(path).convert("RGB")
            tensor = transform(img).unsqueeze(0)
            
            t0 = time.perf_counter()
            out = model_v1(tensor)
            t1 = time.perf_counter()
            
            v1_latencies.append((t1 - t0) * 1000)
            _, pred = torch.max(out, 1)
            if pred.item() == v1_label:
                v1_correct += 1
                
    mem_after_v1 = get_memory_usage_mb()
    v1_acc = (v1_correct / len(test_samples)) * 100
    v1_avg_lat = sum(v1_latencies) / len(v1_latencies)

    # V2 evaluation
    v2_correct = 0
    v2_latencies = []
    mem_before_v2 = get_memory_usage_mb()
    
    with torch.no_grad():
        for path, _, v2_label in test_samples:
            img = Image.open(path).convert("RGB")
            tensor = transform(img).unsqueeze(0)
            
            t0 = time.perf_counter()
            out = model_v2(tensor)
            t1 = time.perf_counter()
            
            v2_latencies.append((t1 - t0) * 1000)
            _, pred = torch.max(out, 1)
            if pred.item() == v2_label:
                v2_correct += 1
                
    mem_after_v2 = get_memory_usage_mb()
    v2_acc = (v2_correct / len(test_samples)) * 100
    v2_avg_lat = sum(v2_latencies) / len(v2_latencies)

    results = {
        "benchmark_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "samples_count": len(test_samples),
        "v1_resnet18": {
            "model_size_mb": os.path.getsize(v1_weights)/(1024*1024) if os.path.exists(v1_weights) else 0.0,
            "accuracy_percent": round(v1_acc, 2),
            "avg_latency_ms": round(v1_avg_lat, 2),
            "memory_usage_mb": round(mem_after_v1 - mem_before_v1, 2)
        },
        "v2_efficientnetv2_s": {
            "model_size_mb": os.path.getsize(v2_weights)/(1024*1024) if os.path.exists(v2_weights) else 0.0,
            "accuracy_percent": round(v2_acc, 2),
            "avg_latency_ms": round(v2_avg_lat, 2),
            "memory_usage_mb": round(mem_after_v2 - mem_before_v2, 2)
        }
    }

    # Print results
    print("\n--------------------------------------------------")
    print("BENCHMARK COMPARISON RESULTS:")
    print("--------------------------------------------------")
    print(f"Model V1 (ResNet18):")
    print(f"  Size: {results['v1_resnet18']['model_size_mb']:.2f} MB")
    print(f"  Accuracy (overlapping classes): {results['v1_resnet18']['accuracy_percent']}%")
    print(f"  Avg Latency: {results['v1_resnet18']['avg_latency_ms']} ms")
    print(f"Model V2 (EfficientNetV2-S):")
    print(f"  Size: {results['v2_efficientnetv2_s']['model_size_mb']:.2f} MB")
    print(f"  Accuracy (overlapping classes): {results['v2_efficientnetv2_s']['accuracy_percent']}%")
    print(f"  Avg Latency: {results['v2_efficientnetv2_s']['avg_latency_ms']} ms")
    print("--------------------------------------------------")

    # Save to JSON
    out_path = os.path.join(args.run_dir, "benchmark_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"✓ Benchmark results written to: {out_path}")

if __name__ == "__main__":
    main()
