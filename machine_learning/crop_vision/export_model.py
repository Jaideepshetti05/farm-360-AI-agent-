import os
import argparse
import torch
import torch.nn as nn
import torchvision.models as models

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, default="", help="Path to the experiment run directory")
    args = parser.parse_args()

    # If run_dir not specified, look for the latest run_XXX in experiments
    experiments_base = os.path.abspath(os.path.join(os.path.dirname(__file__), "experiments"))
    
    if not args.run_dir:
        if not os.path.exists(experiments_base):
            print("❌ Experiments folder does not exist.")
            return
        runs = [d for d in os.listdir(experiments_base) if d.startswith("run_") and os.path.isdir(os.path.join(experiments_base, d))]
        if not runs:
            print("❌ No experiment runs found.")
            return
        # Get latest run
        runs.sort()
        args.run_dir = os.path.join(experiments_base, runs[-1])

    print("==================================================")
    print(f"📦 Exporting Model from: {args.run_dir}")
    print("==================================================")

    weights_path = os.path.join(args.run_dir, "weights.pth")
    if not os.path.exists(weights_path):
        print(f"❌ Weights file {weights_path} not found.")
        return

    # 1. Rebuild model structure (42 classes)
    classes_json_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model_registry", "classes", "crop_disease_v2.json"))
    import json
    with open(classes_json_path, "r", encoding="utf-8") as f:
        classes = json.load(f)
    num_classes = len(classes)

    print(f"✓ Rebuilding EfficientNetV2-S with {num_classes} classes...")
    model = models.efficientnet_v2_s()
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, num_classes)

    # 2. Load state dict
    state_dict = torch.load(weights_path, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.eval()
    print("✓ Model weights loaded successfully.")

    # 3. Create dummy input tensor
    dummy_input = torch.randn(1, 3, 224, 224)

    # 4. Export to TorchScript
    print("Exporting to TorchScript...")
    try:
        traced_model = torch.jit.trace(model, dummy_input)
        ts_path = os.path.join(args.run_dir, "weights.pt")
        traced_model.save(ts_path)
        print(f"✓ TorchScript model saved: {ts_path} ({os.path.getsize(ts_path)/(1024*1024):.2f} MB)")
    except Exception as e:
        print(f"❌ TorchScript export failed: {e}")

    # 5. Export to ONNX
    print("Exporting to ONNX...")
    try:
        onnx_path = os.path.join(args.run_dir, "weights.onnx")
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
        )
        print(f"✓ ONNX model saved: {onnx_path} ({os.path.getsize(onnx_path)/(1024*1024):.2f} MB)")
    except Exception as e:
        print(f"❌ ONNX export failed: {e}")

    print("\n🎉 Export task complete.")

if __name__ == "__main__":
    main()
