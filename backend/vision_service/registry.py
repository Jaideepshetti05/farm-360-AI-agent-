"""
Farm360 AI – Model Registry
==============================
Lazy-loading, version-aware registry for all vision models.

Design principles:
  • Models are NOT loaded at import time — only on first request (lazy).
  • Each model is identified by (task_name, version).
  • The registry reads from a JSON manifest (registry.json) so new models
    can be added without touching any Python code.
  • Hot-swap: call registry.swap(task, version) to atomically replace a model.
  • Graceful degradation: if a model file is missing the task returns None,
    and the route falls back to LLM-only response.

Model interface contract (duck-typed):
    model.predict(tensor) -> dict with at least {"predictions": [...]}

Registry manifest schema (machine_learning/model_registry/registry.json):
    {
      "crop_disease": {
        "latest": "v1.0",
        "versions": {
          "v1.0": {
            "arch": "resnet18",
            "type": "classification",
            "classes_file": "machine_learning/model_registry/classes/crop_disease_v1.json",
            "weights": "machine_learning/crop_vision/models/crop_disease_model.pth",
            "num_classes": 17,
            "input_size": 224
          }
        }
      }
    }
"""
from __future__ import annotations

import json
import os
import threading
from typing import Any, Dict, List, Optional

from loguru import logger

# Torch imports are optional (graceful degrade without GPU)
try:
    import torch
    import torch.nn as nn
    import torchvision.models as tv_models
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False
    logger.warning("[Registry] PyTorch not available – vision models disabled.")


# ── Base path resolution ───────────────────────────────────────────────────────
_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ROOT_DIR = os.path.dirname(_BACKEND_DIR)
_REGISTRY_JSON = os.path.join(_ROOT_DIR, "machine_learning", "model_registry", "registry.json")


# ── Wrapped model ──────────────────────────────────────────────────────────────

class VisionModel:
    """Thin wrapper around a PyTorch model with its class labels."""

    def __init__(
        self,
        model: Any,
        classes: List[str],
        task: str,
        version: str,
        arch: str,
        input_size: int = 224,
    ):
        self.model = model
        self.classes = classes
        self.task = task
        self.version = version
        self.arch = arch
        self.input_size = input_size

    def predict(self, tensor) -> Dict[str, Any]:
        """Run forward pass. Returns top-3 predictions."""
        if not _TORCH_OK or self.model is None:
            return {"predictions": [], "error": "Model not available"}

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0]
            top_values, top_indices = torch.topk(probs, min(3, len(self.classes)))

        predictions = []
        for rank, (idx, prob) in enumerate(
            zip(top_indices.tolist(), top_values.tolist()), start=1
        ):
            raw_label = self.classes[idx] if idx < len(self.classes) else f"class_{idx}"
            predictions.append(
                {
                    "label": raw_label,
                    "display_name": _prettify_label(raw_label),
                    "confidence": round(float(prob), 4),
                    "rank": rank,
                }
            )
        return {"predictions": predictions}


# ── Registry singleton ─────────────────────────────────────────────────────────

class ModelRegistry:
    """
    Thread-safe lazy-loading registry.
    Import the singleton `model_registry` at the bottom of this file.
    """

    def __init__(self):
        self._models: Dict[str, VisionModel] = {}   # task → loaded VisionModel
        self._manifest: Dict[str, Any] = {}          # raw JSON manifest
        self._lock = threading.RLock()
        self._load_manifest()

    # ── Manifest ───────────────────────────────────────────────────────────────

    def _load_manifest(self):
        if not os.path.exists(_REGISTRY_JSON):
            logger.warning(f"[Registry] manifest not found at {_REGISTRY_JSON}. Using built-in defaults.")
            self._manifest = _builtin_manifest()
            return
        with open(_REGISTRY_JSON, "r", encoding="utf-8") as f:
            self._manifest = json.load(f)
        logger.info(f"[Registry] Loaded manifest: {list(self._manifest.keys())}")

    # ── Public API ─────────────────────────────────────────────────────────────

    def get(self, task: str, version: str = "latest") -> Optional[VisionModel]:
        """
        Return a loaded VisionModel for (task, version).
        Loads on first access (lazy). Returns None if unavailable.
        """
        with self._lock:
            cache_key = f"{task}:{version}"
            if cache_key in self._models:
                return self._models[cache_key]

            model = self._load(task, version)
            if model is not None:
                self._models[cache_key] = model
                # Also update "latest" alias
                self._models[f"{task}:latest"] = model
            return model

    def swap(self, task: str, version: str) -> bool:
        """Hot-swap: load a new version and atomically replace the cached entry."""
        with self._lock:
            model = self._load(task, version)
            if model is None:
                return False
            self._models[f"{task}:{version}"] = model
            self._models[f"{task}:latest"] = model
            logger.success(f"[Registry] Hot-swapped {task} → {version}")
            return True

    def available_tasks(self) -> List[str]:
        return list(self._manifest.keys())

    def status(self) -> Dict[str, Any]:
        """Return registry status for the /vision/models endpoint."""
        result = {}
        for task, info in self._manifest.items():
            latest_v = info.get("latest", "v1.0")
            versions = info.get("versions", {})
            result[task] = {
                "latest": latest_v,
                "loaded": f"{task}:latest" in self._models,
                "versions": list(versions.keys()),
                "arch": versions.get(latest_v, {}).get("arch", "unknown"),
            }
        return result

    # ── Internal loading ───────────────────────────────────────────────────────

    def _load(self, task: str, version: str) -> Optional[VisionModel]:
        if task not in self._manifest:
            logger.warning(f"[Registry] Unknown task '{task}'")
            return None

        task_info = self._manifest[task]
        resolved_version = task_info.get("latest", "v1.0") if version == "latest" else version
        version_info = task_info.get("versions", {}).get(resolved_version)

        if version_info is None:
            logger.warning(f"[Registry] No version '{resolved_version}' for task '{task}'")
            return None

        weights_rel = version_info.get("weights", "")
        weights_path = os.path.join(_ROOT_DIR, weights_rel) if weights_rel else ""
        classes_rel = version_info.get("classes_file", "")
        classes_path = os.path.join(_ROOT_DIR, classes_rel) if classes_rel else ""
        arch = version_info.get("arch", "resnet18")
        num_classes = version_info.get("num_classes", 17)
        input_size = version_info.get("input_size", 224)

        # Load class labels
        classes = _load_classes(classes_path, num_classes, task)

        # Load model weights
        if not _TORCH_OK:
            logger.warning(f"[Registry] PyTorch unavailable, skipping {task}")
            return None

        if not os.path.exists(weights_path):
            logger.warning(f"[Registry] Weights not found: {weights_path}")
            return None

        try:
            pytorch_model = _build_torch_model(arch, num_classes, weights_path, input_size)
            logger.success(f"[Registry] Loaded {task} ({arch} {resolved_version}) — {num_classes} classes")
            return VisionModel(
                model=pytorch_model,
                classes=classes,
                task=task,
                version=resolved_version,
                arch=arch,
                input_size=input_size,
            )
        except Exception as exc:
            logger.error(f"[Registry] Failed to load {task}: {exc}")
            return None


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_classes(classes_path: str, num_classes: int, task: str) -> List[str]:
    """Load class labels from JSON file, or use auto-generated labels as fallback."""
    if classes_path and os.path.exists(classes_path):
        with open(classes_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data.get(str(i), f"class_{i}") for i in range(num_classes)]

    # Built-in fallback class lists for known tasks
    if task == "crop_disease":
        return CROP_DISEASE_17_CLASSES
    logger.warning(f"[Registry] No class file for {task}, using auto-labels")
    return [f"class_{i}" for i in range(num_classes)]


def _build_torch_model(arch: str, num_classes: int, weights_path: str, input_size: int):
    """Build and load a PyTorch torchvision model."""
    arch_lower = arch.lower()

    if arch_lower in ("resnet18",):
        model = tv_models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif arch_lower in ("resnet50",):
        model = tv_models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif arch_lower in ("efficientnet_b3", "efficientnet-b3"):
        model = tv_models.efficientnet_b3(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif arch_lower in ("efficientnet_b0", "efficientnet-b0"):
        model = tv_models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif arch_lower in ("mobilenet_v3_large", "mobilenetv3"):
        model = tv_models.mobilenet_v3_large(weights=None)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    else:
        raise ValueError(f"Unsupported architecture: {arch}")

    state = torch.load(weights_path, map_location=torch.device("cpu"), weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def _prettify_label(raw: str) -> str:
    """Convert 'Rice___Leaf_Blast' → 'Rice Leaf Blast'."""
    return raw.replace("___", " ").replace("__", " ").replace("_", " ").strip()


def _builtin_manifest() -> Dict[str, Any]:
    """Fallback manifest when registry.json doesn't exist yet."""
    return {
        "crop_disease": {
            "latest": "v1.0",
            "versions": {
                "v1.0": {
                    "arch": "resnet18",
                    "type": "classification",
                    "classes_file": "",
                    "weights": "machine_learning/crop_vision/models/crop_disease_model.pth",
                    "num_classes": 17,
                    "input_size": 224,
                }
            },
        },
        "breed": {
            "latest": "v1.0",
            "versions": {
                "v1.0": {
                    "arch": "efficientnet_b3",
                    "type": "classification",
                    "classes_file": "machine_learning/model_registry/classes/breed_v1.json",
                    "weights": "machine_learning/model_registry/models/breed/v1.0/weights.pth",
                    "num_classes": 41,
                    "input_size": 224,
                }
            },
        },
    }


# ── Built-in class labels ──────────────────────────────────────────────────────

CROP_DISEASE_17_CLASSES = [
    "Corn___Common_Rust",
    "Corn___Gray_Leaf_Spot",
    "Corn___Healthy",
    "Corn___Northern_Leaf_Blight",
    "Potato___Early_Blight",
    "Potato___Healthy",
    "Potato___Late_Blight",
    "Rice___Brown_Spot",
    "Rice___Healthy",
    "Rice___Leaf_Blast",
    "Rice___Neck_Blast",
    "Sugarcane___Bacterial_Blight",
    "Sugarcane___Healthy",
    "Sugarcane___Red_Rot",
    "Wheat___Brown_Rust",
    "Wheat___Healthy",
    "Wheat___Yellow_Rust",
]

INDIAN_BOVINE_41_CLASSES = [
    "Alambadi", "Amritmahal", "Ayrshire", "Banni", "Bargur",
    "Bhadawari", "Brown_Swiss", "Dangi", "Deoni", "Gir",
    "Guernsey", "Hallikar", "Hariana", "Holstein_Friesian", "Jaffrabadi",
    "Jersey", "Kangayam", "Kankrej", "Kasargod", "Kenkatha",
    "Kherigarh", "Khillari", "Krishna_Valley", "Malnad_gidda", "Mehsana",
    "Murrah", "Nagori", "Nagpuri", "Nili_Ravi", "Nimari",
    "Ongole", "Pulikulam", "Rathi", "Red_Dane", "Red_Sindhi",
    "Sahiwal", "Surti", "Tharparkar", "Toda", "Umblachery",
    "Vechur",
]


# ── Singleton ──────────────────────────────────────────────────────────────────
model_registry = ModelRegistry()
