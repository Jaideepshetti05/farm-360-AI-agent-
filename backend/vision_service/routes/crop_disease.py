"""
POST /vision/crop-disease
==========================
Accepts a crop/leaf image and returns:
  • Top-3 disease/healthy predictions with confidence scores
  • Crop type, disease name, severity level
  • LLM-generated farmer-friendly explanation (optional)
  • Treatment recommendations + product names

Supported classes (v1.0 — ResNet18, 17 classes):
  Corn, Potato, Rice, Sugarcane, Wheat × diseases + Healthy

Upgrade path: swap to EfficientNet-B3 (42 classes) via registry hot-swap.
"""
from __future__ import annotations

import os
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from loguru import logger

from backend.config import settings
from backend.vision_service.engine import inference_engine
from backend.vision_service.explainer import vision_explainer
from backend.vision_service.schemas import PredictionResult
from backend.vision_service.security import image_validator

# Lazy import to avoid circular dependency at module load time
def _get_provider_manager():
    from backend.provider_manager import provider_manager
    return provider_manager

def _get_memory():
    from backend.memory.session import MemoryManager
    return MemoryManager()

# ── Router ─────────────────────────────────────────────────────────────────────
router = APIRouter(prefix="/vision", tags=["Vision — Crop Disease"])

# Upload dir (reuse existing temp_uploads)
_TEMP_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "temp_uploads")

# Crop-specific metadata
_CROP_META = {
    "Corn":      {"causative_agent": "Fungal / Bacterial", "action_window_days": 7},
    "Potato":    {"causative_agent": "Oomycete (Phytophthora / Alternaria)", "action_window_days": 3},
    "Rice":      {"causative_agent": "Fungal (Magnaporthe / Bipolaris)", "action_window_days": 5},
    "Sugarcane": {"causative_agent": "Bacterial / Fungal", "action_window_days": 14},
    "Wheat":     {"causative_agent": "Fungal (Puccinia / Fusarium)", "action_window_days": 7},
}

_TREATMENT_QUICK_REF = {
    "Leaf_Blast":         "Tricyclazole 75WP @ 6 g/10 L water",
    "Neck_Blast":         "Tricyclazole 75WP @ 6 g/10 L water",
    "Brown_Spot":         "Mancozeb 75WP @ 20 g/10 L water",
    "Early_Blight":       "Chlorothalonil 75WP @ 2 g/L + Mancozeb",
    "Late_Blight":        "Metalaxyl + Mancozeb @ 2.5 g/L — URGENT",
    "Common_Rust":        "Propiconazole 25 EC @ 1 mL/L",
    "Gray_Leaf_Spot":     "Azoxystrobin 23 SC @ 1 mL/L",
    "Northern_Leaf_Blight": "Mancozeb 75WP @ 2 g/L",
    "Yellow_Rust":        "Propiconazole 25 EC @ 1 mL/L + Tebuconazole",
    "Brown_Rust":         "Propiconazole 25 EC @ 0.5 mL/L",
    "Bacterial_Blight":   "Copper Oxychloride 50WP @ 3 g/L",
    "Red_Rot":            "Remove infected stalks; Carbendazim 50WP @ 1 g/L soil drench",
}


@router.post(
    "/crop-disease",
    response_model=PredictionResult,
    summary="Detect crop disease from leaf/plant image",
    description=(
        "Upload a JPEG/PNG/WEBP image of a crop leaf or plant. "
        "Returns disease classification with confidence scores and "
        "LLM-generated treatment advice."
    ),
)
async def crop_disease_endpoint(
    image: UploadFile = File(..., description="Crop/leaf image (JPEG, PNG, WEBP — max 10 MB)"),
    lang: str = Form(default="en", description="Response language (en/hi/te/kn/mr/pa/bn/ta)"),
    session_id: str = Form(default="default"),
    include_explanation: bool = Form(default=True),
    model_version: str = Form(default="latest"),
):
    if not image.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")

    # 1. Validate + save image securely
    saved_path = await image_validator.validate_and_save(image, _TEMP_DIR)

    try:
        # 2. Run inference
        result: PredictionResult = await run_in_threadpool(
            inference_engine.run,
            saved_path,
            "crop_disease",
            model_version,
        )

        # 3. Enrich with crop-specific metadata
        if result.success and result.predictions:
            top = result.predictions[0]
            crop_name = _extract_crop(top.label)
            disease_name = _extract_disease(top.label)
            treatment = _get_treatment(disease_name)
            result.extra.update(
                {
                    "crop_type": crop_name,
                    "disease_name": disease_name,
                    "is_healthy": "Healthy" in top.label,
                    "quick_treatment": treatment,
                    **(_CROP_META.get(crop_name, {})),
                }
            )

        # 4. LLM explanation
        if include_explanation and result.success and result.predictions:
            pm = _get_provider_manager()
            if pm.has_any_key:
                profile = {"location": "India", "primary_crop": result.extra.get("crop_type", "mixed")}
                explanation = await run_in_threadpool(
                    vision_explainer.explain,
                    "crop_disease",
                    result.predictions,
                    profile,
                    lang,
                    pm,
                )
                result.explanation = explanation

        return result

    finally:
        # 5. Clean up temp file
        if os.path.exists(saved_path):
            try:
                os.remove(saved_path)
            except OSError:
                pass


# ── Helpers ────────────────────────────────────────────────────────────────────

def _extract_crop(label: str) -> str:
    """'Rice___Leaf_Blast' → 'Rice'"""
    crops = ["Corn", "Potato", "Rice", "Sugarcane", "Wheat"]
    for c in crops:
        if label.startswith(c):
            return c
    return label.split("___")[0].split("_")[0]


def _extract_disease(label: str) -> str:
    """'Rice___Leaf_Blast' → 'Leaf Blast'"""
    parts = label.split("___")
    if len(parts) > 1:
        return parts[1].replace("_", " ")
    return label.replace("_", " ")


def _get_treatment(disease_name: str) -> Optional[str]:
    for keyword, treatment in _TREATMENT_QUICK_REF.items():
        if keyword.replace("_", " ").lower() in disease_name.lower():
            return treatment
    return None
