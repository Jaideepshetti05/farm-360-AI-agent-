"""
POST /vision/breed
===================
Identifies Indian bovine breeds from a photo.
Supports 41 breeds including Gir, Sahiwal, Murrah, Jersey, HF and others.

Model: EfficientNetV2-S (when trained) / ResNet18 (stub until trained)
Dataset: Indian_bovine_breeds — 41 classes
"""
from __future__ import annotations

import os

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from loguru import logger

from backend.vision_service.engine import inference_engine
from backend.vision_service.explainer import vision_explainer
from backend.vision_service.schemas import PredictionResult
from backend.vision_service.security import image_validator

router = APIRouter(prefix="/vision", tags=["Vision — Animal Breed"])

_TEMP_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "temp_uploads"
)

# Breed metadata: milk yield, draft use, region
_BREED_META = {
    "Gir":              {"type": "Dual", "avg_milk_L_day": 8,  "region": "Gujarat"},
    "Sahiwal":          {"type": "Dairy", "avg_milk_L_day": 10, "region": "Punjab/Rajasthan"},
    "Holstein_Friesian":{"type": "Dairy", "avg_milk_L_day": 25, "region": "Exotic/Crossbreed"},
    "Jersey":           {"type": "Dairy", "avg_milk_L_day": 15, "region": "Exotic/Crossbreed"},
    "Murrah":           {"type": "Buffalo-Dairy", "avg_milk_L_day": 12, "region": "Haryana/Punjab"},
    "Kankrej":          {"type": "Draft", "avg_milk_L_day": 5,  "region": "Gujarat/Rajasthan"},
    "Ongole":           {"type": "Draft", "avg_milk_L_day": 3,  "region": "Andhra Pradesh"},
    "Red_Sindhi":       {"type": "Dairy", "avg_milk_L_day": 7,  "region": "Sindh (Pakistan)/India"},
    "Tharparkar":       {"type": "Dual", "avg_milk_L_day": 6,  "region": "Rajasthan"},
    "Rathi":            {"type": "Dual", "avg_milk_L_day": 7,  "region": "Rajasthan"},
    "Hariana":          {"type": "Draft", "avg_milk_L_day": 4,  "region": "Haryana/UP"},
    "Jaffrabadi":       {"type": "Buffalo-Dairy", "avg_milk_L_day": 10, "region": "Gujarat"},
    "Nili_Ravi":        {"type": "Buffalo-Dairy", "avg_milk_L_day": 14, "region": "Punjab"},
    "Mehsana":          {"type": "Buffalo-Dairy", "avg_milk_L_day": 9,  "region": "Gujarat"},
    "Surti":            {"type": "Buffalo-Dairy", "avg_milk_L_day": 7,  "region": "Gujarat"},
    "Ayrshire":         {"type": "Dairy", "avg_milk_L_day": 18, "region": "Exotic"},
    "Brown_Swiss":      {"type": "Dual", "avg_milk_L_day": 20, "region": "Exotic"},
    "Guernsey":         {"type": "Dairy", "avg_milk_L_day": 16, "region": "Exotic"},
    "Red_Dane":         {"type": "Dual", "avg_milk_L_day": 18, "region": "Exotic"},
}


@router.post(
    "/breed",
    response_model=PredictionResult,
    summary="Identify Indian bovine breed from photo",
    description=(
        "Upload a photo of a cow or buffalo. Returns the top predicted breed "
        "with milk yield data, region, and management advice."
    ),
)
async def breed_endpoint(
    image: UploadFile = File(...),
    lang: str = Form(default="en"),
    session_id: str = Form(default="default"),
    include_explanation: bool = Form(default=True),
    model_version: str = Form(default="latest"),
):
    if not image.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")

    saved_path = await image_validator.validate_and_save(image, _TEMP_DIR)

    try:
        result: PredictionResult = await run_in_threadpool(
            inference_engine.run,
            saved_path,
            "breed",
            model_version,
        )

        # Enrich with breed metadata
        if result.success and result.predictions:
            top_breed = result.predictions[0].label.replace(" ", "_")
            meta = _BREED_META.get(top_breed, {})
            result.extra.update(
                {
                    "breed_name": result.predictions[0].display_name,
                    "animal_type": meta.get("type", "Unknown"),
                    "avg_milk_L_day": meta.get("avg_milk_L_day"),
                    "origin_region": meta.get("region", "India"),
                }
            )

        if include_explanation and result.success and result.predictions:
            from backend.provider_manager import provider_manager
            if provider_manager.has_any_key:
                profile = {"location": "India", "livestock": "cattle"}
                explanation = await run_in_threadpool(
                    vision_explainer.explain,
                    "breed",
                    result.predictions,
                    profile,
                    lang,
                    provider_manager,
                )
                result.explanation = explanation

        return result

    finally:
        if os.path.exists(saved_path):
            try:
                os.remove(saved_path)
            except OSError:
                pass
