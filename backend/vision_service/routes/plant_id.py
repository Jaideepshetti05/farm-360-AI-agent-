"""
POST /vision/plant-id
======================
Identifies plant species from a photo.
Model: ConvNeXt-Small (Phase 3 — LLM advisory until trained)
"""
from __future__ import annotations
import os
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from backend.vision_service.engine import inference_engine
from backend.vision_service.explainer import vision_explainer
from backend.vision_service.schemas import PredictionResult
from backend.vision_service.security import image_validator

router = APIRouter(prefix="/vision", tags=["Vision — Plant Identification"])
_TEMP_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "temp_uploads")


@router.post(
    "/plant-id",
    response_model=PredictionResult,
    summary="Identify plant species from image",
    description="Upload a photo of a plant/leaf. Returns species identification and agricultural significance.",
)
async def plant_id_endpoint(
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
            inference_engine.run, saved_path, "plant_id", model_version
        )

        if include_explanation and result.success:
            from backend.provider_manager import provider_manager
            if provider_manager.has_any_key:
                explanation = await run_in_threadpool(
                    vision_explainer.explain,
                    "plant_id", result.predictions, {"location": "India"}, lang, provider_manager
                )
                result.explanation = explanation

        return result

    finally:
        if os.path.exists(saved_path):
            try:
                os.remove(saved_path)
            except OSError:
                pass
