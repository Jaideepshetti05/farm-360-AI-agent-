"""
POST /vision/detect
====================
General object detection endpoint.
Uses annotation-based animal detector or falls back to LLM description.

Model: YOLOv11s (Phase 2 — stub until trained from XML annotations)
"""
from __future__ import annotations

import os

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from backend.vision_service.engine import inference_engine
from backend.vision_service.explainer import vision_explainer
from backend.vision_service.schemas import PredictionResult
from backend.vision_service.security import image_validator

router = APIRouter(prefix="/vision", tags=["Vision — Object Detection"])

_TEMP_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "temp_uploads"
)


@router.post(
    "/detect",
    response_model=PredictionResult,
    summary="General object detection in farm scene",
    description="Detect animals, crops, equipment in a farm image. Returns bounding boxes.",
)
async def detect_endpoint(
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
            "detect",
            model_version,
        )

        if include_explanation and result.success:
            from backend.provider_manager import provider_manager
            if provider_manager.has_any_key:
                profile = {"location": "India"}
                explanation = await run_in_threadpool(
                    vision_explainer.explain,
                    "detect",
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
