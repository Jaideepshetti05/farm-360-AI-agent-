"""
POST /vision/fruit-grade — Fruit quality grading
POST /vision/fruit-detect — Fruit detection + counting
Both routes use YOLOv11s (Phase 3 — LLM advisory until trained).
"""
from __future__ import annotations
import os
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from backend.vision_service.engine import inference_engine
from backend.vision_service.explainer import vision_explainer
from backend.vision_service.schemas import PredictionResult
from backend.vision_service.security import image_validator

router = APIRouter(prefix="/vision", tags=["Vision — Fruit"])
_TEMP_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "temp_uploads")

_GRADE_META = {
    "A": {"market": "Export / Premium retail", "price_premium": "+30%"},
    "B": {"market": "Local retail / Supermarkets", "price_premium": "Standard"},
    "C": {"market": "Processing / Juice", "price_premium": "-20%"},
}


@router.post("/fruit-grade", response_model=PredictionResult, summary="Grade harvested fruit quality")
async def fruit_grade_endpoint(
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
            inference_engine.run, saved_path, "fruit_grade", model_version
        )

        if result.success and result.predictions:
            top = result.predictions[0]
            grade_key = top.label.upper()
            result.extra.update(_GRADE_META.get(grade_key, {}))

        if include_explanation and result.success:
            from backend.provider_manager import provider_manager
            if provider_manager.has_any_key:
                explanation = await run_in_threadpool(
                    vision_explainer.explain,
                    "fruit_grade", result.predictions, {"location": "India"}, lang, provider_manager
                )
                result.explanation = explanation

        return result

    finally:
        if os.path.exists(saved_path):
            try:
                os.remove(saved_path)
            except OSError:
                pass


@router.post("/fruit-detect", response_model=PredictionResult, summary="Detect and count fruits in image")
async def fruit_detect_endpoint(
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
            inference_engine.run, saved_path, "fruit_detect", model_version
        )

        if include_explanation and result.success:
            from backend.provider_manager import provider_manager
            if provider_manager.has_any_key:
                explanation = await run_in_threadpool(
                    vision_explainer.explain,
                    "detect", result.predictions, {"location": "India"}, lang, provider_manager
                )
                result.explanation = explanation

        return result

    finally:
        if os.path.exists(saved_path):
            try:
                os.remove(saved_path)
            except OSError:
                pass
