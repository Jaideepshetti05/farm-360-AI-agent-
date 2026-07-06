"""
Farm360 AI – Inference Engine
================================
Centralised preprocessing + inference layer.

Responsibilities:
  1. Load and preprocess the image (resize, normalise)
  2. Delegate to the correct VisionModel from the registry
  3. Return a standardised PredictionResult
  4. Log timing for monitoring

Usage:
    engine = InferenceEngine()
    result = await engine.run(image_path, task="crop_disease", version="latest")
"""
from __future__ import annotations

import os
import time
from typing import Optional

from loguru import logger

from backend.vision_service.registry import model_registry, VisionModel
from backend.vision_service.schemas import (
    ClassPrediction,
    ImageMeta,
    PredictionResult,
)

try:
    from PIL import Image
    import torch
    import torchvision.transforms as T
    _DEPS_OK = True
except ImportError:
    _DEPS_OK = False


# ── Standard ImageNet normalisation ───────────────────────────────────────────
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


class InferenceEngine:
    """
    Stateless inference engine.
    The singleton `inference_engine` is used by all route handlers.
    """

    # ── Public API ─────────────────────────────────────────────────────────────

    def run(
        self,
        image_path: str,
        task: str,
        version: str = "latest",
        confidence_threshold: float = 0.10,
    ) -> PredictionResult:
        """
        Full inference pipeline:
            load image → preprocess → forward → postprocess → PredictionResult
        """
        t_start = time.perf_counter()

        # 1. Resolve model
        vision_model: Optional[VisionModel] = model_registry.get(task, version)

        # 2. Load image
        if not _DEPS_OK:
            return self._error_result(task, "PyTorch / Pillow not installed")

        if not os.path.exists(image_path):
            return self._error_result(task, f"Image not found: {image_path}")

        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as exc:
            return self._error_result(task, f"Cannot open image: {exc}")

        width, height = img.size

        # 3. If model unavailable, return degraded result (LLM-only path)
        if vision_model is None:
            elapsed = (time.perf_counter() - t_start) * 1000
            logger.warning(f"[Engine] No model for task={task} — returning empty predictions")
            return PredictionResult(
                task=task,
                success=True,
                predictions=[],
                metadata=ImageMeta(
                    width=width,
                    height=height,
                    processing_time_ms=round(elapsed, 1),
                    model_version="unavailable",
                    confidence_threshold=confidence_threshold,
                ),
                extra={"model_available": False},
            )

        # 4. Preprocess
        tensor = self._preprocess(img, vision_model.input_size)

        # 5. Forward pass
        raw = vision_model.predict(tensor)
        if "error" in raw:
            return self._error_result(task, raw["error"], (time.perf_counter() - t_start) * 1000)

        # 6. Build ClassPrediction list (filter by threshold)
        predictions = [
            ClassPrediction(**p)
            for p in raw.get("predictions", [])
            if p["confidence"] >= confidence_threshold
        ]

        elapsed = (time.perf_counter() - t_start) * 1000
        logger.info(
            f"[Engine] task={task} top={predictions[0].display_name if predictions else 'none'} "
            f"conf={predictions[0].confidence if predictions else 0:.2f} "
            f"time={elapsed:.0f}ms"
        )

        # Record success metrics
        from backend.vision_service.monitoring import metrics_manager
        metrics_manager.record_request(task, elapsed, success=True)

        # Build extra dictionary containing calibration stats
        extra_data = {
            "arch": vision_model.arch,
            "model_available": True,
        }
        if "entropy" in raw:
            extra_data["entropy"] = round(raw["entropy"], 4)
        if "raw_confidence" in raw:
            extra_data["raw_confidence"] = round(raw["raw_confidence"], 4)
        if "calibrated_confidence" in raw:
            extra_data["calibrated_confidence"] = round(raw["calibrated_confidence"], 4)

        return PredictionResult(
            task=task,
            success=True,
            predictions=predictions,
            metadata=ImageMeta(
                width=width,
                height=height,
                processing_time_ms=round(elapsed, 1),
                model_version=vision_model.version,
                confidence_threshold=confidence_threshold,
            ),
            extra=extra_data,
        )

    # ── Private helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _preprocess(img: "Image.Image", input_size: int):
        """Standard ImageNet preprocessing."""
        transform = T.Compose(
            [
                T.Resize((input_size, input_size)),
                T.ToTensor(),
                T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
            ]
        )
        return transform(img).unsqueeze(0)  # Add batch dim

    def _error_result(self, task: str, message: str, elapsed_ms: float = 0.0) -> PredictionResult:
        logger.error(f"[Engine] {task}: {message}")
        try:
            from backend.vision_service.monitoring import metrics_manager
            metrics_manager.record_request(task, elapsed_ms, success=False)
        except Exception:
            pass
        return PredictionResult(task=task, success=False, error=message)


# ── Singleton ──────────────────────────────────────────────────────────────────
inference_engine = InferenceEngine()
