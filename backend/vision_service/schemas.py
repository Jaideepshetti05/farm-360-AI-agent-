"""
Farm360 AI – Vision Service Schemas
====================================
Standardised Pydantic models for every vision endpoint.
All endpoints share a common PredictionResult; task-specific
fields live inside the `extra` dict so the schema stays open
for future model additions without breaking existing consumers.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ── Request ────────────────────────────────────────────────────────────────────

class VisionRequest(BaseModel):
    """Common parameters attached to every vision upload (via Form fields)."""
    task: str = Field(default="crop-disease", description="Vision task identifier")
    lang: str = Field(default="en", description="Language code for LLM explanation (en/hi/te/kn/mr/pa/bn/ta)")
    session_id: str = Field(default="default", description="Session for prediction history")
    include_explanation: bool = Field(default=True, description="Whether to call LLM for farmer-friendly advice")
    model_version: str = Field(default="latest", description="Model version to use (latest or semver tag)")


# ── Sub-models ─────────────────────────────────────────────────────────────────

class BoundingBox(BaseModel):
    """XYXY bounding box in pixel coordinates."""
    x1: float
    y1: float
    x2: float
    y2: float
    label: str
    confidence: float
    display_name: str = ""


class ClassPrediction(BaseModel):
    """Single-class probability entry."""
    label: str                              # Raw model label
    display_name: str                       # Human-readable name
    confidence: float                       # 0.0 – 1.0
    rank: int = 1                           # 1 = top prediction


class Explanation(BaseModel):
    """LLM-generated farmer-friendly explanation."""
    text: str
    language: str = "en"
    recommendations: List[str] = Field(default_factory=list)
    urgency: str = "Medium"                 # Low / Medium / High / Critical
    treatment_products: List[str] = Field(default_factory=list)


class ImageMeta(BaseModel):
    """Image processing metadata."""
    width: int
    height: int
    channels: int = 3
    processing_time_ms: float
    model_version: str
    confidence_threshold: float = 0.5


# ── Main Response ──────────────────────────────────────────────────────────────

class PredictionResult(BaseModel):
    """Unified response returned by every /vision/* endpoint."""
    task: str
    success: bool = True
    error: Optional[str] = None

    # Classification results (top-3)
    predictions: List[ClassPrediction] = Field(default_factory=list)

    # Detection results (bounding boxes)
    bounding_boxes: Optional[List[BoundingBox]] = None

    # Segmentation (base64-encoded PNG mask or URL)
    segmentation_mask: Optional[str] = None

    # LLM explanation
    explanation: Optional[Explanation] = None

    # Image metadata
    metadata: Optional[ImageMeta] = None

    # Task-specific extra fields (open-schema extension)
    extra: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        json_schema_extra = {
            "example": {
                "task": "crop-disease",
                "success": True,
                "predictions": [
                    {
                        "label": "Rice___Leaf_Blast",
                        "display_name": "Rice Leaf Blast",
                        "confidence": 0.923,
                        "rank": 1,
                    }
                ],
                "bounding_boxes": None,
                "explanation": {
                    "text": "Your rice crop shows signs of Leaf Blast...",
                    "language": "en",
                    "recommendations": ["Apply Tricyclazole 75WP @ 6g/10L water"],
                    "urgency": "High",
                    "treatment_products": ["Tricyclazole 75WP", "Mancozeb 75WP"],
                },
                "metadata": {
                    "width": 1280,
                    "height": 960,
                    "processing_time_ms": 243,
                    "model_version": "resnet18_v1.0",
                    "confidence_threshold": 0.5,
                },
            }
        }


class PredictionHistory(BaseModel):
    """A stored past prediction entry."""
    prediction_id: str
    session_id: str
    task: str
    timestamp: str
    top_label: str
    confidence: float
    image_name: str
