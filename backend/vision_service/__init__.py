"""Vision service package — exports router for registration in app.py."""
from backend.vision_service.routes.crop_disease import router as crop_disease_router
from backend.vision_service.routes.breed import router as breed_router
from backend.vision_service.routes.weed import router as weed_router
from backend.vision_service.routes.detect import router as detect_router
from backend.vision_service.routes.plant_id import router as plant_id_router
from backend.vision_service.routes.fruit import router as fruit_router
from backend.vision_service.registry import model_registry

__all__ = [
    "crop_disease_router",
    "breed_router",
    "weed_router",
    "detect_router",
    "plant_id_router",
    "fruit_router",
    "model_registry",
]
