import os
import io
from loguru import logger

try:
    from PIL import Image
    import torch
    import torchvision.transforms as transforms
    _HAS_IMAGE_DEPS = True
except ImportError as e:
    _HAS_IMAGE_DEPS = False
    _IMAGE_IMPORT_ERROR = str(e)


class MediaPipeline:
    """Pre-processes images for ML models."""
    def __init__(self):
        if not _HAS_IMAGE_DEPS:
            logger.error(
                f"Image dependencies not available: {_IMAGE_IMPORT_ERROR}. "
                "Image processing will fail at runtime. Install with: "
                "pip install pillow torch torchvision"
            )
            self.transform = None
            return

        try:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            logger.info("MediaPipeline initialized with image transforms (224x224, ImageNet norm).")
        except Exception as e:
            logger.error(f"Failed to initialize image transforms: {e}")
            self.transform = None

    def process_image(self, file_path_or_bytes):
        """Returns a standardized tensor for PyTorch vision models."""
        if self.transform is None:
            raise RuntimeError(
                "MediaPipeline not properly initialized. "
                "Missing dependencies: pillow, torch, torchvision."
            )

        try:
            if isinstance(file_path_or_bytes, str):
                if not os.path.exists(file_path_or_bytes):
                    raise FileNotFoundError(f"Image not found at {file_path_or_bytes}")
                image = Image.open(file_path_or_bytes).convert("RGB")
            else:
                image = Image.open(io.BytesIO(file_path_or_bytes)).convert("RGB")

            tensor = self.transform(image).unsqueeze(0)  # Add batch dimension
            return tensor
        except Exception as e:
            raise ValueError(f"Failed to process image: {str(e)}")