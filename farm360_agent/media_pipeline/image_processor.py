import os
import io
try:
    from PIL import Image
    import torch
    import torchvision.transforms as transforms
except ImportError:
    pass

class MediaPipeline:
    """Pre-processes images for ML models."""
    def __init__(self):
        try:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        except NameError:
            self.transform = None

    def process_image(self, file_path_or_bytes):
        """Returns a standardized tensor for PyTorch vision models."""
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
