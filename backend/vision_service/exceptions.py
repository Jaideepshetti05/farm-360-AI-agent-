"""
Farm360 AI – Vision Service Exceptions
=======================================
Custom exception classes for vision service errors.
"""

class VisionError(Exception):
    """Base exception for all vision service errors."""
    pass

class ModelNotFoundError(VisionError):
    """Raised when a requested model or version is not registered or found."""
    pass

class InvalidImageError(VisionError):
    """Raised when the uploaded file is not a valid image or violates security limits."""
    pass

class InferenceError(VisionError):
    """Raised when model inference fails during the forward pass."""
    pass

class LabelMapError(VisionError):
    """Raised when there is a mismatch or load error with the label map JSON."""
    pass
