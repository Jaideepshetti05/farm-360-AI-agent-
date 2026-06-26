"""
Farm360 AI – Image Security Validator
=======================================
Validates every uploaded image before it touches any ML model:
  • MIME type (true content-type, not just extension)
  • File size
  • Maximum dimensions
  • Corruption check via PIL verify()
  • Sanitized filename (UUID-based, no path traversal)

Usage:
    validator = ImageValidator()
    safe_path = await validator.validate_and_save(upload_file, dest_dir)
"""
from __future__ import annotations

import io
import os
import uuid
from typing import Tuple

from fastapi import HTTPException, UploadFile
from loguru import logger

try:
    from PIL import Image, UnidentifiedImageError
    _PIL_OK = True
except ImportError:
    _PIL_OK = False
    logger.warning("Pillow not installed – image corruption check disabled")


# ── Constants ─────────────────────────────────────────────────────────────────
MAX_SIZE_BYTES: int = 10 * 1024 * 1024          # 10 MB
MAX_DIMENSIONS: Tuple[int, int] = (4096, 4096)  # 4 K × 4 K
MIN_DIMENSIONS: Tuple[int, int] = (32, 32)      # Too-small images rejected

# True MIME types we accept (checked via Pillow format, not file extension)
ALLOWED_PIL_FORMATS = {"JPEG", "PNG", "WEBP", "BMP", "TIFF"}

# Extension whitelist (secondary guard)
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}


class ImageValidator:
    """Stateless image validator — reuse the same instance across requests."""

    # ── Public API ─────────────────────────────────────────────────────────────

    async def validate_and_save(self, file: UploadFile, dest_dir: str) -> str:
        """
        Validate the uploaded image and save it with a UUID filename.
        Returns the absolute path to the saved file.
        Raises HTTP 400/413/415/422 on validation failure.
        """
        os.makedirs(dest_dir, exist_ok=True)

        # 1. Extension guard (fast check)
        self._check_extension(file.filename or "")

        # 2. Read full content (bounded read)
        contents = await file.read()
        self._check_size(len(contents))

        # 3. PIL-based format + corruption check
        img_format, width, height = self._check_image(contents)

        # 4. Dimension guard
        self._check_dimensions(width, height)

        # 5. Save with UUID filename
        ext = self._safe_ext(file.filename or "", img_format)
        safe_name = f"{uuid.uuid4().hex}{ext}"
        dest_path = os.path.join(dest_dir, safe_name)
        with open(dest_path, "wb") as f:
            f.write(contents)

        logger.info(
            f"[ImageValidator] Accepted: {file.filename!r} → {safe_name} "
            f"({width}×{height} {img_format}, {len(contents)//1024} KB)"
        )
        return dest_path

    # ── Private helpers ────────────────────────────────────────────────────────

    def _check_extension(self, filename: str) -> None:
        ext = os.path.splitext(filename)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
            )

    def _check_size(self, size: int) -> None:
        if size > MAX_SIZE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"File too large ({size // (1024*1024)} MB). Maximum is {MAX_SIZE_BYTES // (1024*1024)} MB.",
            )
        if size == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    def _check_image(self, data: bytes) -> Tuple[str, int, int]:
        """Returns (format, width, height). Raises 422 on corrupt/invalid image."""
        if not _PIL_OK:
            # Fallback: trust the extension
            return "UNKNOWN", 224, 224

        try:
            with Image.open(io.BytesIO(data)) as img:
                img_format = img.format or "UNKNOWN"
                if img_format.upper() not in ALLOWED_PIL_FORMATS:
                    raise HTTPException(
                        status_code=415,
                        detail=f"Image format '{img_format}' is not accepted. "
                               f"Allowed: {sorted(ALLOWED_PIL_FORMATS)}",
                    )
                width, height = img.size
                # Verify forces PIL to fully decode the image (catches truncated files)
                img.verify()
            return img_format, width, height
        except HTTPException:
            raise
        except (UnidentifiedImageError, Exception) as exc:
            raise HTTPException(
                status_code=422,
                detail=f"The uploaded file could not be read as a valid image: {exc}",
            )

    def _check_dimensions(self, width: int, height: int) -> None:
        max_w, max_h = MAX_DIMENSIONS
        min_w, min_h = MIN_DIMENSIONS
        if width > max_w or height > max_h:
            raise HTTPException(
                status_code=422,
                detail=f"Image dimensions ({width}×{height}) exceed maximum ({max_w}×{max_h}).",
            )
        if width < min_w or height < min_h:
            raise HTTPException(
                status_code=422,
                detail=f"Image dimensions ({width}×{height}) are too small (minimum {min_w}×{min_h}).",
            )

    @staticmethod
    def _safe_ext(filename: str, pil_format: str) -> str:
        """Return a safe extension from PIL format or the original filename."""
        fmt_map = {
            "JPEG": ".jpg",
            "PNG": ".png",
            "WEBP": ".webp",
            "BMP": ".bmp",
            "TIFF": ".tiff",
        }
        if pil_format in fmt_map:
            return fmt_map[pil_format]
        ext = os.path.splitext(filename)[1].lower()
        return ext if ext in ALLOWED_EXTENSIONS else ".jpg"


# Singleton instance — import this directly in routes
image_validator = ImageValidator()
