import os
import secrets
from pydantic_settings import BaseSettings, SettingsConfigDict
from loguru import logger


class Settings(BaseSettings):
    # API Keys - loaded from environment variables
    openrouter_api_key: str | None = None  # Will be loaded from OPENROUTER_API_KEY env var
    farm360_api_key: str | None = None  # Will be loaded from FARM360_API_KEY env var
    openweather_api_key: str | None = None  # Will be loaded from OPENWEATHER_API_KEY env var
    
    # Model base path — explicitly set via env for Docker/local; default to project root
    model_base_path: str = ""  # Will default to project root if empty

    # Model Paths (relative to model_base_path)
    crop_model_path: str = "machine_learning/crop_regression/models/production_model_log.pkl"
    dairy_model_path: str = "machine_learning/models/dairy_intelligence_v1_20260217_210257.pkl"
    animal_model_dir: str = "machine_learning/models/animal_disease_20260218_215356"
    crop_vision_model_path: str = "machine_learning/crop_vision/models/crop_disease_model.pth"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()

# --- Post-init validation and hardening ---

# 1. Generate a secure random API key if none is configured
if not settings.farm360_api_key:
    settings.farm360_api_key = f"fk-{secrets.token_urlsafe(32)}"
    logger.warning(
        "⚠️  FARM360_API_KEY not set in environment. "
        f"Generated temporary key: {settings.farm360_api_key}\n"
        "    Set FARM360_API_KEY in your .env file for a permanent key."
    )

# 2. Log a warning if OpenRouter key is missing
if not settings.openrouter_api_key:
    logger.warning(
        "⚠️  OPENROUTER_API_KEY not set. The AI agent will run in fallback mode "
        "with deterministic responses. Set OPENROUTER_API_KEY in .env for full AI features."
    )

# 3. Resolve model base path
if not settings.model_base_path:
    # Default to the project root (two levels up from backend/)
    settings.model_base_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
    logger.info(f"Model base path defaulted to: {settings.model_base_path}")

# 4. Validate model file existence (log warnings, don't crash)
_model_paths_to_check = [
    ("Crop Regression", os.path.join(settings.model_base_path, settings.crop_model_path)),
    ("Dairy", os.path.join(settings.model_base_path, settings.dairy_model_path)),
    ("Animal Disease", os.path.join(settings.model_base_path, settings.animal_model_dir)),
    ("Crop Vision", os.path.join(settings.model_base_path, settings.crop_vision_model_path)),
]
for name, full_path in _model_paths_to_check:
    if not os.path.exists(full_path):
        logger.warning(f"Model file not found: {name} -> {full_path}")
    else:
        logger.info(f"Model found: {name} -> {full_path}")