import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # API Keys - loaded from environment variables
    google_api_key: str = None  # Will be loaded from GOOGLE_API_KEY env var
    farm360_api_key: str = "secure-secret-key-1234"
    
    # Model Paths (using forward slashes for cross-platform compatibility)
    crop_model_path: str = "crop_regression/models/production_model_log.pkl"
    dairy_model_path: str = "models/dairy_intelligence_v1_20260217_210257.pkl"
    animal_model_dir: str = "models/animal_disease_20260218_215356"
    crop_vision_model_path: str = "crop_vision/models/crop_disease_model.pth"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()
