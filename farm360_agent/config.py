import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    google_api_key: str
    farm360_api_key: str
    
    # Model Paths
    crop_model_path: str = r"..\crop_regression\models\production_model_log.pkl"
    dairy_model_path: str = r"..\models\dairy_intelligence_v1_20260217_210257.pkl"
    animal_model_dir: str = r"..\models\animal_disease_20260218_215356"
    crop_vision_model_path: str = r"..\crop_vision\models\crop_disease_model.pth"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

settings = Settings()
