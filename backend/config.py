import os
import secrets
from pydantic_settings import BaseSettings, SettingsConfigDict
from loguru import logger
from dotenv import load_dotenv

# Load root .env into os.environ so provider_manager can read the indexed keys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
load_dotenv(os.path.join(root_dir, ".env"))

class Settings(BaseSettings):
    # ── Application key ────────────────────────────────────────────────────────
    farm360_api_key: str | None = None

    # ── Legacy single-key fields (kept for backward compatibility) ─────────────
    # These are read if only the old single-key env vars are set.
    # The provider_manager reads GOOGLE_API_KEY_1…5 / OPENROUTER_API_KEY_1…5 directly
    # from os.environ, so no pydantic fields are needed for those.
    google_api_key: str | None = None       # GOOGLE_API_KEY (legacy)
    openrouter_api_key: str | None = None   # OPENROUTER_API_KEY (legacy)
    openai_api_key: str | None = None       # OPENAI_API_KEY (legacy)

    # ── Weather ────────────────────────────────────────────────────────────────
    openweather_api_key: str | None = None

    # ── Model paths ────────────────────────────────────────────────────────────
    model_base_path: str = ""
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

# ── Post-init: back-fill legacy single keys into numbered slots ────────────────
# If a user only has GOOGLE_API_KEY (old format), treat it as GOOGLE_API_KEY_1
def _backfill_legacy(single_env: str, numbered_env: str):
    """Copy legacy single-key to slot _1 if no numbered keys are set."""
    val = os.environ.get(single_env, "").strip()
    if val and not os.environ.get(f"{numbered_env}_1", "").strip():
        os.environ[f"{numbered_env}_1"] = val
        logger.info(
            f"[Config] Back-filled {single_env} → {numbered_env}_1 "
            "(legacy key detected)"
        )

_backfill_legacy("GOOGLE_API_KEY",     "GOOGLE_API_KEY")
_backfill_legacy("OPENROUTER_API_KEY", "OPENROUTER_API_KEY")
_backfill_legacy("OPENAI_API_KEY",     "OPENAI_API_KEY")

# ── 1. Generate a secure random API key if none is configured ──────────────────
if not settings.farm360_api_key:
    settings.farm360_api_key = f"fk-{secrets.token_urlsafe(32)}"
    logger.warning(
        "⚠️  FARM360_API_KEY not set in environment. "
        f"Generated temporary key: {settings.farm360_api_key}\n"
        "    Set FARM360_API_KEY in your .env file for a permanent key."
    )

# ── 2. Resolve model base path ─────────────────────────────────────────────────
if not settings.model_base_path:
    settings.model_base_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
    logger.info(f"Model base path defaulted to: {settings.model_base_path}")

# ── 3. Validate model file existence (log warnings, don't crash) ───────────────
_model_paths_to_check = [
    ("Crop Regression", os.path.join(settings.model_base_path, settings.crop_model_path)),
    ("Dairy",           os.path.join(settings.model_base_path, settings.dairy_model_path)),
    ("Animal Disease",  os.path.join(settings.model_base_path, settings.animal_model_dir)),
    ("Crop Vision",     os.path.join(settings.model_base_path, settings.crop_vision_model_path)),
]
for name, full_path in _model_paths_to_check:
    if not os.path.exists(full_path):
        logger.warning(f"Model file not found: {name} → {full_path}")
    else:
        logger.info(f"Model found: {name} → {full_path}")