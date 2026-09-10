import base64
import os
from loguru import logger
from backend.config import settings

# Attempt to load cryptography Fernet
try:
    from cryptography.fernet import Fernet
    _CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    _CRYPTOGRAPHY_AVAILABLE = False
    logger.warning(
        "[Security] cryptography package not installed. "
        "Falling back to key-based XOR obfuscation for local development. "
        "Install cryptography ('pip install cryptography') for production-grade security."
    )

def is_production() -> bool:
    """Check if application is running in production mode."""
    env = (
        os.environ.get("ENVIRONMENT")
        or os.environ.get("APP_ENV")
        or os.environ.get("ENV")
        or getattr(settings, "environment", "")
        or ""
    ).strip().lower()
    return env in ("production", "prod", "live")

def is_staging() -> bool:
    """Check if application is running in staging/pre-production mode."""
    env = (
        os.environ.get("ENVIRONMENT")
        or os.environ.get("APP_ENV")
        or os.environ.get("ENV")
        or getattr(settings, "environment", "")
        or ""
    ).strip().lower()
    return env in ("staging", "stage", "preprod", "pre-production")

# Fallback key-based XOR encryption for development without dependencies
def _fallback_xor(data: str, key: str) -> str:
    """Simple XOR cipher using key hash as fallback."""
    import hashlib
    key_bytes = hashlib.sha256(key.encode()).digest()
    data_bytes = data.encode()
    cipher_bytes = bytearray(
        data_bytes[i] ^ key_bytes[i % len(key_bytes)] for i in range(len(data_bytes))
    )
    return base64.b64encode(cipher_bytes).decode()

def _fallback_xor_decrypt(cipher_text: str, key: str) -> str:
    """Decrypt key-based XOR cipher."""
    import hashlib
    key_bytes = hashlib.sha256(key.encode()).digest()
    try:
        cipher_bytes = base64.b64decode(cipher_text.encode())
        data_bytes = bytearray(
            cipher_bytes[i] ^ key_bytes[i % len(key_bytes)] for i in range(len(cipher_bytes))
        )
        return data_bytes.decode()
    except Exception:
        return cipher_text

class Encryptor:
    def __init__(self, key: str | None = None):
        # Resolve key from parameter, settings, or environment
        resolved_key = (
            key
            or settings.farm360_encryption_key
            or os.environ.get("FARM360_ENCRYPTION_KEY", "").strip()
            or None
        )

        if not resolved_key:
            if is_production():
                raise ValueError(
                    "FARM360_ENCRYPTION_KEY must be explicitly set in production mode. "
                    "Cannot use fallback encryption key."
                )
            resolved_key = "farm360-default-insecure-encryption-key-12345"
            logger.warning("[Security] FARM360_ENCRYPTION_KEY not set. Using fallback development key.")

        self.key = resolved_key
        self._fernet = None

        if _CRYPTOGRAPHY_AVAILABLE:
            try:
                # Fernet key must be 32 URL-safe base64-encoded bytes
                import hashlib
                hashed_key = hashlib.sha256(resolved_key.encode()).digest()
                b64_key = base64.urlsafe_b64encode(hashed_key)
                self._fernet = Fernet(b64_key)
            except Exception as e:
                if is_production():
                    raise RuntimeError(f"[Security] Failed to initialize Fernet in production: {e}")
                logger.error(f"[Security] Failed to initialize Fernet: {e}. Falling back to XOR.")
                self._fernet = None
        elif is_production():
            raise RuntimeError(
                "[Security] The 'cryptography' package is required for production PII encryption. "
                "Insecure XOR fallback is prohibited in production."
            )

    def encrypt(self, plain_text: str) -> str:
        if not plain_text:
            return ""
        if self._fernet:
            try:
                return self._fernet.encrypt(plain_text.encode()).decode()
            except Exception as e:
                logger.error(f"[Security] Fernet encryption failed: {e}")
        return _fallback_xor(plain_text, self.key)

    def decrypt(self, cipher_text: str) -> str:
        if not cipher_text:
            return ""
        if self._fernet:
            try:
                return self._fernet.decrypt(cipher_text.encode()).decode()
            except Exception as e:
                # Maybe it was encrypted with fallback XOR
                logger.debug(f"[Security] Fernet decryption failed, trying fallback: {e}")
        return _fallback_xor_decrypt(cipher_text, self.key)

encryptor = Encryptor()
