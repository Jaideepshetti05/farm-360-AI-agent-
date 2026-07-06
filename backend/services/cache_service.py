import hashlib
import json
from typing import Optional, Dict, Any
from loguru import logger
from backend.streaming.config import StreamingConfig

_MEM_CACHE: Dict[str, str] = {}

class CacheService:
    @staticmethod
    def generate_key(
        prompt_version: str,
        model: str,
        provider: str,
        language: str,
        context: Dict[str, Any]
    ) -> str:
        """Generates a secure MD5 namespace hash key based on request parameters."""
        context_str = json.dumps(context, sort_keys=True)
        context_hash = hashlib.md5(context_str.encode("utf-8")).hexdigest()
        raw_key = f"{prompt_version}:{model}:{provider}:{language}:{context_hash}"
        return hashlib.md5(raw_key.encode("utf-8")).hexdigest()

    @classmethod
    def get(cls, key: str) -> Optional[str]:
        # Level 1: Memory cache check
        if key in _MEM_CACHE:
            logger.info("[Cache] Level 1 (Memory) cache hit.")
            return _MEM_CACHE[key]
            
        # Level 2: Redis Cache fallback
        try:
            import redis
            r = redis.Redis(host="localhost", port=6379, db=0, socket_timeout=1.0)
            val = r.get(key)
            if val:
                logger.info("[Cache] Level 2 (Redis) cache hit.")
                decoded = val.decode("utf-8")
                _MEM_CACHE[key] = decoded
                return decoded
        except Exception:
            pass  # Redis offline fallback
            
        return None

    @classmethod
    def set(cls, key: str, value: str, ttl: int = None):
        if ttl is None:
            ttl = StreamingConfig.CACHE_DEFAULT_TTL
            
        _MEM_CACHE[key] = value
        
        try:
            import redis
            r = redis.Redis(host="localhost", port=6379, db=0, socket_timeout=1.0)
            r.setex(key, ttl, value)
        except Exception:
            pass

    @classmethod
    def delete(cls, key: str):
        """Invalidates a cache entry (e.g. if final validation blocks or fails)."""
        if key in _MEM_CACHE:
            del _MEM_CACHE[key]
        try:
            import redis
            r = redis.Redis(host="localhost", port=6379, db=0, socket_timeout=1.0)
            r.delete(key)
        except Exception:
            pass
