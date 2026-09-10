import hashlib
import json
import time
from typing import Optional, Dict, Any, Tuple
from loguru import logger

# Memory Cache stores: key -> (value, expires_at)
_MEM_CACHE: Dict[str, Tuple[str, Optional[float]]] = {}

class RedisManager:
    """Manages Redis connection with exponential backoff on failure."""
    def __init__(self):
        self.client = None
        self.offline_until = 0.0
        self.backoff_delay = 1.0

    def get_client(self, request_id: Optional[str] = None) -> Optional[Any]:
        import os
        now = time.time()
        if self.offline_until > now:
            return None

        if self.client is None:
            try:
                import redis
                from backend.config import settings

                host = os.environ.get("REDIS_HOST") or getattr(settings, "redis_host", "127.0.0.1")
                port = int(os.environ.get("REDIS_PORT") or getattr(settings, "redis_port", 6379))
                db = int(os.environ.get("REDIS_DB") or getattr(settings, "redis_db", 0))

                raw_password = os.environ.get("REDIS_PASSWORD") or getattr(settings, "redis_password", "")
                password = str(raw_password).strip() if raw_password else None

                ssl_env = os.environ.get("REDIS_SSL")
                ssl_val = ssl_env if ssl_env is not None else getattr(settings, "redis_ssl", False)
                ssl_enabled = ssl_val if isinstance(ssl_val, bool) else str(ssl_val).strip().lower() in ("true", "1", "yes", "on")

                req_prefix = f"[{request_id}] " if request_id else ""
                auth_desc = " (authenticated)" if password else ""
                ssl_desc = " (SSL/TLS enabled)" if ssl_enabled else ""
                logger.debug(f"{req_prefix}[Cache] Attempting to connect to Redis at {host}:{port}/{db}{auth_desc}{ssl_desc}")

                self.client = redis.Redis(
                    host=host,
                    port=port,
                    db=db,
                    password=password,
                    ssl=ssl_enabled,
                    socket_timeout=1.0,
                    socket_connect_timeout=1.0
                )
                self.client.ping()
                # Success: reset backoff
                self.backoff_delay = 1.0
                logger.info(f"{req_prefix}[Cache] Connected to Redis successfully.")
            except Exception as e:
                self.client = None
                self.offline_until = now + self.backoff_delay
                logger.warning(
                    f"{req_prefix}[Cache] Redis connection failed: {e}. "
                    f"Quarantining Redis for {self.backoff_delay:.1f}s."
                )
                # Exponential backoff up to 60s
                self.backoff_delay = min(self.backoff_delay * 2, 60.0)
                return None
        return self.client

    def handle_error(self, e: Exception, request_id: Optional[str] = None):
        req_prefix = f"[{request_id}] " if request_id else ""
        logger.warning(f"{req_prefix}[Cache] Redis connection error: {e}. Disconnecting Redis client.")
        self.client = None
        self.offline_until = time.time() + self.backoff_delay
        self.backoff_delay = min(self.backoff_delay * 2, 60.0)

_redis_manager = RedisManager()

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
    def get(cls, key: str, request_id: Optional[str] = None) -> Optional[str]:
        from backend.observability.event_bus import EventBus
        req_prefix = f"[{request_id}] " if request_id else ""
        now = time.time()
        
        # Level 1: Memory cache check (evict expired entries on read)
        if key in _MEM_CACHE:
            value, expires_at = _MEM_CACHE[key]
            if expires_at is None or now < expires_at:
                logger.info(f"{req_prefix}[Cache] Level 1 (Memory) cache hit.")
                EventBus.publish("cache_hit", {"key": key, "level": 1, "request_id": request_id})
                return value
            else:
                logger.info(f"{req_prefix}[Cache] Level 1 (Memory) cache entry expired. Evicting.")
                del _MEM_CACHE[key]

        # Level 2: Redis Cache check
        r = _redis_manager.get_client(request_id)
        if r:
            EventBus.publish("redis_op_started", {"op": "get", "key": key, "request_id": request_id})
            try:
                val = r.get(key)
                if val:
                    logger.info(f"{req_prefix}[Cache] Level 2 (Redis) cache hit.")
                    decoded = val.decode("utf-8")
                    
                    # Back-populate Memory cache
                    ttl = r.ttl(key)
                    expires_at = (now + ttl) if ttl > 0 else None
                    _MEM_CACHE[key] = (decoded, expires_at)
                    
                    EventBus.publish("cache_hit", {"key": key, "level": 2, "request_id": request_id})
                    EventBus.publish("redis_op_completed", {"op": "get", "key": key, "success": True, "request_id": request_id})
                    return decoded
                EventBus.publish("redis_op_completed", {"op": "get", "key": key, "success": False, "request_id": request_id})
            except Exception as e:
                _redis_manager.handle_error(e, request_id)
                EventBus.publish("redis_op_completed", {"op": "get", "key": key, "success": False, "error": str(e), "request_id": request_id})

        EventBus.publish("cache_miss", {"key": key, "request_id": request_id})
        return None

    @classmethod
    def set(cls, key: str, value: str, ttl: Optional[int] = None, request_id: Optional[str] = None):
        from backend.observability.event_bus import EventBus
        if ttl is None:
            from backend.streaming.config import StreamingConfig
            ttl = StreamingConfig.CACHE_DEFAULT_TTL
            
        req_prefix = f"[{request_id}] " if request_id else ""
        now = time.time()
        expires_at = (now + ttl) if ttl else None
        
        # Populate Memory Cache
        _MEM_CACHE[key] = (value, expires_at)
        logger.debug(f"{req_prefix}[Cache] Saved to Level 1 (Memory) cache (TTL: {ttl}s).")

        # Populate Redis Cache
        r = _redis_manager.get_client(request_id)
        if r:
            EventBus.publish("redis_op_started", {"op": "set", "key": key, "request_id": request_id})
            try:
                if ttl:
                    r.setex(key, ttl, value)
                else:
                    r.set(key, value)
                logger.debug(f"{req_prefix}[Cache] Saved to Level 2 (Redis) cache.")
                EventBus.publish("redis_op_completed", {"op": "set", "key": key, "success": True, "request_id": request_id})
            except Exception as e:
                _redis_manager.handle_error(e, request_id)
                EventBus.publish("redis_op_completed", {"op": "set", "key": key, "success": False, "error": str(e), "request_id": request_id})

    @classmethod
    def delete(cls, key: str, request_id: Optional[str] = None):
        from backend.observability.event_bus import EventBus
        """Invalidates a cache entry (e.g. if final validation blocks or fails)."""
        req_prefix = f"[{request_id}] " if request_id else ""
        if key in _MEM_CACHE:
            del _MEM_CACHE[key]
            logger.debug(f"{req_prefix}[Cache] Removed from Level 1 (Memory) cache.")
            
        r = _redis_manager.get_client(request_id)
        if r:
            EventBus.publish("redis_op_started", {"op": "delete", "key": key, "request_id": request_id})
            try:
                r.delete(key)
                logger.debug(f"{req_prefix}[Cache] Removed from Level 2 (Redis) cache.")
                EventBus.publish("redis_op_completed", {"op": "delete", "key": key, "success": True, "request_id": request_id})
            except Exception as e:
                _redis_manager.handle_error(e, request_id)
                EventBus.publish("redis_op_completed", {"op": "delete", "key": key, "success": False, "error": str(e), "request_id": request_id})

