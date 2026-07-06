import os

class StreamingConfig:
    # Caching TTL configurations
    CACHE_DEFAULT_TTL = int(os.environ.get("STREAM_CACHE_TTL", 3600))
    
    # Pool configurations
    CONNECTION_POOL_SIZE = int(os.environ.get("STREAM_POOL_SIZE", 20))
    
    # SSE Heartbeat settings
    HEARTBEAT_INTERVAL_SECONDS = float(os.environ.get("STREAM_HEARTBEAT_INTERVAL", 1.0))
    
    # Streaming timeouts
    CHUNK_TIMEOUT_SECONDS = float(os.environ.get("STREAM_CHUNK_TIMEOUT", 10.0))
    
    # Default Provider Failover order
    PROVIDER_FAILOVER_ORDER = ["gemini", "openrouter", "openai"]
