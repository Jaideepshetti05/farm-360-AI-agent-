import os

class ObservabilityConfig:
    """Centralized configuration values for metrics, logging, alerts and tracing."""
    METRICS_ENABLED: bool = os.environ.get("OBS_METRICS_ENABLED", "true").lower() == "true"
    TRACING_ENABLED: bool = os.environ.get("OBS_TRACING_ENABLED", "true").lower() == "true"
    SAMPLING_RATE: float = float(os.environ.get("OBS_SAMPLING_RATE", "1.0"))
    
    ALERT_THRESHOLDS = {
        "slow_request_ms": float(os.environ.get("OBS_ALERT_SLOW_REQUEST", "5000.0")),
        "failed_keys_max": int(os.environ.get("OBS_ALERT_FAILED_KEYS", "3")),
        "ttft_max_ms": float(os.environ.get("OBS_ALERT_TTFT_MAX", "2000.0"))
    }
    
    EXPORT_INTERVAL_SEC: int = int(os.environ.get("OBS_EXPORT_INTERVAL", "15"))
    RETENTION_DAYS: int = int(os.environ.get("OBS_RETENTION_DAYS", "7"))
