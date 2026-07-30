import json
import time
from typing import Any, Dict, Optional
from loguru import logger
from backend.observability.trace_context import TraceContext

def log_structured(
    level: str,
    component: str,
    operation: str,
    message: str,
    trace_ctx: Optional[TraceContext] = None,
    duration_ms: Optional[float] = None,
    status: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
):
    """Writes a standardized, structured JSON log statement to Loguru."""
    record = {
        "timestamp": time.time(),
        "level": level.upper(),
        "request_id": trace_ctx.request_id if trace_ctx else None,
        "trace_id": trace_ctx.trace_id if trace_ctx else None,
        "workflow_id": trace_ctx.workflow_id if trace_ctx else None,
        "component": component,
        "operation": operation,
        "duration_ms": duration_ms,
        "provider": trace_ctx.provider if trace_ctx else None,
        "model": trace_ctx.model if trace_ctx else None,
        "status": status,
        "message": message,
        "metadata": metadata or {}
    }
    
    log_msg = json.dumps(record)
    
    # Map level to loguru
    level_lower = level.lower()
    if level_lower == "debug":
        logger.debug(log_msg)
    elif level_lower == "info":
        logger.info(log_msg)
    elif level_lower == "warning":
        logger.warning(log_msg)
    elif level_lower == "error":
        logger.error(log_msg)
    else:
        logger.info(log_msg)
