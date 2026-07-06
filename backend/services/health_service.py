from typing import Dict, Any
import os
import psutil
from loguru import logger
from sqlalchemy import select
from backend.core.database import async_session
from backend.config import settings

class HealthService:
    @staticmethod
    async def check_health() -> Dict[str, Any]:
        health_status = {
            "status": "healthy",
            "postgres": "healthy",
            "redis": "healthy",
            "disk_storage": "healthy",
            "details": {}
        }
        
        # 1. PostgreSQL check
        try:
            async with async_session() as session:
                await session.execute(select(1))
            health_status["details"]["postgres"] = "Online (SELECT 1 passed)"
        except Exception as e:
            logger.error(f"[Health] PostgreSQL health check failed: {e}")
            health_status["postgres"] = "unhealthy"
            health_status["status"] = "degraded"
            health_status["details"]["postgres"] = f"Offline: {str(e)}"
            
        # 2. Redis check
        try:
            import redis
            redis_host = os.environ.get("REDIS_HOST", "localhost")
            redis_port = int(os.environ.get("REDIS_PORT", 6379))
            r = redis.Redis(host=redis_host, port=redis_port, socket_timeout=1.0)
            r.ping()
            health_status["details"]["redis"] = "Online (Ping passed)"
        except ImportError:
            health_status["details"]["redis"] = "Client package 'redis' not installed. Running in local DB-only mode."
        except Exception as e:
            logger.warning(f"[Health] Redis health check failed: {e}")
            health_status["redis"] = "unhealthy"
            # Caching is non-critical, so we mark redis degraded but keep status healthy/degraded based on postgres
            health_status["details"]["redis"] = f"Offline: {str(e)}"

        # 3. Disk storage check
        try:
            usage = psutil.disk_usage(".")
            health_status["details"]["disk_free_gb"] = round(usage.free / (1024**3), 2)
            if usage.percent > 95:
                health_status["disk_storage"] = "warning"
                health_status["status"] = "degraded"
        except Exception as e:
            health_status["details"]["disk_storage"] = f"Check failed: {str(e)}"

        return health_status
