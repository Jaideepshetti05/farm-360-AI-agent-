from typing import Optional, Dict
from backend.observability import EventBus

class MemoryMetrics:
    """Helper to record Memory V2 metrics to the central EventBus for telemetry collectors."""

    @staticmethod
    def record_lookup_latency(duration_ms: float, category: Optional[str] = None):
        """Publishes the memory retrieval latency."""
        EventBus.publish("metrics_recorded", {
            "category": "Memory",
            "metric_name": "farm360_memory_lookup_ms",
            "value": duration_ms,
            "tags": {"memory_category": category or "all"}
        })

    @staticmethod
    def record_retrieval_hit(count: int, category: Optional[str] = None):
        """Records a hit metric for matching memories retrieved."""
        EventBus.publish("metrics_recorded", {
            "category": "Memory",
            "metric_name": "farm360_memory_hits_total",
            "value": float(count),
            "tags": {"memory_category": category or "all"}
        })

    @staticmethod
    def record_forgetting_event(action: str, memory_id: str):
        """Records forgetting status shifts (archive, compress, delete)."""
        EventBus.publish("metrics_recorded", {
            "category": "Memory",
            "metric_name": "farm360_memory_forgetting_total",
            "value": 1.0,
            "tags": {"action": action, "memory_id": memory_id}
        })
