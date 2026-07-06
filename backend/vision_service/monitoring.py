"""
Farm360 AI – Vision Metrics & Monitoring
==========================================
Simple, thread-safe in-process metrics counter for tracking vision requests,
errors, and latencies. Integrated with backend health status.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import threading
from typing import Dict, Any


@dataclass
class TaskMetrics:
    request_count: int = 0
    error_count: int = 0
    total_latency_ms: float = 0.0

    @property
    def avg_latency_ms(self) -> float:
        if self.request_count == 0:
            return 0.0
        return round(self.total_latency_ms / self.request_count, 2)


class VisionMetricsManager:
    """Thread-safe metrics manager for tracking vision service usage."""

    def __init__(self):
        self._metrics: dict[str, TaskMetrics] = {}
        self._lock = threading.Lock()

    def record_request(self, task: str, latency_ms: float, success: bool = True):
        """Record an inference call for a specific task."""
        with self._lock:
            if task not in self._metrics:
                self._metrics[task] = TaskMetrics()
            
            metrics = self._metrics[task]
            metrics.request_count += 1
            metrics.total_latency_ms += latency_ms
            if not success:
                metrics.error_count += 1

    def get_metrics(self) -> Dict[str, Any]:
        """Get summary dict of all recorded metrics."""
        with self._lock:
            summary = {}
            for task, m in self._metrics.items():
                summary[task] = {
                    "request_count": m.request_count,
                    "error_count": m.error_count,
                    "avg_latency_ms": m.avg_latency_ms,
                }
            return summary


# Singleton instance
metrics_manager = VisionMetricsManager()
