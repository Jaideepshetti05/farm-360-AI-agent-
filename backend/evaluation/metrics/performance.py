from backend.evaluation.metrics.base import BaseMetric
from typing import Dict, Any

class PerformanceMetric(BaseMetric):
    def __init__(self):
        super().__init__("performance_metric")

    async def evaluate(self, response: str, test_case: Dict[str, Any]) -> float:
        rules = test_case.get("evaluation_rules", {})
        max_latency = rules.get("max_latency", 5.0)
        
        actual_latency = test_case.get("actual_latency", 0.0)
        if actual_latency <= max_latency:
            return 1.0
        excess = actual_latency - max_latency
        score = max(0.0, 1.0 - (excess / max_latency))
        return score
