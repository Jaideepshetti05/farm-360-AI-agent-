from backend.evaluation.metrics.base import BaseMetric
from typing import Dict, Any

class ExactMatch(BaseMetric):
    def __init__(self):
        super().__init__("exact_match")

    async def evaluate(self, response: str, test_case: Dict[str, Any]) -> float:
        expected = test_case.get("expected_answer", "").strip().lower()
        actual = response.strip().lower()
        return 1.0 if expected == actual else 0.0
