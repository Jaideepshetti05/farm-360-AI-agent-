from backend.evaluation.metrics.base import BaseMetric
from typing import Dict, Any

class KeywordMatch(BaseMetric):
    def __init__(self):
        super().__init__("keyword_match")

    async def evaluate(self, response: str, test_case: Dict[str, Any]) -> float:
        keywords = test_case.get("expected_keywords", [])
        if not keywords:
            return 1.0  # Vacuously true if no keywords expected
            
        actual_lower = response.lower()
        matches = sum(1 for kw in keywords if kw.lower() in actual_lower)
        return float(matches / len(keywords))
