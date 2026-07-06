from backend.evaluation.metrics.base import BaseMetric
from typing import Dict, Any

class SemanticScore(BaseMetric):
    def __init__(self):
        super().__init__("semantic_score")

    async def evaluate(self, response: str, test_case: Dict[str, Any]) -> float:
        # Completeness: Check required markdown sections exist in actual response
        required = test_case.get("required_sections", [])
        if not required:
            return 1.0
            
        actual_lower = response.lower()
        found_sections = 0
        for section in required:
            sec_lower = section.lower()
            if sec_lower in actual_lower:
                found_sections += 1
                
        return float(found_sections / len(required))
