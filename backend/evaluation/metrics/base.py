from typing import Dict, Any

class BaseMetric:
    """Base interface for all evaluation metrics."""
    def __init__(self, name: str):
        self.name = name

    async def evaluate(self, response: str, test_case: Dict[str, Any]) -> float:
        """
        Evaluate the response text against a test case specification.
        Returns a score between 0.0 (complete failure) and 1.0 (perfect match).
        """
        raise NotImplementedError
