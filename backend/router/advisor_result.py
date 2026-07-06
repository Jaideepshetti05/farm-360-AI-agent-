from typing import Dict, Any

class AdvisorResult:
    """Standardized response object returned by all domain advisors."""
    def __init__(
        self,
        response_text: str,
        confidence: float,
        advisor_name: str,
        execution_time: float,
        metadata: Dict[str, Any] = None
    ):
        self.response_text = response_text
        self.confidence = confidence
        self.advisor_name = advisor_name
        self.execution_time = execution_time
        self.metadata = metadata or {}

    def to_dict(self) -> dict:
        return {
            "response_text": self.response_text,
            "confidence": self.confidence,
            "advisor_name": self.advisor_name,
            "execution_time": self.execution_time,
            "metadata": self.metadata
        }
