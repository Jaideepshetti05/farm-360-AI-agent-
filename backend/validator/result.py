from typing import List, Dict, Any

class ValidationResult:
    """Contains outcome metrics of individual validator executions."""
    def __init__(
        self,
        status: str,  # PASS, WARNING, FAIL, BLOCK
        score: float, # 0.0 to 1.0
        issues: List[str] = None,
        recommendations: List[str] = None,
        execution_time_ms: float = 0.0,
        metadata: Dict[str, Any] = None
    ):
        self.status = status
        self.score = score
        self.issues = issues or []
        self.recommendations = recommendations or []
        self.execution_time_ms = execution_time_ms
        self.metadata = metadata or {}

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "score": self.score,
            "issues": self.issues,
            "recommendations": self.recommendations,
            "execution_time_ms": self.execution_time_ms,
            "metadata": self.metadata
        }
