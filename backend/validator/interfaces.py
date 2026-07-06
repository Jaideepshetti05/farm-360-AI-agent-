from typing import Dict, Any
from backend.validator.result import ValidationResult

class BaseValidator:
    """Interface that must be implemented by all safety and advice checkers."""
    def __init__(self, name: str, severity: str = "warning"):
        self.name = name
        self.val_severity = severity

    async def validate(self, text: str, context: Dict[str, Any]) -> ValidationResult:
        """
        Runs validation checks on output response text.
        Returns a ValidationResult containing status and issues.
        """
        raise NotImplementedError
