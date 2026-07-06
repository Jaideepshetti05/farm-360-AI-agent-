import time
from typing import Dict, Any
from backend.validator.interfaces import BaseValidator
from backend.validator.result import ValidationResult

class FormattingValidator(BaseValidator):
    def __init__(self):
        super().__init__("FormattingValidator", severity="warning")

    async def validate(self, text: str, context: Dict[str, Any]) -> ValidationResult:
        start_time = time.time()
        issues = []
        recommendations = []
        
        if len(text.strip()) < 10:
            issues.append("Response is too short or empty.")
            
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        unique_paragraphs = set(paragraphs)
        if len(paragraphs) != len(unique_paragraphs):
            issues.append("Duplicate paragraphs or text blocks detected.")
            recommendations.append("Apply deduplication cleanups on the response.")
            
        status = "FAIL" if issues else "PASS"
        score = 0.50 if issues else 1.0
        
        elapsed = (time.time() - start_time) * 1000.0
        return ValidationResult(
            status=status,
            score=score,
            issues=issues,
            recommendations=recommendations,
            execution_time_ms=elapsed
        )
