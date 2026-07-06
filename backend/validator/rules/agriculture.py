import re
import time
from typing import Dict, Any
from backend.validator.interfaces import BaseValidator
from backend.validator.result import ValidationResult
from backend.validator.config import ValidatorConfig

class AgricultureValidator(BaseValidator):
    def __init__(self):
        super().__init__("AgricultureValidator", severity="warning")

    def _extract_dosage(self, text: str, chemical: str) -> float:
        """Helper to extract dosage numbers near chemical occurrences."""
        pattern = rf"{chemical}.*?(\d+(?:\.\d+)?)\s*(?:kg/ha|kg/hectare|g/L|g/l)"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass
        return 0.0

    async def validate(self, text: str, context: Dict[str, Any]) -> ValidationResult:
        start_time = time.time()
        issues = []
        recommendations = []
        
        text_lower = text.lower()
        if "glyphosate" in text_lower:
            dosage = self._extract_dosage(text, "glyphosate")
            if dosage > ValidatorConfig.MAX_GLYPHOSATE_DOSAGE_KG_HA:
                issues.append(f"Glyphosate dosage ({dosage} kg/ha) exceeds safe threshold ({ValidatorConfig.MAX_GLYPHOSATE_DOSAGE_KG_HA} kg/ha).")
                recommendations.append("Reduce recommended glyphosate dosage to safe thresholds.")
                
        if "urea" in text_lower:
            dosage = self._extract_dosage(text, "urea")
            if dosage > ValidatorConfig.MAX_UREA_DOSAGE_KG_HA:
                issues.append(f"Urea dosage ({dosage} kg/ha) exceeds recommended standard nitrogen fertilization limits.")
                recommendations.append("Verify urea application requirements or add soil testing warning details.")
                
        livestock_terms = ["cow", "cattle", "buffalo", "sick", "veterinary", "vaccine", "medicine", "treatment"]
        if any(term in text_lower for term in livestock_terms):
            disclaimer_triggers = ["consult a veterinarian", "disclaimer", "veterinary officer", "extension officer"]
            if not any(trigger in text_lower for trigger in disclaimer_triggers):
                issues.append("Veterinary advice detected without safety consultation disclaimer.")
                recommendations.append("Add standard veterinary warning disclaimer: 'Please consult a local veterinary officer before administering treatments.'")
                
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
