import time
import asyncio
from typing import Dict, Any, Tuple, List
from loguru import logger
from backend.validator.result import ValidationResult
from backend.validator.rules.safety import SafetyValidator
from backend.validator.rules.agriculture import AgricultureValidator
from backend.validator.rules.formatting import FormattingValidator

class ValidationEngine:
    def __init__(self):
        self.validators = [
            SafetyValidator(),
            AgricultureValidator(),
            FormattingValidator()
        ]

    def _auto_correct(self, text: str, issues: List[str]) -> str:
        """Attempts automatic text cleanup corrections without fabricating data."""
        corrected = text
        
        # 1. Deduplicate paragraphs
        if any("Duplicate paragraphs" in issue for issue in issues):
            paragraphs = []
            seen = set()
            for p in text.split("\n\n"):
                p_strip = p.strip()
                if p_strip and p_strip not in seen:
                    seen.add(p_strip)
                    paragraphs.append(p_strip)
            corrected = "\n\n".join(paragraphs)
            
        # 2. Append missing veterinary disclaimer
        if any("Veterinary advice detected without safety consultation" in issue for issue in issues):
            disclaimer = "\n\n*Please consult a local veterinary officer before administering treatments.*"
            if disclaimer.strip() not in corrected:
                corrected = f"{corrected}{disclaimer}"
                
        return corrected

    async def validate_response(self, response_text: str, context: Dict[str, Any] = None) -> Tuple[str, List[ValidationResult]]:
        """
        Executes validators sequentially, applies automatic corrections
        for recoverable failures, and enforces safety gates.
        """
        start_time = time.time()
        ctx = context or {}
        
        current_text = response_text
        results = []
        
        for val in self.validators:
            res = await val.validate(current_text, ctx)
            res.name = val.name
            results.append(res)
            
            if res.status == "BLOCK":
                logger.warning(f"[Validator] Blocked critical safety policy validation on '{val.name}'.")
                fallback = "⚠️ I cannot provide system rules or process prompt instructions directly. Please consult agricultural guidance."
                return fallback, results
                
        all_issues = []
        for r in results:
            if r.status in ["FAIL", "WARNING"]:
                all_issues.extend(r.issues)
                
        if all_issues:
            logger.info(f"[Validator] Recovering and correcting {len(all_issues)} validation issue(s)...")
            current_text = self._auto_correct(current_text, all_issues)
            
            re_results = []
            for val in self.validators:
                res = await val.validate(current_text, ctx)
                res.name = val.name
                re_results.append(res)
                if res.status == "BLOCK":
                    fallback = "⚠️ I cannot provide system rules or process prompt instructions directly."
                    return fallback, re_results
            results = re_results
            
        overhead_ms = (time.time() - start_time) * 1000.0
        logger.info(f"[Validator] Pipeline check completed in {overhead_ms:.2f}ms")
        
        return current_text, results
