import re
import time
from typing import Dict, Any
from backend.validator.interfaces import BaseValidator
from backend.validator.result import ValidationResult

class SafetyValidator(BaseValidator):
    def __init__(self):
        super().__init__("SafetyValidator", severity="block")

    async def validate(self, text: str, context: Dict[str, Any]) -> ValidationResult:
        start_time = time.time()
        issues = []
        
        # 1. Jailbreak and Prompt Echo checks
        leak_triggers = [
            "ignore previous instructions",
            "you are farm360 ai",
            "system prompt rules",
            "farmer profile parameters"
        ]
        
        text_lower = text.lower()
        for trigger in leak_triggers:
            if trigger in text_lower:
                issues.append(f"Prompt leakage or jailbreak pattern detected: '{trigger}'")
                
        # 2. Secret and Credential leaks (JWT, SSH, API Keys)
        key_patterns = [
            r"sk-[a-zA-Z0-9]{32,}",
            r"AIzaSy[a-zA-Z0-9_-]{33}",
            r"bearer\s+[a-zA-Z0-9_\-\.]+"
        ]
        
        for pattern in key_patterns:
            if re.search(pattern, text):
                issues.append("Internal API token or credential signature leakage detected.")
                
        status = "BLOCK" if issues else "PASS"
        score = 0.0 if issues else 1.0
        
        elapsed = (time.time() - start_time) * 1000.0
        return ValidationResult(
            status=status,
            score=score,
            issues=issues,
            recommendations=["Refuse execution and yield safe fallback prompt reply."] if issues else [],
            execution_time_ms=elapsed
        )
