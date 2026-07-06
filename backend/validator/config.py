import os

class ValidatorConfig:
    # Enabled checks
    ENABLE_SAFETY_CHECKS = True
    ENABLE_AGRICULTURAL_CHECKS = True
    ENABLE_FORMATTING_CHECKS = True
    
    # Thresholds
    MAX_GLYPHOSATE_DOSAGE_KG_HA = 5.0  # Warning limit for weedicide
    MAX_UREA_DOSAGE_KG_HA = 250.0      # Warning limit for nitrogen fertilizer
    
    # Severity levels
    DEFAULT_POLICY = "balanced"  # strict, balanced, relaxed
