import math
from datetime import datetime
from typing import Optional
from backend.memory_v2.memory_models import MemoryRecordModel

class MemoryRanker:
    """Calculates relative relevance scores for candidate memory records."""
    
    @staticmethod
    def calculate_recency(timestamp: datetime, current_time: Optional[datetime] = None, half_life_hours: float = 168.0) -> float:
        """Calculates exponential time decay. Default half-life of 7 days (168 hours)."""
        if current_time is None:
            current_time = datetime.utcnow()
            
        elapsed_seconds = (current_time - timestamp).total_seconds()
        elapsed_hours = max(0.0, elapsed_seconds / 3600.0)
        
        # λ = ln(2) / half_life
        decay_constant = 0.69314718 / half_life_hours
        return math.exp(-decay_constant * elapsed_hours)

    @classmethod
    def score_memory(
        cls,
        record: MemoryRecordModel,
        semantic_similarity: float = 1.0,
        current_time: Optional[datetime] = None
    ) -> float:
        """
        Calculates the Final Score using the formula:
        Final Score = Importance * Recency * Access Frequency * Semantic Similarity * User Feedback
        """
        importance = max(0.1, record.importance)
        
        # Calculate exponential decay
        recency = cls.calculate_recency(record.timestamp, current_time=current_time)
        
        # Logarithmic scaling for access frequency to avoid dominance
        frequency_score = 1.0 + 0.2 * math.log(max(1, record.access_frequency))
        
        # User feedback multiplier (feedback_score defaults to 0.0)
        feedback_multiplier = 1.0 + max(-0.9, record.feedback_score)
        
        # Standardize similarity minimum to avoid zeroing out completely on keyword hits
        similarity_weight = max(0.01, semantic_similarity)
        
        final_score = importance * recency * frequency_score * similarity_weight * feedback_multiplier
        return round(final_score, 4)
