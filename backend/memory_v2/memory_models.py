from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field
import uuid

class MemoryRecordModel(BaseModel):
    """Pydantic model representing a memory record in the Memory V2 subsystem."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    category: str = Field(..., description="working | short_term | long_term_semantic | episodic")
    content: str
    importance: float = 1.0
    recency: float = 1.0
    access_frequency: int = 1
    confidence: float = 1.0
    embedding: Optional[List[float]] = None
    source: Optional[str] = "unknown"
    tags: List[str] = Field(default_factory=list)
    expiration_policy: Optional[str] = None  # e.g., "TTL:86400"
    status: str = "active"  # active | archived | compressed | deleted
    feedback_score: float = 0.0

    class Config:
        from_attributes = True
