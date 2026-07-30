import uuid
import time
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field

class TraceContext(BaseModel):
    """
    Model representing distributed trace context data.
    Ensures OpenTelemetry compatibility by tracking span parent-child hierarchies.
    """
    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    span_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    parent_span_id: Optional[str] = None
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    workflow_id: Optional[str] = None
    session_id: str = "default_session"
    user_id_hashed: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    prompt_version: Optional[str] = "1.0.0"
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def create_child(self) -> "TraceContext":
        """Spawns a child TraceContext linking the parent span ID."""
        child = self.model_copy()
        child.parent_span_id = self.span_id
        child.span_id = str(uuid.uuid4())
        return child
