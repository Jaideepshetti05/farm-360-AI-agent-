import time
from typing import Any, Dict, Optional, Literal
from pydantic import BaseModel, Field

EventType = Literal["start", "token", "progress", "heartbeat", "warning", "error", "complete"]

class StreamEvent(BaseModel):
    event_type: EventType
    timestamp: float = Field(default_factory=time.time)
    stream_id: str
    sequence: int
    provider: str = ""
    model: str = ""
    payload: Any = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    is_terminal: bool = False
