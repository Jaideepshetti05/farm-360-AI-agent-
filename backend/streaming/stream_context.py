import time
import uuid
from typing import Any, Dict, List, Optional, Tuple, Literal
from pydantic import BaseModel, Field

StreamState = Literal[
    "Created",
    "Initializing",
    "CacheLookup",
    "Routing",
    "Generating",
    "Validating",
    "Saving",
    "Completed",
    "Cancelled",
    "Failed"
]

class StreamContext(BaseModel):
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = "default_session"
    query: str = ""
    user_profile: Dict[str, Any] = Field(default_factory=dict)
    language: str = "en"
    prompt_version: str = "1.0.0"
    model: str = "gemini-2.5-flash"
    provider: str = "gemini"
    start_time: float = Field(default_factory=time.time)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    current_state: StreamState = "Created"
    state_transitions: List[Tuple[StreamState, float]] = Field(default_factory=list)

    def __init__(self, **data):
        super().__init__(**data)
        if not self.state_transitions:
            self.state_transitions.append(("Created", self.start_time))

    def transition_to(self, new_state: StreamState):
        """Transition stream state and record timestamp."""
        now = time.time()
        self.current_state = new_state
        self.state_transitions.append((new_state, now))
