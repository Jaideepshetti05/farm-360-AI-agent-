import time
from typing import Dict, Any, Optional, List, Tuple
from pydantic import BaseModel, Field
from backend.streaming.stream_context import StreamState

class StreamMetrics(BaseModel):
    start_time: float = Field(default_factory=time.time)
    first_token_time: Optional[float] = None
    complete_time: Optional[float] = None
    token_count: int = 0
    provider: str = ""
    model: str = ""
    error_count: int = 0
    state_transitions: List[Tuple[StreamState, float]] = Field(default_factory=list)

    @property
    def ttft_ms(self) -> Optional[float]:
        """Time to First Token in milliseconds."""
        if self.first_token_time is not None:
            return (self.first_token_time - self.start_time) * 1000.0
        return None

    @property
    def total_duration_ms(self) -> float:
        end = self.complete_time or time.time()
        return (end - self.start_time) * 1000.0

    @property
    def tokens_per_second(self) -> float:
        duration = (self.complete_time or time.time()) - self.start_time
        if duration > 0:
            return self.token_count / duration
        return 0.0

    def record_first_token(self):
        if self.first_token_time is None:
            self.first_token_time = time.time()

    def increment_tokens(self, count: int = 1):
        self.token_count += count

    def finish(self):
        self.complete_time = time.time()

    def get_state_timings(self) -> Dict[str, float]:
        """Calculates time spent in each state in milliseconds."""
        timings = {}
        if len(self.state_transitions) < 2:
            return timings
            
        for i in range(len(self.state_transitions) - 1):
            state, t1 = self.state_transitions[i]
            next_state, t2 = self.state_transitions[i+1]
            timings[f"state_{state}_to_{next_state}_ms"] = (t2 - t1) * 1000.0
        return timings
