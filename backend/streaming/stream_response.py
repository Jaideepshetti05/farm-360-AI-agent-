import json
from abc import ABC, abstractmethod
from typing import Any, Optional
from backend.streaming.stream_events import StreamEvent

class TransportAdapter(ABC):
    @abstractmethod
    def serialize(self, event: StreamEvent) -> Any:
        pass

class SSETransportAdapter(TransportAdapter):
    def serialize(self, event: StreamEvent) -> str:
        """Converts a strongly-typed StreamEvent into a standard SSE formatted chunk."""
        lines = [f"event: {event.event_type}"]
        if event.stream_id:
            lines.append(f"id: {event.stream_id}_{event.sequence}")
        lines.append("retry: 5000")
        
        # Dump model data
        data_str = json.dumps(event.model_dump())
        lines.append(f"data: {data_str}")
        return "\n".join(lines) + "\n\n"

# ── Legacy SSESerializer (Backward Compatibility) ───────────────────────────
class SSESerializer:
    @staticmethod
    def format_event(
        event_type: str,
        data: Any,
        message_id: Optional[str] = None,
        retry: int = 5000
    ) -> str:
        """Formats standard Server-Sent Events outputs. Kept for legacy support."""
        lines = [f"event: {event_type}"]
        if message_id:
            lines.append(f"id: {message_id}")
        lines.append(f"retry: {retry}")
        
        if isinstance(data, (dict, list)):
            data_str = json.dumps(data)
        else:
            data_str = str(data)
            
        lines.append(f"data: {data_str}")
        return "\n".join(lines) + "\n\n"
