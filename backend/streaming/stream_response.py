import json
from typing import Dict, Any, Optional

class SSESerializer:
    @staticmethod
    def format_event(
        event_type: str,
        data: Any,
        message_id: Optional[str] = None,
        retry: int = 5000
    ) -> str:
        """Formats standard Server-Sent Events outputs."""
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
