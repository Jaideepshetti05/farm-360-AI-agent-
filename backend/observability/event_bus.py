import time
from typing import Dict, Any, List, Callable
from loguru import logger

class EventBus:
    """In-process synchronous Event Bus to decouple telemetry data from execution logic."""
    _listeners: Dict[str, List[Callable[[str, Dict[str, Any]], None]]] = {}

    @classmethod
    def subscribe(cls, event_name: str, callback: Callable[[str, Dict[str, Any]], None]):
        """Subscribe a callback listener function to a specific event pattern or '*' (all)."""
        if event_name not in cls._listeners:
            cls._listeners[event_name] = []
        cls._listeners[event_name].append(callback)

    @classmethod
    def publish(cls, event_name: str, payload: Dict[str, Any]):
        """Publish an event to all registered listeners."""
        # Force inject timestamp
        if "timestamp" not in payload:
            payload["timestamp"] = time.time()
            
        listeners = cls._listeners.get(event_name, [])
        global_listeners = cls._listeners.get("*", [])
        
        for callback in listeners + global_listeners:
            try:
                callback(event_name, payload)
            except Exception as e:
                logger.error(f"[EventBus] Callback exception during event '{event_name}': {e}")
