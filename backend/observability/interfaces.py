from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class MetricsCollector(ABC):
    @abstractmethod
    def record_counter(self, name: str, value: float = 1.0, tags: Optional[Dict[str, str]] = None):
        pass
        
    @abstractmethod
    def record_gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        pass

class TraceCollector(ABC):
    @abstractmethod
    def start_span(self, name: str, context: Any) -> Any:
        pass
        
    @abstractmethod
    def end_span(self, span: Any):
        pass

class EventCollector(ABC):
    @abstractmethod
    def publish_event(self, event_name: str, payload: Dict[str, Any]):
        pass
