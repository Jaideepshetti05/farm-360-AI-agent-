from abc import ABC, abstractmethod
from typing import AsyncGenerator, Dict, Any, Optional

class IStreamManager(ABC):
    @abstractmethod
    async def stream_query(
        self,
        query: str,
        context: Dict[str, Any],
        prompt_key: str = "general_assistant",
        request_id: Optional[str] = None
    ) -> AsyncGenerator[Any, None]:  # Yields StreamEvent objects
        pass

class IStreamingProvider(ABC):
    @abstractmethod
    async def stream_completion_async(
        self,
        messages: list[dict],
        request_id: str,
        temperature: float = 0.7,
        max_tokens: int = 2048
    ) -> AsyncGenerator[str, None]:  # Yields raw string tokens
        pass
