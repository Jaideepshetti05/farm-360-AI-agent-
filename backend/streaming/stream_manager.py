import time
import re
import asyncio
from typing import AsyncGenerator, Dict, Any
from loguru import logger
from backend.services.cache_service import CacheService
from backend.streaming.stream_response import SSESerializer
from backend.validator.engine import ValidationEngine

class StreamManager:
    def __init__(self):
        self.validator_engine = ValidationEngine()

    def _validate_chunk(self, chunk: str) -> bool:
        """Lightweight per-token regex checks for secrets/leaks."""
        key_patterns = [
            r"sk-[a-zA-Z0-9]{10,}",
            r"AIzaSy[a-zA-Z0-9_-]{10}"
        ]
        for pattern in key_patterns:
            if re.search(pattern, chunk):
                return False
        return True

    async def stream_query(
        self,
        query: str,
        context: Dict[str, Any],
        prompt_key: str = "general_assistant"
    ) -> AsyncGenerator[str, None]:
        """
        Coordinates connection streaming, cache checks, per-token sanitizers,
        and post-stream validator execution.
        """
        cache_key = CacheService.generate_key(
            prompt_version="1.0.0",
            model="gemini-2.5-flash",
            provider="gemini",
            language=context.get("language", "en"),
            context={"query": query, "profile": context.get("user_profile")}
        )
        
        cached = CacheService.get(cache_key)
        if cached:
            yield SSESerializer.format_event("start", {})
            for word in cached.split(" "):
                yield SSESerializer.format_event("token", word + " ")
                await asyncio.sleep(0.01)
            yield SSESerializer.format_event("complete", {})
            return

        yield SSESerializer.format_event("start", {})
        
        from backend.router.router import IntentRouter
        router = IntentRouter()
        
        advisors_res = await router.route(query, context)
        full_response = advisors_res.response_text
        
        words = full_response.split(" ")
        assembled_chunks = []
        
        for word in words:
            chunk = word + " "
            if not self._validate_chunk(chunk):
                logger.error("[StreamManager] Suspicious signature detected in token chunk. Closing stream.")
                yield SSESerializer.format_event("error", "Safety validation exception.")
                return
            assembled_chunks.append(chunk)
            yield SSESerializer.format_event("token", chunk)
            await asyncio.sleep(0.02)  # simulate stream interval

        assembled_text = "".join(assembled_chunks)
        
        validated_text, results = await self.validator_engine.validate_response(assembled_text, context)
        
        is_blocked = any(r.status == "BLOCK" for r in results)
        if is_blocked:
            logger.error("[StreamManager] Response blocked by Validator. Clearing caches.")
            CacheService.delete(cache_key)
            yield SSESerializer.format_event("error", "Response blocked due to safety guidelines.")
            return
            
        CacheService.set(cache_key, validated_text)
        
        try:
            from backend.services.database_service import UnitOfWork
            from backend.models.database import ConversationHistory
            
            async def _save():
                async with UnitOfWork() as uow:
                    session_id = context.get("session_id", "default_session")
                    uow.session.add(ConversationHistory(session_id=session_id, role="user", content=query))
                    uow.session.add(ConversationHistory(session_id=session_id, role="assistant", content=validated_text))
                    await uow.session.commit()
            await _save()
        except Exception as e:
            logger.warning(f"[StreamManager] Failed to persist message turn: {e}")
            
        yield SSESerializer.format_event("complete", {})
