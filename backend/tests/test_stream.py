import sys
import os
import asyncio
import unittest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from backend.core.database import engine
from backend.models.database import Base
from backend.streaming.stream_manager import StreamManager
from backend.services.cache_service import CacheService

class TestStreamingPerformance(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def asyncTearDown(self):
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)

    def test_cache_namespaces(self):
        key1 = CacheService.generate_key("1.0.0", "gemini", "native", "en", {"q": "rice"})
        key2 = CacheService.generate_key("1.0.0", "gemini", "native", "en", {"q": "rice"})
        key3 = CacheService.generate_key("1.0.0", "gemini", "native", "en", {"q": "wheat"})
        
        self.assertEqual(key1, key2)
        self.assertNotEqual(key1, key3)

    def test_sse_event_serialization(self):
        from backend.streaming.stream_response import SSESerializer
        event = SSESerializer.format_event("token", "data_content")
        self.assertIn("event: token", event)
        self.assertIn("data: data_content", event)

    def test_per_token_validation_interception(self):
        manager = StreamManager()
        self.assertFalse(manager._validate_chunk("sk-12345abcde67890"))
        self.assertTrue(manager._validate_chunk("Standard advice content."))

    async def test_full_stream_resolution(self):
        manager = StreamManager()
        context = {
            "user_profile": {"location": "Assam"},
            "language": "en",
            "session_id": "test_stream_session"
        }
        
        events = []
        async for event in manager.stream_query("How to grow rice?", context):
            events.append(event)
            
        self.assertTrue(len(events) > 0)
        self.assertIn("event: start", events[0])
        self.assertIn("event: complete", events[-1])

if __name__ == "__main__":
    unittest.main()
