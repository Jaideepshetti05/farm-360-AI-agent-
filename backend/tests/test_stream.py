import sys
import os
import asyncio
import time
import unittest
from unittest.mock import AsyncMock, patch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from backend.core.database import engine
from backend.models.database import Base
from backend.streaming.stream_manager import StreamManager
from backend.services.cache_service import CacheService
from backend.streaming.stream_response import SSESerializer, SSETransportAdapter
from backend.streaming.stream_events import StreamEvent

class TestStreamingPerformance(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from backend.services.cache_service import _redis_manager
        self.original_offline_until = _redis_manager.offline_until
        # Quarantine Redis connectivity for 1 hour by default to keep unit tests fast and offline
        _redis_manager.offline_until = time.time() + 3600
        
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def asyncTearDown(self):
        from backend.services.cache_service import _redis_manager
        _redis_manager.offline_until = self.original_offline_until
        
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)

    def test_cache_namespaces(self):
        key1 = CacheService.generate_key("1.0.0", "gemini-2.5-flash", "gemini", "en", {"q": "rice"})
        key2 = CacheService.generate_key("1.0.0", "gemini-2.5-flash", "gemini", "en", {"q": "rice"})
        key3 = CacheService.generate_key("1.0.0", "gemini-2.5-flash", "gemini", "en", {"q": "wheat"})
        
        self.assertEqual(key1, key2)
        self.assertNotEqual(key1, key3)

    def test_cache_expiry_ttl(self):
        # Set short TTL key
        CacheService.set("ttl_test_key", "val", ttl=1)
        self.assertEqual(CacheService.get("ttl_test_key"), "val")
        
        # Sleep for expiration
        time.sleep(1.1)
        self.assertIsNone(CacheService.get("ttl_test_key"))

    def test_redis_disconnect_fallback(self):
        from backend.services.cache_service import _redis_manager
        
        class BrokenRedis:
            def get(self, *args, **kwargs):
                raise Exception("Redis Refused Connection")
            def setex(self, *args, **kwargs):
                raise Exception("Redis Refused Connection")
            def ping(self, *args, **kwargs):
                raise Exception("Redis Refused Connection")

        # Mock Redis manager client to return broken client
        original_client = _redis_manager.client
        original_offline_until = _redis_manager.offline_until
        
        _redis_manager.client = BrokenRedis()
        _redis_manager.offline_until = 0.0  # reset quarantine cooldown

        # Should fallback gracefully to RAM caching without crash
        CacheService.set("fallback_key", "fallback_val", ttl=10)
        self.assertEqual(CacheService.get("fallback_key"), "fallback_val")

        # Restore client and offline settings
        _redis_manager.client = original_client
        _redis_manager.offline_until = original_offline_until

    def test_sse_event_serialization(self):
        # Legacy serializer check
        event = SSESerializer.format_event("token", "data_content")
        self.assertIn("event: token", event)
        self.assertIn("data: data_content", event)

        # SSETransportAdapter check
        adapter = SSETransportAdapter()
        se = StreamEvent(
            event_type="token",
            stream_id="req-1234",
            sequence=1,
            payload="adapted_content"
        )
        serialized = adapter.serialize(se)
        self.assertIn("event: token", serialized)
        self.assertIn("id: req-1234_1", serialized)
        self.assertIn("adapted_content", serialized)

    def test_per_token_validation_interception(self):
        manager = StreamManager()
        self.assertFalse(manager._validate_chunk("sk-12345abcde67890"))
        self.assertTrue(manager._validate_chunk("Standard advice content."))

    @patch("backend.provider_manager.provider_manager.stream_completion_async")
    async def test_full_stream_resolution(self, mock_stream):
        async def mock_generator(*args, **kwargs):
            yield "This "
            yield "is "
            yield "rice."
            
        mock_stream.return_value = mock_generator()
        manager = StreamManager()
        context = {
            "user_profile": {"location": "Assam"},
            "language": "en",
            "session_id": "test_stream_session",
            "request_id": "req-id-123"
        }
        
        events = []
        async for event in manager.stream_query("How to grow rice?", context):
            events.append(event)
            
        self.assertTrue(len(events) > 0)
        self.assertIn("event: start", events[0])
        # Assert correlation ID exists inside serialization
        self.assertIn("req-id-123", events[0])
        self.assertIn("event: complete", events[-1])

    @patch("backend.provider_manager.provider_manager.stream_completion_async")
    async def test_stream_state_transitions(self, mock_stream):
        async def mock_generator(*args, **kwargs):
            yield "Mocked "
            yield "token."
        mock_stream.return_value = mock_generator()
        
        manager = StreamManager()
        context = {
            "user_profile": {},
            "language": "en",
            "session_id": "session_state",
            "request_id": "req-state-1"
        }

        events = []
        async for event in manager.stream_query("Test state query", context):
            events.append(event)

        # Verify completed state is present in the final complete event
        self.assertIn("event: complete", events[-1])
        self.assertIn('"current_state": "Completed"', events[-1])

    @patch("backend.provider_manager.provider_manager.stream_completion_async")
    async def test_stream_cancellation(self, mock_stream):
        cleanup_called = False
        async def mock_generator(*args, **kwargs):
            nonlocal cleanup_called
            try:
                yield "Part 1"
                await asyncio.sleep(2.0)
                yield "Part 2"
            finally:
                cleanup_called = True

        mock_stream.return_value = mock_generator()
        manager = StreamManager()
        context = {
            "user_profile": {},
            "language": "en",
            "session_id": "session_cancel"
        }

        # Cancel iterating generator prematurely after first token
        try:
            async for event in manager.stream_query("Cancel query", context):
                if "Part 1" in event:
                    break
        except Exception:
            pass

        await asyncio.sleep(0.1)
        self.assertTrue(cleanup_called)

    @patch("backend.provider_manager.provider_manager.stream_completion_async")
    async def test_partial_stream_failure(self, mock_stream):
        async def mock_generator(*args, **kwargs):
            yield "Starting stream"
            raise Exception("LLM connection timed out")

        mock_stream.return_value = mock_generator()
        manager = StreamManager()
        context = {
            "user_profile": {},
            "language": "en",
            "session_id": "session_fail"
        }

        events = []
        async for event in manager.stream_query("Fail query", context):
            events.append(event)

        # Assert terminal event was error event
        self.assertIn("event: error", events[-1])
        self.assertIn("LLM generation failed", events[-1])

    @patch("backend.provider_manager.provider_manager.stream_completion_async")
    async def test_concurrent_streams(self, mock_stream):
        async def mock_generator(*args, **kwargs):
            yield "Token"

        mock_stream.return_value = mock_generator()
        manager = StreamManager()
        context1 = {"user_profile": {}, "language": "en", "session_id": "s1", "request_id": "req-1"}
        context2 = {"user_profile": {}, "language": "en", "session_id": "s2", "request_id": "req-2"}

        async def run_query(ctx, q):
            events = []
            async for ev in manager.stream_query(q, ctx):
                events.append(ev)
            return events

        res = await asyncio.gather(
            run_query(context1, "How to sow wheat"),
            run_query(context2, "How to sow corn")
        )

        self.assertEqual(len(res), 2)
        self.assertTrue(len(res[0]) > 0)
        self.assertTrue(len(res[1]) > 0)

if __name__ == "__main__":
    unittest.main()
