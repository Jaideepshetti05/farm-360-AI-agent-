import sys
import os
import asyncio
import time
import unittest
import datetime
from unittest.mock import AsyncMock, patch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from backend.core.database import engine
from backend.models.database import Base
from backend.memory_v2 import MemoryRecordModel, MemoryStore, MemoryRanker, MemoryRetriever, MemorySummarizer, MemoryScheduler, MemoryManagerV2

class TestMemoryV2System(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # Build clean test database tables
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
            await conn.run_sync(Base.metadata.create_all)
            
        # Standardize an isolated MemoryManagerV2 instance for tests
        self.mgr = MemoryManagerV2(interval_seconds=1)

    async def asyncTearDown(self):
        self.mgr.stop()
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)

    async def test_memory_insertion_and_retrieval(self):
        # 1. Test insertion of different categories
        m1 = await self.mgr.add_interaction("What fertilizer should I use for rice?", "Use NPK 10-26-26.", category="working")
        m2 = await self.mgr.add_fact("Farm location is Assam", importance=8.0, tags=["assam", "location"])
        
        # Verify store contains the added records
        rec1 = await self.mgr.store.get_memory(m1.id)
        rec2 = await self.mgr.store.get_memory(m2.id)
        
        self.assertIsNotNone(rec1)
        self.assertIsNotNone(rec2)
        self.assertEqual(rec1.category, "working")
        self.assertEqual(rec2.category, "long_term_semantic")
        self.assertEqual(rec2.importance, 8.0)
        self.assertIn("assam", rec2.tags)

    async def test_memory_ranking_and_decay(self):
        # Test exponential decay calculation
        now = datetime.datetime.utcnow()
        past_time = now - datetime.timedelta(hours=24)
        
        recency_now = MemoryRanker.calculate_recency(now, current_time=now)
        recency_past = MemoryRanker.calculate_recency(past_time, current_time=now)
        
        # Recency score must decay over time
        self.assertEqual(recency_now, 1.0)
        self.assertTrue(recency_past < 1.0)
        
        # Test final score ranking calculations
        record = MemoryRecordModel(category="episodic", content="Record harvest yield", importance=5.0)
        score_high_sim = MemoryRanker.score_memory(record, semantic_similarity=0.9, current_time=now)
        score_low_sim = MemoryRanker.score_memory(record, semantic_similarity=0.2, current_time=now)
        
        self.assertTrue(score_high_sim > score_low_sim)

    async def test_hybrid_retrieval(self):
        # Seed test data
        await self.mgr.add_fact("Cattle breed is Holstein Friesian", tags=["cattle", "breed"])
        await self.mgr.add_fact("Assam receives heavy rainfall in July", tags=["assam", "weather"])
        await self.mgr.add_fact("Paddy crop requires high nitrogen", tags=["paddy", "fertilizer"])
        
        # Retrieve by keyword matches
        results = await self.mgr.retriever.retrieve(query="rainfall in Assam", category="long_term_semantic")
        self.assertTrue(len(results) >= 1)
        self.assertIn("Assam receives heavy rainfall in July", [r.content for r in results])
        
        # Retrieve with tag filtering
        results_tags = await self.mgr.retriever.retrieve(query="nitrogen", tags=["paddy"])
        self.assertEqual(len(results_tags), 1)
        self.assertEqual(results_tags[0].content, "Paddy crop requires high nitrogen")

    async def test_duplicate_merging(self):
        # Seed exact duplicate contents
        m1 = MemoryRecordModel(category="working", content="Soil type is alluvial", embedding=[0.1]*768, tags=["soil"])
        m2 = MemoryRecordModel(category="working", content="Soil type is alluvial", embedding=[0.1]*768, tags=["soil", "alluvial"])
        
        await self.mgr.store.add_memory(m1)
        await self.mgr.store.add_memory(m2)
        
        # Run duplicate merge
        merged = await self.mgr.summarizer.merge_duplicates(threshold=0.95)
        self.assertEqual(merged, 1)
        
        # Verify only one remains active
        active = await self.mgr.store.get_all_active_memories(category="working")
        self.assertEqual(len(active), 1)
        self.assertIn("alluvial", active[0].tags)
        self.assertIn("soil", active[0].tags)

    async def test_forgetting_policy_lifecycle(self):
        now = datetime.datetime.utcnow()
        expired_ttl = now - datetime.timedelta(seconds=100)
        
        # Record with TTL policy already expired
        r1 = MemoryRecordModel(
            category="working",
            content="Temporary connection debug log",
            timestamp=expired_ttl,
            expiration_policy="TTL:50"
        )
        # Low relevance active memory
        r2 = MemoryRecordModel(
            category="working",
            content="Irrelevant message turn",
            timestamp=now - datetime.timedelta(days=35),
            importance=0.1
        )
        
        await self.mgr.store.add_memory(r1)
        await self.mgr.store.add_memory(r2)
        
        # Run forgetting scheduler step
        await self.mgr.scheduler.apply_forgetting_policy()
        
        # Verify status updates
        stored_r1 = await self.mgr.store.get_memory(r1.id)
        stored_r2 = await self.mgr.store.get_memory(r2.id)
        
        self.assertEqual(stored_r1.status, "deleted")
        self.assertEqual(stored_r2.status, "archived")

    async def test_local_fallback_storage(self):
        # Force MemoryStore DB to be disabled to test offline fallback
        self.mgr.store.use_db = False
        
        m = await self.mgr.add_fact("Local offline crop notes", tags=["offline"])
        self.assertFalse(self.mgr.store.use_db)
        
        # Verify local file write
        self.assertTrue(os.path.exists(self.mgr.store.fallback_file))
        
        # Query offline memories
        offline_memories = await self.mgr.store.get_all_active_memories()
        self.assertTrue(len(offline_memories) >= 1)
        self.assertIn("Local offline crop notes", [om.content for om in offline_memories])

    async def test_concurrency_locks(self):
        # Concurrently add multiple memories to store to verify thread locks
        async def add_worker(idx: int):
            m = MemoryRecordModel(category="working", content=f"Concurrent record {idx}")
            await self.mgr.store.add_memory(m)

        tasks = [add_worker(i) for i in range(15)]
        await asyncio.gather(*tasks)
        
        all_recs = await self.mgr.store.get_all_records()
        self.assertTrue(len(all_recs) >= 15)
