import asyncio
import time
from typing import List, Optional
from loguru import logger

from backend.memory_v2.memory_models import MemoryRecordModel
from backend.memory_v2.memory_store import MemoryStore
from backend.memory_v2.memory_ranker import MemoryRanker
from backend.memory_v2.memory_retriever import MemoryRetriever
from backend.memory_v2.memory_summarizer import MemorySummarizer
from backend.memory_v2.memory_scheduler import MemoryScheduler
from backend.memory_v2.memory_metrics import MemoryMetrics
from backend.rag.embedder import GeminiEmbedder

class MemoryManagerV2:
    """Orchestrates stores, retrievers, consolidation summaries, and forgetting workers."""
    def __init__(self, interval_seconds: int = 3600):
        self.store = MemoryStore()
        self.embedder = GeminiEmbedder()
        self.retriever = MemoryRetriever(self.store, self.embedder)
        self.summarizer = MemorySummarizer(self.store)
        self.scheduler = MemoryScheduler(self.store, self.summarizer, interval_seconds=interval_seconds)

    def start(self):
        """Starts background maintenance tasks."""
        self.scheduler.start()

    def stop(self):
        """Stops background maintenance tasks."""
        self.scheduler.stop()

    async def add_interaction(
        self,
        query: str,
        response: str,
        category: str = "working",
        importance: float = 3.0,
        source: str = "user_chat",
        tags: Optional[List[str]] = None
    ) -> MemoryRecordModel:
        """Stores a new interaction event in memory storage."""
        combined_text = f"User asked: {query} | Assistant responded: {response}"
        
        # Populate embedding async
        embedding = await self.embedder.get_embedding(combined_text)
        
        model = MemoryRecordModel(
            category=category,
            content=combined_text,
            importance=importance,
            embedding=embedding,
            source=source,
            tags=tags or []
        )
        
        await self.store.add_memory(model)
        logger.debug(f"[MemoryManagerV2] Saved interaction to category '{category}' (ID: {model.id})")
        return model

    async def add_fact(
        self,
        fact: str,
        importance: float = 5.0,
        tags: Optional[List[str]] = None
    ) -> MemoryRecordModel:
        """Manually records a long-term semantic preference or fact about the farm."""
        embedding = await self.embedder.get_embedding(fact)
        
        model = MemoryRecordModel(
            category="long_term_semantic",
            content=fact,
            importance=importance,
            embedding=embedding,
            source="system_fact",
            tags=tags or ["preference"]
        )
        await self.store.add_memory(model)
        return model

    async def retrieve_relevant_context(
        self,
        query: str,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
        token_budget: int = 1500
    ) -> str:
        """
        Retrieves matching memories and formats them as a clean Markdown segment
        to be injected into prompt system instructions.
        """
        start_time = time.time()
        
        memories = await self.retriever.retrieve(
            query=query,
            category=category,
            tags=tags,
            token_budget=token_budget
        )
        
        duration_ms = (time.time() - start_time) * 1000.0
        MemoryMetrics.record_lookup_latency(duration_ms, category)
        MemoryMetrics.record_retrieval_hit(len(memories), category)
        
        if not memories:
            return ""
            
        context_parts = []
        context_parts.append("\n=== RELEVANT LONG-TERM MEMORIES ===")
        for idx, m in enumerate(memories):
            context_parts.append(f"{idx+1}. [{m.category.upper()}] {m.content}")
        context_parts.append("=====================================\n")
        
        return "\n".join(context_parts)
