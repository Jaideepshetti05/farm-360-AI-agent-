import asyncio
import threading
import time
from typing import Optional
from loguru import logger
from datetime import datetime, timedelta

from backend.memory_v2.memory_store import MemoryStore
from backend.memory_v2.memory_summarizer import MemorySummarizer
from backend.memory_v2.memory_ranker import MemoryRanker
from backend.rag.embedder import GeminiEmbedder

class MemoryScheduler:
    """Asynchronous background scheduler that performs deduplication, summary compressions, and forgetting policies."""
    def __init__(self, store: MemoryStore, summarizer: MemorySummarizer, interval_seconds: int = 3600):
        self.store = store
        self.summarizer = summarizer
        self.interval_seconds = interval_seconds
        self.embedder = GeminiEmbedder()
        
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def start(self):
        """Starts the background worker thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, name="MemorySchedulerThread", daemon=True)
        self._thread.start()
        logger.info("[MemoryScheduler] Background worker thread started.")

    def stop(self):
        """Stops the background worker thread."""
        self._running = False
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=2.0)
        logger.info("[MemoryScheduler] Background worker thread stopped.")

    def _run_loop(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._main_cycle())
        except RuntimeError:
            pass  # Loop stopped explicitly on thread exit

    async def _main_cycle(self):
        while self._running:
            try:
                logger.info("[MemoryScheduler] Starting memory maintenance cycle...")
                await self.run_maintenance_now()
                logger.success("[MemoryScheduler] Completed memory maintenance cycle.")
            except Exception as e:
                logger.error(f"[MemoryScheduler] Error during cycle: {e}")
                
            # Sleep in increments so we can exit quickly if stopped
            for _ in range(self.interval_seconds):
                if not self._running:
                    break
                await asyncio.sleep(1.0)

    async def run_maintenance_now(self):
        """Runs deduplication, embedding verification, and the forgetting policy immediately."""
        # 1. Populate missing embeddings
        await self.refresh_missing_embeddings()
        
        # 2. Merge duplicates
        merged = await self.summarizer.merge_duplicates()
        if merged > 0:
            logger.info(f"[MemoryScheduler] Merged {merged} duplicate memories.")
            
        # 3. Apply forgetting policy transitions
        await self.apply_forgetting_policy()

    async def refresh_missing_embeddings(self):
        """Checks for active memories without vector embeddings and populates them."""
        memories = await self.store.get_all_active_memories()
        for m in memories:
            if not m.embedding or all(x == 0.0 for x in m.embedding):
                logger.info(f"[MemoryScheduler] Refreshing embedding for memory: {m.id}")
                emb = await self.embedder.get_embedding(m.content)
                if emb and not all(x == 0.0 for x in emb):
                    m.embedding = emb
                    await self.store.update_record(m)

    async def apply_forgetting_policy(self):
        """
        Applies retention policies:
        - Active memories decay. If score < 0.1 or timestamp older than 30 days, move to Archived.
        - Archived memories older than 60 days move to Compressed.
        - Compressed memories older than 90 days are set to Deleted.
        """
        records = await self.store.get_all_records()
        now = datetime.utcnow()
        
        for r in records:
            if r.status == "deleted":
                continue
                
            # 1. Custom expiration policy parsing
            if r.expiration_policy and r.expiration_policy.startswith("TTL:"):
                try:
                    ttl_seconds = int(r.expiration_policy.split(":")[1])
                    expire_time = r.timestamp + timedelta(seconds=ttl_seconds)
                    if now > expire_time:
                        logger.info(f"[Forgetting] TTL expired for memory {r.id}. Deleting.")
                        await self.store.update_status(r.id, "deleted")
                        continue
                except ValueError:
                    pass

            # 2. Score and time decay policies
            elapsed_days = (now - r.timestamp).days
            
            if r.status == "active":
                # Compute current relevance score (1.0 similarity assumes a default relevance test)
                score = MemoryRanker.score_memory(r, current_time=now)
                
                # Active -> Archived: older than 30 days OR score decays below 0.1
                if elapsed_days >= 30 or score < 0.1:
                    logger.info(f"[Forgetting] Archiving memory {r.id} (Days: {elapsed_days}, Score: {score:.2f})")
                    await self.store.update_status(r.id, "archived")
                    
            elif r.status == "archived":
                # Archived -> Compressed: older than 60 days
                if elapsed_days >= 60:
                    logger.info(f"[Forgetting] Compressing archived memory {r.id} (Days: {elapsed_days})")
                    await self.store.update_status(r.id, "compressed")
                    
            elif r.status == "compressed":
                # Compressed -> Deleted: older than 90 days
                if elapsed_days >= 90:
                    logger.info(f"[Forgetting] Deleting expired compressed memory {r.id} (Days: {elapsed_days})")
                    await self.store.update_status(r.id, "deleted")
