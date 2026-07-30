import json
import os
import time
import asyncio
import threading
from typing import List, Optional
from loguru import logger
from datetime import datetime

from backend.models.database import MemoryRecord
from backend.services.database_service import UnitOfWork
from backend.memory_v2.memory_models import MemoryRecordModel
from backend.core.database import async_session

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class MemoryStore:
    """Handles persistence of memory records with direct DB writing and local JSON fallback."""
    def __init__(self, fallback_file="memory_v2_fallback.json"):
        self.lock = threading.Lock()
        self.storage_dir = os.path.join(BASE_DIR, "logs")
        os.makedirs(self.storage_dir, exist_ok=True)
        self.fallback_file = os.path.join(self.storage_dir, fallback_file)
        
        self.use_db = False
        self.fallback_memories: List[MemoryRecordModel] = []
        
        # Test connection
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Test connection asynchronously
                self.use_db = True
                logger.success("[MemoryStore] PostgreSQL connection active.")
            else:
                asyncio.run(self._test_connection())
                self.use_db = True
                logger.success("[MemoryStore] PostgreSQL connection initialized.")
        except Exception as e:
            logger.warning(f"[MemoryStore] DB offline: {e}. Using JSON fallback {fallback_file}")
            self.use_db = False
            self._load_fallback()

    async def _test_connection(self):
        from sqlalchemy import select
        async with async_session() as session:
            await session.execute(select(1))

    def _load_fallback(self):
        if os.path.exists(self.fallback_file):
            try:
                with open(self.fallback_file, "r") as f:
                    data = json.load(f)
                self.fallback_memories = [MemoryRecordModel(**m) for m in data]
                logger.info(f"[Fallback] Loaded {len(self.fallback_memories)} fallback memory records.")
            except Exception as e:
                logger.error(f"[Fallback] Failed to load local memory_v2 file: {e}")

    def _save_fallback(self):
        with self.lock:
            try:
                data = [m.model_dump() for m in self.fallback_memories]
                # Write to temp file first to prevent corruption
                temp_path = self.fallback_file + ".tmp"
                with open(temp_path, "w") as f:
                    json.dump(data, f, default=str, indent=4)
                os.replace(temp_path, self.fallback_file)
            except Exception as e:
                logger.error(f"[Fallback] Failed to save fallback memory_v2 file: {e}")

    async def add_memory(self, model: MemoryRecordModel) -> MemoryRecordModel:
        if not self.use_db:
            with self.lock:
                self.fallback_memories.append(model)
            self._save_fallback()
            return model

        try:
            async with UnitOfWork() as uow:
                db_record = MemoryRecord(
                    id=model.id,
                    timestamp=model.timestamp,
                    category=model.category,
                    content=model.content,
                    importance=model.importance,
                    recency=model.recency,
                    access_frequency=model.access_frequency,
                    confidence=model.confidence,
                    embedding=model.embedding,
                    source=model.source,
                    tags=model.tags,
                    expiration_policy=model.expiration_policy,
                    status=model.status,
                    feedback_score=model.feedback_score
                )
                await uow.memory_repo.save_memory(db_record)
            return model
        except Exception as e:
            logger.error(f"[MemoryStore] Failed to save memory to DB: {e}. Saving to fallback.")
            self.use_db = False
            self._load_fallback()
            return await self.add_memory(model)

    async def get_memory(self, memory_id: str) -> Optional[MemoryRecordModel]:
        if not self.use_db:
            with self.lock:
                for m in self.fallback_memories:
                    if m.id == memory_id:
                        return m
            return None

        try:
            async with UnitOfWork() as uow:
                db_rec = await uow.memory_repo.get_memory_by_id(memory_id)
                if db_rec:
                    return MemoryRecordModel.model_validate(db_rec)
            return None
        except Exception as e:
            logger.error(f"[MemoryStore] Failed to read DB memory: {e}. Querying fallback.")
            self.use_db = False
            self._load_fallback()
            return await self.get_memory(memory_id)

    async def get_all_active_memories(self, category: Optional[str] = None) -> List[MemoryRecordModel]:
        if not self.use_db:
            with self.lock:
                results = [m for m in self.fallback_memories if m.status == "active"]
                if category:
                    results = [m for m in results if m.category == category]
                return results

        try:
            async with UnitOfWork() as uow:
                db_recs = await uow.memory_repo.get_active_memories(category)
                return [MemoryRecordModel.model_validate(r) for r in db_recs]
        except Exception as e:
            logger.error(f"[MemoryStore] Failed to query active memories: {e}. Falling back.")
            self.use_db = False
            self._load_fallback()
            return await self.get_all_active_memories(category)

    async def get_all_records(self) -> List[MemoryRecordModel]:
        if not self.use_db:
            return list(self.fallback_memories)
        try:
            async with UnitOfWork() as uow:
                db_recs = await uow.memory_repo.get_all_records()
                return [MemoryRecordModel.model_validate(r) for r in db_recs]
        except Exception:
            return list(self.fallback_memories)

    async def update_status(self, memory_id: str, status: str) -> bool:
        if not self.use_db:
            with self.lock:
                for m in self.fallback_memories:
                    if m.id == memory_id:
                        m.status = status
                        self._save_fallback()
                        return True
            return False

        try:
            async with UnitOfWork() as uow:
                return await uow.memory_repo.update_memory_status(memory_id, status)
        except Exception as e:
            logger.error(f"[MemoryStore] Failed to update memory status: {e}.")
            return False

    async def update_record(self, record: MemoryRecordModel) -> bool:
        if not self.use_db:
            with self.lock:
                for i, m in enumerate(self.fallback_memories):
                    if m.id == record.id:
                        self.fallback_memories[i] = record
                        self._save_fallback()
                        return True
            return False

        try:
            async with UnitOfWork() as uow:
                db_rec = await uow.memory_repo.get_memory_by_id(record.id)
                if db_rec:
                    db_rec.importance = record.importance
                    db_rec.recency = record.recency
                    db_rec.access_frequency = record.access_frequency
                    db_rec.confidence = record.confidence
                    db_rec.status = record.status
                    db_rec.feedback_score = record.feedback_score
                    db_rec.tags = record.tags
                    db_rec.content = record.content
                    return True
            return False
        except Exception as e:
            logger.error(f"[MemoryStore] Failed to update memory fields: {e}.")
            return False
