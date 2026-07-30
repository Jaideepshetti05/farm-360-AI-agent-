from typing import Optional, List
from sqlalchemy import select, update, or_
from backend.repositories.base import BaseRepository
from backend.models.database import MemoryRecord
import datetime

class MemoryRepository(BaseRepository):
    async def save_memory(self, record: MemoryRecord) -> MemoryRecord:
        self.session.add(record)
        await self.session.flush()
        return record

    async def get_memory_by_id(self, memory_id: str) -> Optional[MemoryRecord]:
        stmt = select(MemoryRecord).where(MemoryRecord.id == memory_id)
        res = await self.session.execute(stmt)
        return res.scalars().first()

    async def get_active_memories(self, category: Optional[str] = None) -> List[MemoryRecord]:
        stmt = select(MemoryRecord).where(MemoryRecord.status == "active")
        if category:
            stmt = stmt.where(MemoryRecord.category == category)
        res = await self.session.execute(stmt)
        return list(res.scalars().all())

    async def search_memories_keyword(self, query_str: str, category: Optional[str] = None) -> List[MemoryRecord]:
        stmt = select(MemoryRecord).where(MemoryRecord.status == "active")
        if category:
            stmt = stmt.where(MemoryRecord.category == category)
        
        # Simple fuzzy keyword match via ILIKE on content
        stmt = stmt.where(MemoryRecord.content.ilike(f"%{query_str}%"))
        res = await self.session.execute(stmt)
        return list(res.scalars().all())

    async def update_memory_status(self, memory_id: str, status: str) -> bool:
        stmt = (
            update(MemoryRecord)
            .where(MemoryRecord.id == memory_id)
            .values(status=status)
        )
        res = await self.session.execute(stmt)
        return res.rowcount > 0

    async def get_all_records(self) -> List[MemoryRecord]:
        stmt = select(MemoryRecord)
        res = await self.session.execute(stmt)
        return list(res.scalars().all())
