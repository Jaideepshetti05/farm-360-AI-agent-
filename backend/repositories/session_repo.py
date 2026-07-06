from typing import Optional, List
from sqlalchemy import select, update
from sqlalchemy.orm import selectinload
from backend.repositories.base import BaseRepository
from backend.models.database import ChatSession, ConversationHistory, MemorySummary
import datetime

class SessionRepository(BaseRepository):
    async def create_session(self, user_id: str, title: str = "New Conversation") -> ChatSession:
        db_session = ChatSession(user_id=user_id, title=title)
        self.session.add(db_session)
        await self.session.flush()  # flush to populate DB-generated fields/ID
        return db_session

    async def get_session_by_id(self, session_id: str) -> Optional[ChatSession]:
        stmt = (
            select(ChatSession)
            .where(ChatSession.id == session_id)
            .where(ChatSession.deleted_at.is_(None))
            .options(
                selectinload(ChatSession.history),
                selectinload(ChatSession.summary)
            )
        )
        res = await self.session.execute(stmt)
        return res.scalars().first()

    async def get_user_sessions(self, user_id: str) -> List[ChatSession]:
        stmt = (
            select(ChatSession)
            .where(ChatSession.user_id == user_id)
            .where(ChatSession.deleted_at.is_(None))
            .order_by(ChatSession.updated_at.desc())
            .options(selectinload(ChatSession.history))
        )
        res = await self.session.execute(stmt)
        return list(res.scalars().all())

    async def soft_delete_session(self, session_id: str) -> bool:
        stmt = (
            update(ChatSession)
            .where(ChatSession.id == session_id)
            .values(deleted_at=datetime.datetime.utcnow())
        )
        res = await self.session.execute(stmt)
        return res.rowcount > 0

    async def add_history_message(self, session_id: str, role: str, content: str) -> ConversationHistory:
        msg = ConversationHistory(session_id=session_id, role=role, content=content)
        self.session.add(msg)
        
        # Touch parent session's updated_at timestamp
        stmt = (
            update(ChatSession)
            .where(ChatSession.id == session_id)
            .values(updated_at=datetime.datetime.utcnow())
        )
        await self.session.execute(stmt)
        return msg

    async def update_session_title(self, session_id: str, new_title: str) -> bool:
        stmt = (
            update(ChatSession)
            .where(ChatSession.id == session_id)
            .values(title=new_title, updated_at=datetime.datetime.utcnow())
        )
        res = await self.session.execute(stmt)
        return res.rowcount > 0

    # ── Summary CRUD ──
    async def get_summary(self, session_id: str) -> Optional[MemorySummary]:
        stmt = select(MemorySummary).where(MemorySummary.session_id == session_id)
        res = await self.session.execute(stmt)
        return res.scalars().first()

    async def save_or_update_summary(self, session_id: str, summary_text: str, last_processed_msg_id: int) -> MemorySummary:
        stmt = select(MemorySummary).where(MemorySummary.session_id == session_id)
        res = await self.session.execute(stmt)
        summary = res.scalars().first()
        
        if summary:
            summary.summary_text = summary_text
            summary.last_processed_message_id = last_processed_msg_id
            summary.updated_at = datetime.datetime.utcnow()
        else:
            summary = MemorySummary(
                session_id=session_id,
                summary_text=summary_text,
                last_processed_message_id=last_processed_msg_id
            )
            self.session.add(summary)
            
        return summary
