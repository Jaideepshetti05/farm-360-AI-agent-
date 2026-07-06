from typing import Optional, List
from sqlalchemy import select, update
from backend.repositories.base import BaseRepository
from backend.models.database import PromptTemplate
import datetime

class PromptRepository(BaseRepository):
    async def get_by_name(self, name: str, version: Optional[str] = None) -> Optional[PromptTemplate]:
        stmt = select(PromptTemplate).where(PromptTemplate.name == name)
        if version:
            stmt = stmt.where(PromptTemplate.version == version)
        else:
            stmt = stmt.order_by(PromptTemplate.created_at.desc())  # get newest version
        res = await self.session.execute(stmt)
        return res.scalars().first()

    async def get_all_templates(self) -> List[PromptTemplate]:
        stmt = select(PromptTemplate).order_by(PromptTemplate.name, PromptTemplate.version.desc())
        res = await self.session.execute(stmt)
        return list(res.scalars().all())

    async def save_or_update(
        self, name: str, version: str, template_text: str, config: Optional[dict] = None
    ) -> PromptTemplate:
        stmt = select(PromptTemplate).where(PromptTemplate.name == name, PromptTemplate.version == version)
        res = await self.session.execute(stmt)
        prompt = res.scalars().first()
        
        if prompt:
            prompt.template_text = template_text
            if config is not None:
                prompt.config = config
        else:
            prompt = PromptTemplate(
                name=name,
                version=version,
                template_text=template_text,
                config=config or {}
            )
            self.session.add(prompt)
            
        return prompt
