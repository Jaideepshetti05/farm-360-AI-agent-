from typing import Optional
from sqlalchemy import select
from backend.repositories.base import BaseRepository
from backend.models.database import UserProfile, Setting

class ProfileRepository(BaseRepository):
    async def get_profile_by_user_id(self, user_id: str) -> Optional[UserProfile]:
        stmt = select(UserProfile).where(UserProfile.user_id == user_id)
        res = await self.session.execute(stmt)
        return res.scalars().first()

    async def create_or_update_profile(
        self, user_id: str, location: Optional[str] = None, gps_coordinates: Optional[str] = None
    ) -> UserProfile:
        stmt = select(UserProfile).where(UserProfile.user_id == user_id)
        res = await self.session.execute(stmt)
        profile = res.scalars().first()
        
        if profile:
            if location is not None:
                profile.location = location
            if gps_coordinates is not None:
                profile.gps_coordinates = gps_coordinates
        else:
            profile = UserProfile(
                user_id=user_id,
                location=location or "",
                gps_coordinates=gps_coordinates or ""
            )
            self.session.add(profile)
            
        return profile

    async def get_settings_by_user_id(self, user_id: str) -> Optional[Setting]:
        stmt = select(Setting).where(Setting.user_id == user_id)
        res = await self.session.execute(stmt)
        return res.scalars().first()

    async def create_or_update_settings(
        self, user_id: str, language: Optional[str] = None, notifications_enabled: Optional[bool] = None
    ) -> Setting:
        stmt = select(Setting).where(Setting.user_id == user_id)
        res = await self.session.execute(stmt)
        setting = res.scalars().first()
        
        if setting:
            if language is not None:
                setting.language = language
            if notifications_enabled is not None:
                setting.notifications_enabled = notifications_enabled
        else:
            setting = Setting(
                user_id=user_id,
                language=language or "en",
                notifications_enabled=notifications_enabled if notifications_enabled is not None else True
            )
            self.session.add(setting)
            
        return setting
