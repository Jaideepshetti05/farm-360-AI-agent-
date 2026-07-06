from backend.core.database import async_session
from backend.repositories.session_repo import SessionRepository
from backend.repositories.profile_repo import ProfileRepository
from backend.repositories.prompt_repo import PromptRepository

class UnitOfWork:
    """
    Implements the Unit of Work pattern using an async context manager
    to manage session lifecycles, coordinate repositories, and commit transactions.
    """
    def __init__(self):
        self.session_factory = async_session
        self.session = None
        self.session_repo = None
        self.profile_repo = None
        self.prompt_repo = None

    async def __aenter__(self):
        self.session = self.session_factory()
        self.session_repo = SessionRepository(self.session)
        self.profile_repo = ProfileRepository(self.session)
        self.prompt_repo = PromptRepository(self.session)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        try:
            if exc_type is not None:
                await self.session.rollback()
            else:
                try:
                    await self.session.commit()
                except Exception:
                    await self.session.rollback()
                    raise
        finally:
            await self.session.close()
