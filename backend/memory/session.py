import json
import os
import threading
import tempfile
import asyncio
from loguru import logger
from backend.config import settings

# Database & Services imports
from backend.services.database_service import UnitOfWork
from backend.models.database import ChatSession, UserProfile, Setting, User
from backend.core.database import async_session

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAX_SESSIONS = 100
MAX_MESSAGES_PER_SESSION = 200

def run_async_sync(coro):
    """Bridges async database calls into synchronous execution paths safely."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # Run in thread pool to prevent blocking the async event loop
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(lambda: asyncio.run(coro))
            return future.result()
    else:
        return asyncio.run(coro)

class MemoryManager:
    """
    Manages short-term conversation memory and persistent farm profiles safely across threads.
    Delegates to PostgreSQL + Redis with a local JSON fallback if DB is offline.
    """
    def __init__(self, storage_file="memory.json"):
        self.lock = threading.Lock()
        self.storage_dir = os.path.join(BASE_DIR, "logs")
        os.makedirs(self.storage_dir, exist_ok=True)
        self.storage_file = os.path.join(self.storage_dir, storage_file)
        
        # Legacy fallback in-memory states
        self.sessions = {}
        self.profiles = {}
        self._dirty = False
        
        # Test DB connection health on startup
        self.use_db = False
        try:
            run_async_sync(self._test_db_connection())
            self.use_db = True
            logger.success("[Database] Database connection successful. Memory persistence active.")
            
            # Run JSON migration to PostgreSQL in background
            run_async_sync(self._migrate_legacy_data())
        except Exception as e:
            logger.warning(f"[Database] Connection failed: {e}. Falling back to local JSON memory storage.")
            self.use_db = False
            self._load_fallback()

    async def _test_db_connection(self):
        from sqlalchemy import select
        async with async_session() as session:
            await session.execute(select(1))

    async def _migrate_legacy_data(self):
        """Migrates data from legacy memory.json to the database if migration hasn't happened yet."""
        if os.path.exists(self.storage_file):
            try:
                with open(self.storage_file, "r") as f:
                    data = json.load(f)
                sessions = data.get("sessions", {})
                profiles = data.get("profiles", {})
                
                if not sessions and not profiles:
                    return
                    
                logger.info(f"[Database] Migrating legacy data: {len(profiles)} profiles, {len(sessions)} sessions.")
                
                async with async_session() as session:
                    # 1. Migrate Profiles
                    for user_id, p_data in profiles.items():
                        from sqlalchemy import select
                        
                        user_stmt = select(User).where(User.id == user_id)
                        res = await session.execute(user_stmt)
                        db_user = res.scalars().first()
                        
                        if not db_user:
                            # Create placeholder user for persistence boundary
                            db_user = User(id=user_id, email=f"farmer_{user_id}@farm360.com", hashed_password="migrated_user")
                            session.add(db_user)
                            await session.flush()
                            
                        profile_stmt = select(UserProfile).where(UserProfile.user_id == user_id)
                        p_res = await session.execute(profile_stmt)
                        db_profile = p_res.scalars().first()
                        
                        if not db_profile:
                            db_profile = UserProfile(
                                user_id=user_id,
                                location=p_data.get("location", "Unknown"),
                                gps_coordinates=p_data.get("gps_coordinates", "")
                            )
                            session.add(db_profile)
                            
                    # 2. Migrate Sessions & Messages
                    for session_id, messages in sessions.items():
                        from backend.models.database import ChatSession, ConversationHistory
                        sess_stmt = select(ChatSession).where(ChatSession.id == session_id)
                        s_res = await session.execute(sess_stmt)
                        db_session = s_res.scalars().first()
                        
                        if not db_session:
                            db_session = ChatSession(id=session_id, user_id="farmer_1", title="Migrated Chat")
                            session.add(db_session)
                            await session.flush()
                            
                        # Add messages
                        for msg in messages:
                            role = msg.get("role", "user")
                            content = msg.get("content", "")
                            role_mapped = "user" if role == "user" else "assistant"
                            
                            # Check if message already exists
                            msg_stmt = select(ConversationHistory).where(
                                ConversationHistory.session_id == session_id,
                                ConversationHistory.content == content
                            )
                            m_res = await session.execute(msg_stmt)
                            if not m_res.scalars().first():
                                db_msg = ConversationHistory(
                                    session_id=session_id,
                                    role=role_mapped,
                                    content=content
                                )
                                session.add(db_msg)
                                
                    await session.commit()
                
                # Rename legacy file to avoid migrating on next boots
                migrated_file = self.storage_file + ".migrated"
                os.replace(self.storage_file, migrated_file)
                logger.success(f"[Database] Legacy data successfully migrated. File renamed to {os.path.basename(migrated_file)}.")
            except Exception as e:
                logger.error(f"[Database] Failed to migrate legacy memory: {e}")

    def _load_fallback(self):
        """Loads memory from local JSON file if DB is unavailable."""
        if os.path.exists(self.storage_file):
            try:
                with open(self.storage_file, "r") as f:
                    data = json.load(f)
                self.sessions = data.get("sessions", {})
                self.profiles = data.get("profiles", {})
                logger.info(f"[Fallback] Loaded {len(self.sessions)} sessions from local JSON storage.")
            except Exception as e:
                logger.error(f"[Fallback] Failed to load local memory file: {e}")

    def _save_fallback(self):
        """Saves memory to local JSON file if DB is unavailable."""
        if not self._dirty:
            return
        try:
            tmp = tempfile.NamedTemporaryFile(
                mode="w",
                dir=self.storage_dir,
                prefix="memory_fallback_",
                suffix=".tmp",
                delete=False,
            )
            json.dump({"sessions": self.sessions, "profiles": self.profiles}, tmp, indent=4)
            tmp.close()
            os.replace(tmp.name, self.storage_file)
            self._dirty = False
        except Exception as e:
            logger.error(f"[Fallback] Failed to save fallback memory: {e}")

    # ── Public APIs ──

    def get_chat_history(self, session_id, max_turns=5):
        """Retrieves conversation messages."""
        if not self.use_db:
            with self.lock:
                if session_id not in self.sessions:
                    self.sessions[session_id] = []
                history = self.sessions[session_id][-max_turns * 2:] if max_turns else self.sessions[session_id]
                return list(history)
                
        # Query PostgreSQL
        async def _get():
            async with UnitOfWork() as uow:
                db_session = await uow.session_repo.get_session_by_id(session_id)
                if not db_session:
                    # Create the chat session lazily
                    await uow.session_repo.create_session(user_id="farmer_1", title="Conversation")
                    return []
                history = [{"role": h.role, "content": h.content} for h in db_session.history]
                return history[-max_turns * 2:] if max_turns else history
                
        try:
            return run_async_sync(_get())
        except Exception as e:
            logger.error(f"[MemoryManager] Failed to fetch DB session history: {e}. Switching to fallback.")
            self.use_db = False
            self._load_fallback()
            return self.get_chat_history(session_id, max_turns)

    def add_message(self, session_id, role, content):
        """Adds a message to the history logs."""
        if not self.use_db:
            with self.lock:
                if session_id not in self.sessions:
                    self.sessions[session_id] = []
                self.sessions[session_id].append({"role": role, "content": content})
                self._dirty = True
            self._save_fallback()
            return

        async def _add():
            async with UnitOfWork() as uow:
                db_session = await uow.session_repo.get_session_by_id(session_id)
                if not db_session:
                    await uow.session_repo.create_session(session_id, user_id="farmer_1")
                role_mapped = "user" if role == "user" else "assistant"
                await uow.session_repo.add_history_message(session_id, role_mapped, content)
                
        try:
            run_async_sync(_add())
        except Exception as e:
            logger.error(f"[MemoryManager] Failed to append DB message: {e}. Switching to fallback.")
            self.use_db = False
            self.add_message(session_id, role, content)

    def set_user_profile(self, user_id, profile_dict):
        """Updates user profile properties."""
        if not self.use_db:
            with self.lock:
                if user_id not in self.profiles:
                    self.profiles[user_id] = {}
                self.profiles[user_id].update(profile_dict)
                self._dirty = True
            self._save_fallback()
            return

        async def _set():
            async with UnitOfWork() as uow:
                location = profile_dict.get("location")
                gps = profile_dict.get("gps_coordinates")
                await uow.profile_repo.create_or_update_profile(
                    user_id=user_id,
                    location=location,
                    gps_coordinates=gps
                )
                
        try:
            run_async_sync(_set())
        except Exception as e:
            logger.error(f"[MemoryManager] Failed to update user profile in DB: {e}. Switching to fallback.")
            self.use_db = False
            self.set_user_profile(user_id, profile_dict)

    def get_user_profile(self, user_id):
        """Retrieves user profile metadata."""
        if not self.use_db:
            with self.lock:
                return dict(self.profiles.get(user_id, {}))

        async def _get():
            async with UnitOfWork() as uow:
                profile = await uow.profile_repo.get_profile_by_user_id(user_id)
                if not profile:
                    return {}
                return {
                    "location": profile.location,
                    "gps_coordinates": profile.gps_coordinates
                }
                
        try:
            return run_async_sync(_get())
        except Exception as e:
            logger.error(f"[MemoryManager] Failed to read user profile: {e}. Switching to fallback.")
            self.use_db = False
            self._load_fallback()
            return self.get_user_profile(user_id)

    def clear_session(self, session_id):
        """Soft-deletes a specific conversation history."""
        if not self.use_db:
            with self.lock:
                if session_id in self.sessions:
                    del self.sessions[session_id]
                    self._dirty = True
            self._save_fallback()
            return

        async def _clear():
            async with UnitOfWork() as uow:
                await uow.session_repo.soft_delete_session(session_id)
                
        try:
            run_async_sync(_clear())
        except Exception as e:
            logger.error(f"[MemoryManager] Failed to clear DB session: {e}. Switching to fallback.")
            self.use_db = False
            self.clear_session(session_id)

    def get_summary(self, session_id):
        """Retrieves the conversation summary for a session."""
        if not self.use_db:
            return None
        async def _get():
            async with UnitOfWork() as uow:
                summary = await uow.session_repo.get_summary(session_id)
                return summary.summary_text if summary else None
        try:
            return run_async_sync(_get())
        except Exception:
            return None