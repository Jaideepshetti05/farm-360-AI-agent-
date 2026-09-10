import os
import asyncio
from typing import Optional
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from backend.config import settings
from loguru import logger

_main_loop: Optional[asyncio.AbstractEventLoop] = None

def set_main_loop(loop: Optional[asyncio.AbstractEventLoop]) -> None:
    """Registers the primary application event loop (e.g. Uvicorn loop)."""
    global _main_loop
    _main_loop = loop

def get_main_loop() -> Optional[asyncio.AbstractEventLoop]:
    """Returns the primary application event loop if registered and open."""
    global _main_loop
    if _main_loop is not None and not _main_loop.is_closed():
        return _main_loop
    return None

# Fetch connection string. Fallback to a local SQLite database for easy development/testing if no Postgres is provided.
DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()
if not DATABASE_URL:
    DATABASE_URL = "sqlite+aiosqlite:///./farm360.db"
    logger.warning(f"[Database] DATABASE_URL not set. Falling back to local SQLite: {DATABASE_URL}")
else:
    logger.info(f"[Database] Initializing database engine.")

# Configure connection pooling options
engine_kwargs = {}
if DATABASE_URL.startswith("postgresql"):
    engine_kwargs = {
        "pool_size": 20,
        "max_overflow": 10,
        "pool_recycle": 1800,
        "pool_pre_ping": True,  # Checks connection health before checks
    }

engine = create_async_engine(DATABASE_URL, **engine_kwargs)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

async def get_db_session():
    """Dependency generator that provides a scoped database session per request."""
    async with async_session() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
