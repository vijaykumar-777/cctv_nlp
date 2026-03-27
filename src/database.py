"""Async database engine, session factory, and helpers."""

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.config import settings
from src.models import Base

# Async engine — single instance for the process
engine = create_async_engine(settings.database_url, echo=False)

# Session factory
async_session_factory = async_sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency — yields an AsyncSession and closes it after use."""
    async with async_session_factory() as session:
        try:
            yield session
        finally:
            await session.close()


async def init_db() -> None:
    """Create all tables defined in Base.metadata.

    Uses a synchronous connection under the hood via run_sync.
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
