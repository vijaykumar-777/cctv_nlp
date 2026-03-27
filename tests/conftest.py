"""Shared test fixtures for the CCTV Intelligence test suite."""

from __future__ import annotations

import numpy as np
import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.detector.yolo_detector import Detection
from src.models import Base
from src.tracker.boxmot_tracker import Track


@pytest.fixture
def sample_frame() -> np.ndarray:
    """480×640×3 black frame (uint8 zeros)."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def red_frame() -> np.ndarray:
    """480×640×3 frame filled with pure red (BGR: 0,0,255)."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:, :, 2] = 255  # BGR — set red channel
    return frame


@pytest.fixture
def sample_detection() -> Detection:
    return Detection(
        bbox=[100.0, 100.0, 200.0, 200.0],
        confidence=0.9,
        class_id=2,
        class_name="car",
    )


@pytest.fixture
def sample_track() -> Track:
    return Track(
        track_id=1,
        bbox=[100.0, 100.0, 200.0, 200.0],
        confidence=0.9,
        class_id=2,
        class_name="car",
    )


@pytest_asyncio.fixture
async def async_session():
    """In-memory SQLite async session with all tables created."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )

    async with session_factory() as session:
        yield session

    await engine.dispose()
