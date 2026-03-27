"""SQLAlchemy 2.0 ORM models for CCTV Intelligence Platform."""

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    BigInteger,
    DateTime,
    Float,
    ForeignKey,
    Index,
    JSON,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all ORM models."""
    pass


class Camera(Base):
    __tablename__ = "cameras"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String, nullable=False)
    location: Mapped[str | None] = mapped_column(String, nullable=True)
    rtsp_url: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    # Relationships
    events: Mapped[list["Event"]] = relationship(
        back_populates="camera", cascade="all, delete-orphan"
    )
    scene_diffs: Mapped[list["SceneDiff"]] = relationship(
        back_populates="camera", cascade="all, delete-orphan"
    )


class Event(Base):
    __tablename__ = "events"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    camera_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("cameras.id", ondelete="CASCADE"),
        nullable=False,
    )
    track_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    timestamp_utc: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), index=True
    )
    frame_number: Mapped[int] = mapped_column(BigInteger)
    class_name: Mapped[str] = mapped_column(String, index=True)
    attributes: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    bbox: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    confidence: Mapped[float] = mapped_column(Float)
    caption: Mapped[str | None] = mapped_column(Text, nullable=True)
    clip_path: Mapped[str | None] = mapped_column(String, nullable=True)
    qdrant_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    # Relationships
    camera: Mapped["Camera"] = relationship(back_populates="events")

    __table_args__ = (
        Index("ix_events_camera_timestamp", "camera_id", "timestamp_utc"),
    )


class SceneDiff(Base):
    __tablename__ = "scene_diffs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    camera_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("cameras.id", ondelete="CASCADE"),
        nullable=False,
    )
    object_label: Mapped[str] = mapped_column(String)
    event_type: Mapped[str] = mapped_column(String)  # "removed" | "appeared"
    timestamp_utc: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), index=True
    )
    frame_number: Mapped[int] = mapped_column(BigInteger)
    baseline_frame_path: Mapped[str | None] = mapped_column(String, nullable=True)
    diff_frame_path: Mapped[str | None] = mapped_column(String, nullable=True)
    clip_path: Mapped[str | None] = mapped_column(String, nullable=True)

    # Relationships
    camera: Mapped["Camera"] = relationship(back_populates="scene_diffs")
