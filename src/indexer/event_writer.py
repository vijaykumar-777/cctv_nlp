"""Event writer — persists EventRecords and SceneDiffEvents to the database."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from src.annotator.event_builder import EventRecord
from src.models import Event, SceneDiff
from src.scene_diff.diff_detector import SceneDiffEvent


class EventWriter:
    """Writes detection events and scene diffs to the database."""

    async def write_event(
        self,
        session: AsyncSession,
        record: EventRecord,
        qdrant_id: str | None,
        clip_path: str | None,
    ) -> Event:
        """Persist an EventRecord as an Event ORM row."""
        event = Event(
            camera_id=record.camera_id,
            track_id=record.track_id,
            timestamp_utc=record.timestamp_utc,
            frame_number=record.frame_number,
            class_name=record.class_name,
            attributes=record.attributes,
            bbox=record.bbox,
            confidence=record.confidence,
            caption=record.caption,
            clip_path=clip_path,
            qdrant_id=qdrant_id,
        )
        session.add(event)
        await session.flush()
        return event

    async def write_scene_diff(
        self,
        session: AsyncSession,
        diff_event: SceneDiffEvent,
        clip_path: str | None,
    ) -> SceneDiff:
        """Persist a SceneDiffEvent as a SceneDiff ORM row."""
        scene_diff = SceneDiff(
            camera_id=diff_event.camera_id,
            object_label=diff_event.object_label,
            event_type=diff_event.event_type,
            timestamp_utc=diff_event.timestamp_utc,
            frame_number=diff_event.frame_number,
            baseline_frame_path=diff_event.baseline_frame_path,
            diff_frame_path=diff_event.diff_frame_path,
            clip_path=clip_path,
        )
        session.add(scene_diff)
        await session.flush()
        return scene_diff
