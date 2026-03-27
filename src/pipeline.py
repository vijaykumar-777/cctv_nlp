"""CCTV Intelligence Pipeline — end-to-end video processing."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import structlog
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.annotator.event_builder import EventBuilder
from src.clipper.ffmpeg_clipper import FFmpegClipper
from src.config import settings
from src.detector.attribute_extractor import AttributeExtractor
from src.detector.yolo_detector import YOLODetector
from src.indexer.clip_embedder import CLIPEmbedder
from src.indexer.event_writer import EventWriter
from src.indexer.qdrant_writer import QdrantWriter
from src.ingestor.frame_sampler import FrameSampler
from src.models import Base, Event
from src.scene_diff.baseline_manager import BaselineManager
from src.scene_diff.diff_detector import DiffDetector
from src.tracker.boxmot_tracker import BoxMOTTracker

log = structlog.get_logger()


class CCTVPipeline:
    """Processes a video file end-to-end: detect → track → embed → index.

    Orchestrates all components: ingestor, detector, tracker, scene diff,
    annotator, embedder, vector store, and persistence.
    """

    def __init__(
        self,
        video_path: str,
        camera_id: str,
        camera_name: str = "",
        register_baseline: bool = False,
    ) -> None:
        self._video_path = video_path
        self._camera_id = camera_id
        self._camera_name = camera_name
        self._register_baseline = register_baseline

        # ── Components ──────────────────────────────────────
        self._sampler = FrameSampler(video_path, target_fps=settings.sample_fps)
        self._detector = YOLODetector()
        self._attr_extractor = AttributeExtractor()
        self._tracker = BoxMOTTracker()
        self._baseline_mgr = BaselineManager()
        self._diff_detector = DiffDetector(self._baseline_mgr)
        self._event_builder = EventBuilder()
        self._embedder = CLIPEmbedder()
        self._qdrant = QdrantWriter()
        self._qdrant.ensure_collection()
        self._clipper = FFmpegClipper()
        self._event_writer = EventWriter()

        # ── Database ────────────────────────────────────────
        self._engine = create_async_engine(settings.database_url, echo=False)
        self._session_factory = async_sessionmaker(
            self._engine, class_=AsyncSession, expire_on_commit=False
        )

    async def run(self) -> dict:
        """Process the entire video and return summary statistics.

        Returns:
            dict with keys: events_written, diffs_written, frames_processed
        """
        events_written = 0
        diffs_written = 0
        frames_processed = 0
        commit_batch = 0

        # Ensure tables exist
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        async with self._session_factory() as session:
            for frame_num, ts_sec, frame in self._sampler:
                frames_processed += 1
                timestamp = datetime.fromtimestamp(ts_sec, tz=timezone.utc)

                # ── Detect ──────────────────────────────────
                detections = self._detector.detect(frame)

                # ── Register baseline if requested (after ~30 frames)
                if self._register_baseline and frames_processed == 30:
                    self._baseline_mgr.register(
                        self._camera_id, frame, detections
                    )
                    log.info(
                        "baseline_registered",
                        camera_id=self._camera_id,
                        objects=len(detections),
                    )

                # ── Track ───────────────────────────────────
                tracks = self._tracker.update(detections, frame)

                # ── Scene diff ──────────────────────────────
                diffs = self._diff_detector.check(
                    self._camera_id, frame, detections, timestamp, frame_num
                )

                # ── Process tracks ──────────────────────────
                for track in tracks:
                    attrs = {
                        "color": self._attr_extractor.extract_color(
                            frame, track.bbox
                        ),
                        "size": self._attr_extractor.extract_size(
                            track.bbox, frame.shape
                        ),
                    }

                    record = self._event_builder.build(
                        camera_id=self._camera_id,
                        track=track,
                        attributes=attrs,
                        timestamp_utc=timestamp,
                        frame_number=frame_num,
                        frame=frame,
                        video_path=self._video_path,
                    )

                    # Embed the cropped image
                    vec = self._embedder.embed_image(record.frame_crop)

                    # Persist to PostgreSQL
                    event = await self._event_writer.write_event(
                        session, record, None, None
                    )

                    # Index in Qdrant
                    qdrant_point_id = str(event.id)
                    self._qdrant.upsert(
                        qdrant_point_id,
                        vec,
                        {
                            "camera_id": self._camera_id,
                            "class_name": track.class_name,
                            "timestamp_utc": ts_sec,
                            "track_id": track.track_id,
                            "attributes": attrs,
                        },
                    )

                    # Update qdrant_id on the Event row
                    stmt = (
                        update(Event)
                        .where(Event.id == event.id)
                        .values(qdrant_id=uuid.UUID(qdrant_point_id))
                    )
                    await session.execute(stmt)
                    events_written += 1

                # ── Process diffs ───────────────────────────
                for diff in diffs:
                    await self._event_writer.write_scene_diff(
                        session, diff, None
                    )
                    diffs_written += 1

                # ── Progress logging ────────────────────────
                if frame_num % 100 == 0:
                    log.info(
                        "progress",
                        frame=frame_num,
                        tracks=len(tracks),
                        events=events_written,
                    )

                # ── Batch commit every 50 frames ────────────
                commit_batch += 1
                if commit_batch >= 50:
                    await session.commit()
                    commit_batch = 0

            # Final commit
            await session.commit()

        summary = {
            "events_written": events_written,
            "diffs_written": diffs_written,
            "frames_processed": frames_processed,
        }
        log.info("pipeline_complete", **summary)
        return summary
