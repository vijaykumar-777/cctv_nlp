"""Clips router — serves video clips and manages scene-diff baselines."""

from __future__ import annotations

import os
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.clipper.ffmpeg_clipper import FFmpegClipper
from src.database import get_db
from src.models import Event
from src.scene_diff.baseline_manager import BaselineManager

router = APIRouter()
_clipper = FFmpegClipper()
_baseline_manager = BaselineManager()


@router.get("/clips/{event_id}")
async def get_clip(
    event_id: str,
    request: Request,
    session: AsyncSession = Depends(get_db),
):
    """Serve a video clip for an event.

    If no clip has been extracted yet, extract it on-the-fly using FFmpeg
    and persist the clip_path back to the database.
    """
    try:
        uid = uuid.UUID(event_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid event ID")

    stmt = select(Event).where(Event.id == uid)
    result = await session.execute(stmt)
    event = result.scalar_one_or_none()

    if event is None:
        raise HTTPException(status_code=404, detail="Event not found")

    # If clip already exists, serve it
    if event.clip_path and os.path.exists(event.clip_path):
        return FileResponse(event.clip_path, media_type="video/mp4")

    # Extract clip on the fly
    if not event.caption:
        raise HTTPException(status_code=404, detail="No video source for clip extraction")

    clips_dir = "./clips"
    os.makedirs(clips_dir, exist_ok=True)
    output_path = os.path.join(clips_dir, f"{event.id}.mp4")

    # Use timestamp from the event (approximate seconds from frame_number)
    timestamp_sec = float(event.frame_number) / 30.0  # rough estimate

    try:
        _clipper.extract(
            video_path=event.caption,  # video_path stored contextually
            timestamp_sec=timestamp_sec,
            output_path=output_path,
        )
    except Exception:
        raise HTTPException(status_code=500, detail="Clip extraction failed")

    # Persist clip_path
    event.clip_path = output_path
    session.add(event)
    await session.commit()

    return FileResponse(output_path, media_type="video/mp4")


class RegisterBaselineRequest(BaseModel):
    camera_id: str
    object_label: str


@router.post("/scene-diff/register")
async def register_baseline(
    body: RegisterBaselineRequest,
    video_path: str = "",
):
    """Register a scene baseline for a camera.

    The baseline captures current objects in the scene for future
    diff detection (removal / theft alerts).
    """
    import cv2

    if not video_path:
        raise HTTPException(status_code=400, detail="video_path query param required")

    # Grab the last frame from the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Cannot open video")

    # Seek to near the end (last 10 seconds)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = max(0, total - int(fps * 10))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    last_frame = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        last_frame = frame
    cap.release()

    if last_frame is None:
        raise HTTPException(status_code=400, detail="Could not read frame from video")

    # Detect objects in the frame
    from src.detector.yolo_detector import YOLODetector
    detector = YOLODetector()
    detections = detector.detect(last_frame)

    # Register baseline
    _baseline_manager.register(body.camera_id, last_frame, detections)

    return {"status": "registered", "camera_id": body.camera_id}
