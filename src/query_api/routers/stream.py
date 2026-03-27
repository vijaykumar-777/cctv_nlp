import asyncio
import io
import time
import os
import cv2
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

router = APIRouter(tags=["Stream"])
VIDEO_PATH = "videos/test.mp4"

def generate_frames(detector):
    """Reads video, runs YOLO, draws BBoxes, yields MJPEG bytes."""
    if not os.path.exists(VIDEO_PATH):
        # Yield a blank frame or error frame if not found
        yield b""
        return

    cap = cv2.VideoCapture(VIDEO_PATH)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or not fps:
        fps = 30.0
    sleep_time = 1.0 / fps

    while True:
        success, frame = cap.read()
        if not success:
            time.sleep(1.0)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # Resize for faster processing if needed
        frame = cv2.resize(frame, (640, 360))

        # Run inference using the shared detector from app.state
        results = detector.model.track(frame, persist=True, verbose=False)
        annotated_frame = results[0].plot()

        # Add CRT styling or timestamps
        cv2.putText(
            annotated_frame, 
            "LIVE CAM_01", 
            (20, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (0, 255, 255), 
            2
        )

        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        if not ret:
            continue
            
        frame_bytes = buffer.tobytes()

        # Yield multipart response
        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
        )
        
        # Don't overload the CPU, keep it at real-time
        time.sleep(sleep_time)


@router.get("/live_feed")
async def live_feed(request: Request):
    """Streams a continuous live feed with YOLO bounding boxes overlay."""
    detector = request.app.state.detector
    return StreamingResponse(
        generate_frames(detector),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )
