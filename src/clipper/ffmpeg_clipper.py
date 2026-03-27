"""FFmpeg clipper — extracts short video clips around timestamps."""

from __future__ import annotations

import structlog
import ffmpeg

log = structlog.get_logger()


class FFmpegClipper:
    """Extracts video clips around a given timestamp using ffmpeg-python."""

    def extract(
        self,
        video_path: str,
        timestamp_sec: float,
        output_path: str,
        padding: float = 5.0,
    ) -> str:
        """Extract a clip centred on *timestamp_sec* with ±padding.

        Args:
            video_path: Path to the source video file.
            timestamp_sec: Centre timestamp in seconds.
            output_path: Where to write the output clip.
            padding: Seconds of context before and after the timestamp.

        Returns:
            The output_path on success.

        Raises:
            ffmpeg.Error: If ffmpeg fails.
        """
        start = max(0, timestamp_sec - padding)
        duration = padding * 2

        try:
            (
                ffmpeg
                .input(video_path, ss=start, t=duration)
                .output(output_path, vcodec="libx264", acodec="copy")
                .overwrite_output()
                .run(quiet=True)
            )
        except ffmpeg.Error as exc:
            log.error(
                "ffmpeg_clip_failed",
                video=video_path,
                timestamp=timestamp_sec,
                error=str(exc),
            )
            raise

        return output_path
