#!/usr/bin/env python
"""CLI script — ingest a video file through the CCTV Intelligence pipeline."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

# Ensure project root is on sys.path when running as script
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import CCTVPipeline  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest a video file into the CCTV Intelligence Platform."
    )
    parser.add_argument(
        "--video",
        required=True,
        help="Path to the video file to ingest.",
    )
    parser.add_argument(
        "--camera-id",
        required=True,
        help="Unique identifier for the camera source (e.g. cam_01).",
    )
    parser.add_argument(
        "--camera-name",
        default="",
        help="Human-readable camera name (e.g. 'Parking North').",
    )
    parser.add_argument(
        "--register-baseline",
        action="store_true",
        default=False,
        help="Register a scene baseline after the first ~30 frames for diff detection.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.video):
        print(f"Error: video file not found: {args.video}", file=sys.stderr)
        sys.exit(1)

    print(f"📹 Ingesting: {args.video}")
    print(f"📷 Camera:    {args.camera_id} ({args.camera_name or 'unnamed'})")
    if args.register_baseline:
        print("🔒 Baseline registration enabled")
    print()

    pipeline = CCTVPipeline(
        video_path=args.video,
        camera_id=args.camera_id,
        camera_name=args.camera_name,
        register_baseline=args.register_baseline,
    )

    summary = asyncio.run(pipeline.run())

    print()
    print("✅ Ingestion complete!")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
