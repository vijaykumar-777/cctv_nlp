# CCTV Intelligence Platform

An end-to-end surveillance intelligence system that processes video feeds using **Computer Vision** (YOLOv8 + ByteTrack) and **NLP** (OpenCLIP + Anthropic Claude) to enable natural-language forensic search over detected events.

## Architecture

```
Video File → FrameSampler → YOLODetector → BoxMOTTracker → AttributeExtractor
                                                              ↓
                                                         EventBuilder
                                                              ↓
                                              ┌───────────────┼───────────────┐
                                              ▼               ▼               ▼
                                          PostgreSQL      Qdrant           Clips
                                          (Events)      (Vectors)        (FFmpeg)
                                              │               │
                                              └───────┬───────┘
                                                      ▼
                                              HybridRetriever
                                                      ↓
                                              FastAPI /query
```

## Features

- **Object Detection** — YOLOv8m with 0.4 confidence threshold
- **Multi-Object Tracking** — ByteTrack for persistent track IDs across frames
- **Attribute Extraction** — Dominant color (10 colors) and relative size
- **Scene Diff Detection** — IoU-based removal/theft alerts against registered baselines
- **CLIP Embeddings** — ViT-B-32 image+text vectors for semantic similarity search
- **Hybrid Retrieval** — SQL metadata filters + CLIP vector search, merged and ranked
- **NLP Query Parsing** — Claude converts natural language to structured JSON filters
- **Video Clipping** — On-demand FFmpeg clip extraction around event timestamps

## Setup

### 1. Clone and create virtual environment

```bash
git clone <your-repo-url> && cd cctv_intelligence
python3.11 -m venv .venv && source .venv/bin/activate
```

### 2. Install dependencies (5 steps)

```bash
# Step 1 — PyTorch (CPU)
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cpu

# Step 2 — CV stack
pip install numpy==1.26.4 Pillow==10.3.0 opencv-python-headless==4.9.0.80 ultralytics==8.2.100 boxmot==11.0.9 open_clip_torch==2.26.1

# Step 3 — API stack
pip install fastapi==0.111.0 "uvicorn[standard]==0.30.1" sqlalchemy==2.0.30 asyncpg==0.29.0 alembic==1.13.1 "qdrant-client==1.9.1" redis==5.0.4 pydantic==2.7.1 pydantic-settings==2.3.0 aiosqlite==0.20.0

# Step 4 — NLP stack
pip install anthropic==0.28.0 sentence-transformers==3.0.1 transformers==4.41.2 tokenizers==0.19.1

# Step 5 — Utilities
pip install ffmpeg-python==0.2.0 structlog==24.1.0 python-dotenv==1.0.1 httpx==0.27.0 pytest==8.2.2 pytest-asyncio==0.23.7
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env and set your ANTHROPIC_API_KEY
```

### 4. Start infrastructure

```bash
docker-compose up -d
```

This starts:
- **PostgreSQL 16** on port 5432
- **Qdrant v1.9.1** on port 6333
- **Redis 7** on port 6379

### 5. Run database migrations

```bash
alembic upgrade head
```

## Usage

### Ingest a video

```bash
python scripts/ingest_video.py \
  --video ./videos/parking.mp4 \
  --camera-id cam_01 \
  --camera-name "Parking North"
```

### Register a scene baseline (for theft/removal detection)

```bash
python scripts/ingest_video.py \
  --video ./videos/lobby.mp4 \
  --camera-id cam_02 \
  --camera-name "Lobby" \
  --register-baseline
```

### Start the API

```bash
uvicorn src.query_api.main:app --reload --port 8000
```

### Example queries

```bash
# Find a red car after 2pm
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "find the red car after 2pm", "top_k": 5}'

# Scene change / removal query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "when was the bag removed from the table", "top_k": 3}'

# Health check
curl http://localhost:8000/
```

## Project Structure

```
├── src/
│   ├── config.py                    # Pydantic settings singleton
│   ├── models.py                    # SQLAlchemy ORM (Camera, Event, SceneDiff)
│   ├── database.py                  # Async engine, session factory
│   ├── pipeline.py                  # CCTVPipeline — end-to-end orchestrator
│   ├── ingestor/
│   │   ├── video_reader.py          # VideoReader (cv2 iterator)
│   │   └── frame_sampler.py         # FrameSampler (target FPS decimation)
│   ├── detector/
│   │   ├── yolo_detector.py         # YOLODetector + Detection dataclass
│   │   └── attribute_extractor.py   # Color + size extraction
│   ├── tracker/
│   │   └── boxmot_tracker.py        # BoxMOTTracker + Track dataclass
│   ├── scene_diff/
│   │   ├── baseline_manager.py      # BaselineManager (PNG + JSON storage)
│   │   └── diff_detector.py         # DiffDetector (IoU-based removal)
│   ├── annotator/
│   │   └── event_builder.py         # EventBuilder + EventRecord dataclass
│   ├── indexer/
│   │   ├── clip_embedder.py         # CLIPEmbedder (ViT-B-32)
│   │   ├── qdrant_writer.py         # QdrantWriter (upsert + search)
│   │   └── event_writer.py          # EventWriter (async ORM persist)
│   ├── clipper/
│   │   └── ffmpeg_clipper.py        # FFmpegClipper (±5s clip extraction)
│   └── query_api/
│       ├── main.py                  # FastAPI app with lifespan
│       ├── schemas.py               # Pydantic v2 request/response models
│       ├── nlp/
│       │   ├── query_parser.py      # QueryParser (Claude → JSON)
│       │   └── retriever.py         # HybridRetriever (SQL + vector)
│       └── routers/
│           ├── query.py             # POST /query
│           └── clips.py             # GET /clips/{id}, POST /scene-diff/register
├── scripts/
│   └── ingest_video.py              # CLI ingest script
├── alembic/                         # Database migrations
├── tests/                           # Test suite
├── clips/                           # Generated video clips
├── videos/                          # Source video files
├── requirements.txt                 # Pinned dependencies
├── docker-compose.yml               # Postgres + Qdrant + Redis
└── .env.example                     # Environment template
```

## License

MIT
