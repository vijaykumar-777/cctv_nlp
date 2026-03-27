# 👁️ Sentinel AI: CCTV Intelligence Platform

![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)
![YOLO](https://img.shields.io/badge/YOLO-Ultralytics-orange.svg)
![Qdrant](https://img.shields.io/badge/Vector_DB-Qdrant-red.svg)

**Sentinel AI** (CCTV Intelligence Platform) is an advanced, end-to-end local surveillance processing system. It automatically ingests video feeds, tracks objects, computes visual embeddings, and allows users to search historical footage using pure natural language (e.g., *"find the red car"* or *"when was the bag removed"*).

The entire stack is **local-first** and **offline-capable**, requiring entirely local databases (SQLite + Local file-based Qdrant) and zero cloud APIs for parsing or embedding.

---

## ✨ Key Features

- **Natural Language Querying:** Type complex semantic searches. A local SpaCy-based NLP pipeline parses object classes, colors, camera IDs, and time conditions.
- **Object Detection & Tracking:** Leverages Ultralytics YOLO models combined with BoxMOT tracking to uniquely associate and follow entities across frames.
- **Deep Visual Embeddings:** Uses OpenAI's `CLIP` model to convert visual crops of detected objects into dense semantic vectors for fuzzy semantic search.
- **Scene Differencing (Theft Detection):** Automatically computes baselines and registers "Scene Diffs" to detect when objects are abruptly removed or left behind.
- **Hybrid Retrieval System:** Combines precision SQL metadata filtering (timestamps, camera IDs, exact classes) with Qdrant Vector Semantic Search scoring.
- **Modern Web Dashboard:** A sleek, dark-mode glassmorphism UI served directly alongside the API for immediate video interrogation.
- **Automatic Clip Extraction:** Generates sub-clips via `FFmpeg` automatically whenever a specific event is played in the UI.

---

## 🏗️ Architecture Stack

* **Backend API:** FastAPI
* **NLP Parser:** SpaCy (`en_core_web_sm`)
* **Computer Vision:** OpenCV + Ultralytics YOLOv8 + BoxMOT (StrongSORT)
* **Embedding Model:** Transformers (`openai/clip-vit-base-patch32`)
* **Relational Database:** SQLite (via SQLAlchemy + aiosqlite) for metadata and track states.
* **Vector Database:** Qdrant (Local disk mode) for storing CLIP embeddings.
* **Frontend:** Vanilla HTML/CSS/JS (served via FastAPI StaticFiles).

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have the following installed on your system:
- **Python 3.11+**
- **FFmpeg** (For automated video clip extraction)
  - *macOS*: `brew install ffmpeg`
  - *Ubuntu*: `sudo apt install ffmpeg`

### 2. Installation
Clone the repository and install dependencies into a virtual environment:

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Download the local SpaCy English model for NLP
python -m spacy download en_core_web_sm
```

### 3. Environment Variables
Create a `.env` file in the root directory (the system falls back to these if not set):

```env
DATABASE_URL=sqlite+aiosqlite:///cctv.db
QDRANT_PATH=./qdrant_storage
# No cloud API keys required!
```

---

## 💻 Usage

### Step 1: Ingesting Video Data
To process a video and build the database and vector index, use the pipeline script. Place your video in a `videos/` folder:

```bash
# Inside the virtual environment
python -c "
import asyncio
from src.pipeline import CCTVPipeline

async def run():
    pipe = CCTVPipeline(
        video_path='videos/test.mp4',
        camera_id='cam_01', 
        camera_name='Main Entrance',
        register_baseline=True # Enables theft/removal detection
    )
    await pipe.run()

asyncio.run(run())
"
```
*Depending on the length of the video and your computer's resources, YOLO processing may take a few minutes.*

### Step 2: Running the Server
Start the FastAPI application:

```bash
uvicorn src.query_api.main:app --reload --port 8000
```

### Step 3: Accessing the Dashboard UI
Open your browser and navigate to:
**[http://localhost:8000/ui/](http://localhost:8000/ui/)**

Try queries such as:
- *"show me the person in red clothing"*
- *"find all cars on cam_01"*
- *"when was the bag removed"*
- *"blue truck after 2pm"*

---

## 📂 Project Structure

```text
├── public/                 # Modern web UI (index.html, JS, CSS)
├── src/                    # Core Python Source Code
│   ├── annotator/          # Builds Event records and captions
│   ├── clipper/            # FFmpeg sub-clip extraction
│   ├── detector/           # YOLO Object detection & Color extractors
│   ├── indexer/            # Qdrant vector DB & SQL DB writers
│   ├── ingestor/           # Video frame sampling 
│   ├── query_api/          # FastAPI Routes, SpaCy NLP, Hybrid Retriever
│   ├── scene_diff/         # Background subtraction & removal detection
│   └── tracker/            # BoxMOT object tracking
├── tests/                  # Pytest automated test suite
├── videos/                 # Target directory for raw video files
├── cctv.db                 # Local SQLite database (Auto-generated)
└── qdrant_storage/         # Local Qdrant vector storage (Auto-generated)
```

---

## 🧪 Testing

The platform includes comprehensive test coverage for routing, parsing, detection, and scene diffing:

```bash
pytest tests/ -v
```

---

## 📜 Notice

This software processes local video feeds offline. Ensure you have the right to process video surveillance footage following local privacy and surveillance laws before deployment.
