# ROADMAP.md

## Milestone: v1.0 Foundation
This milestone focuses on building the end-to-end CCTV Intelligence Platform as described in your 10-step plan.

| Phase | Description | Status |
|---|---|---|
| Phase 1 | **Project Scaffold** — Directory structure, requirements.txt, .env.example, docker-compose.yml. | [x] ✅ |
| Phase 2 | **Install & Setup** — Dependencies, torches, opencv-python, and API packages. | [x] ✅ |
| Phase 3 | **Database Layer** — SQLAlchemy ORM (models.py), database.py, and initial Alembic migrations. | [x] ✅ |
| Phase 4 | **Vision Modules** — VideoReader, FrameSampler, YOLODetector, and BoxMOTTracker. | [x] ✅ |
| Phase 5 | **Scene Diff & Annotation** — BaselineManager, DiffDetector, and EventBuilder. | [x] ✅ |
| Phase 6 | **Indexing Layer** — CLIPEmbedder, QdrantWriter, and FFmpegClipper. | [x] ✅ |
| Phase 7 | **NLP Query & API** — QueryParser (Anthropic), HybridRetriever, and FastAPI routes. | [x] ✅ |
| Phase 8 | **Pipeline & CLI** — CCTVPipeline, ingest_video.py script, and final README. | [x] ✅ |
| Phase 9 | **Test Suite** — Unit and integration tests for all layers. | [x] ✅ |
| Phase 10 | **Final Integration** — Static checks, import chain, and API smoke tests. | [x] ✅ |

---
*Last updated: 2026-03-28 during initialization*
