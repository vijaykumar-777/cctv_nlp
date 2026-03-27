# CCTV Intelligence Platform (GSD Project)

## Project Vision
A comprehensive CCTV intelligence platform that processes surveillance feeds using Computer Vision and NLP. The system provides real-time object detection, tracking, scene change detection, and a natural language query interface to search through recorded events.

## Core Value
Transforming raw surveillance metadata into a searchable, intelligent forensic and real-time alerts system.

## Requirements

### Active (Hypotheses)
- [ ] **Infrastructure** (Phase 1/2) — Multi-repo scaffold, dependencies, and environment configuration.
- [ ] **Database Layer** (Phase 3) — SQLAlchemy ORM and Postgres/Qdrant logic for event persistence.
- [ ] **Vision Stack** (Phase 4/5) — YOLO detection, BoxMOT tracking, and scene diff logic.
- [ ] **Indexing Layer** (Phase 6) — CLIP-based image/text embeddings and Qdrant vector search.
- [ ] **NLP Query API** (Phase 7) — FastAPI app with Anthropic LLM for natural language search.
- [ ] **Pipeline and CLI** (Phase 8/9/10) — Integrated processing loop, ingest scripts, and full testing.

### Out of Scope
- [Mobile App] — Focus is on HTTP API and CLI first.
- [Multi-camera live dashboard] — Focus is on forensic search first.

## Key Decisions
| Decision | Rationale | Outcome |
|----------|-----------|---------|
| **YOLOv8** | Industry standard for real-time speed/accuracy tradeoff. | — Pending |
| **Qdrant** | High-performance vector search with JSON payload support. | — Pending |
| **Anthropic Claude** | Superior NLP-to-JSON parsing ability for structured queries. | — Pending |

---
*Last updated: 2026-03-28 after initialization*
