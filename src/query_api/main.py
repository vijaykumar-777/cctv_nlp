"""FastAPI application — CCTV Intelligence API."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.database import init_db
from src.detector.yolo_detector import YOLODetector
from src.indexer.clip_embedder import CLIPEmbedder
from src.indexer.qdrant_writer import QdrantWriter
from src.query_api.nlp.query_parser import QueryParser
from src.query_api.nlp.retriever import HybridRetriever
from src.query_api.routers import clips, query


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Startup / shutdown lifecycle for the application."""

    # ── Startup ─────────────────────────────────────────────
    await init_db()

    # Load heavy models into app.state (loaded once, shared)
    application.state.detector = YOLODetector()
    application.state.embedder = CLIPEmbedder()

    qdrant = QdrantWriter()
    qdrant.ensure_collection()
    application.state.qdrant = qdrant

    application.state.parser = QueryParser()
    application.state.retriever = HybridRetriever(
        clip_embedder=application.state.embedder,
        qdrant_writer=application.state.qdrant,
    )

    yield

    # ── Shutdown ────────────────────────────────────────────
    # (nothing to clean up — GC handles model memory)


app = FastAPI(
    title="CCTV Intelligence API",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(query.router)
app.include_router(clips.router)

import os
if os.path.exists("public"):
    app.mount("/ui", StaticFiles(directory="public", html=True), name="ui")

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "service": "CCTV Intelligence API"}
