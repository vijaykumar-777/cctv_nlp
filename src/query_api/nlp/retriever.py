"""Hybrid retriever — combines SQL filtering with vector similarity search."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.indexer.clip_embedder import CLIPEmbedder
from src.indexer.qdrant_writer import QdrantWriter
from src.models import Event, SceneDiff
from src.query_api.schemas import ParsedQuery


class HybridRetriever:
    """Retrieves events using a combination of SQL filters and CLIP vector search.

    Step A: SQL query with structured filters.
    Step B: Vector search via Qdrant (if text_query is non-empty).
    Step C: Merge and deduplicate by event_id, sort by combined score.
    """

    def __init__(self, clip_embedder: CLIPEmbedder, qdrant_writer: QdrantWriter) -> None:
        self._embedder = clip_embedder
        self._qdrant = qdrant_writer

    async def retrieve(
        self, parsed: ParsedQuery, session: AsyncSession
    ) -> list[tuple[Event | SceneDiff, float]]:
        """Execute hybrid retrieval and return scored results."""

        fetch_limit = parsed.limit * 3

        # ── Step A: SQL filtering ───────────────────────────────
        if parsed.event_type == "scene_diff":
            sql_results = await self._sql_scene_diff(parsed, session, fetch_limit)
        else:
            sql_results = await self._sql_events(parsed, session, fetch_limit)

        # ── Step B: Vector search ───────────────────────────────
        qdrant_scores: dict[str, float] = {}
        if parsed.text_query:
            vec = self._embedder.embed_text(parsed.text_query)
            hits = self._qdrant.search(vec, top_k=fetch_limit)
            for hit_id, score, _ in hits:
                qdrant_scores[hit_id] = score

        # ── Step C: Merge and deduplicate ───────────────────────
        scored: dict[str, tuple[Event | SceneDiff, float]] = {}

        for obj in sql_results:
            eid = str(obj.id)
            q_score = qdrant_scores.get(eid, 0.0)
            combined = 0.5 + 0.5 * q_score  # sql_hit = 0.5
            scored[eid] = (obj, combined)

        # Add any Qdrant-only hits that weren't in SQL results
        for hit_id, q_score in qdrant_scores.items():
            if hit_id not in scored:
                # Try to load from DB
                try:
                    uid = uuid.UUID(hit_id)
                    stmt = select(Event).where(Event.id == uid)
                    result = await session.execute(stmt)
                    event = result.scalar_one_or_none()
                    if event:
                        scored[hit_id] = (event, 0.5 * q_score)
                except (ValueError, Exception):
                    pass

        # Sort by score descending, take top-limit
        ranked = sorted(scored.values(), key=lambda x: x[1], reverse=True)
        return ranked[: parsed.limit]

    # ── SQL helpers ─────────────────────────────────────────────

    async def _sql_events(
        self, parsed: ParsedQuery, session: AsyncSession, limit: int
    ) -> list[Event]:
        stmt = select(Event)

        if parsed.class_filter:
            stmt = stmt.where(Event.class_name.in_(parsed.class_filter))

        if parsed.camera_ids:
            camera_uuids = []
            for cid in parsed.camera_ids:
                try:
                    camera_uuids.append(uuid.UUID(cid))
                except ValueError:
                    pass
            if camera_uuids:
                stmt = stmt.where(Event.camera_id.in_(camera_uuids))

        if parsed.time_range:
            after = parsed.time_range.get("after")
            before = parsed.time_range.get("before")
            if after:
                stmt = stmt.where(Event.timestamp_utc >= after)
            if before:
                stmt = stmt.where(Event.timestamp_utc <= before)

        stmt = stmt.order_by(Event.timestamp_utc.desc()).limit(limit)
        result = await session.execute(stmt)
        return list(result.scalars().all())

    async def _sql_scene_diff(
        self, parsed: ParsedQuery, session: AsyncSession, limit: int
    ) -> list[SceneDiff]:
        stmt = select(SceneDiff)

        if parsed.class_filter:
            stmt = stmt.where(SceneDiff.object_label.in_(parsed.class_filter))

        if parsed.camera_ids:
            camera_uuids = []
            for cid in parsed.camera_ids:
                try:
                    camera_uuids.append(uuid.UUID(cid))
                except ValueError:
                    pass
            if camera_uuids:
                stmt = stmt.where(SceneDiff.camera_id.in_(camera_uuids))

        if parsed.time_range:
            after = parsed.time_range.get("after")
            before = parsed.time_range.get("before")
            if after:
                stmt = stmt.where(SceneDiff.timestamp_utc >= after)
            if before:
                stmt = stmt.where(SceneDiff.timestamp_utc <= before)

        stmt = stmt.order_by(SceneDiff.timestamp_utc.desc()).limit(limit)
        result = await session.execute(stmt)
        return list(result.scalars().all())
