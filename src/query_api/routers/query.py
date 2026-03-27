"""Query router — POST /query endpoint for natural-language CCTV search."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from src.database import get_db
from src.models import Event
from src.query_api.schemas import EventResponse, QueryRequest, QueryResponse

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def query_events(
    body: QueryRequest,
    request: Request,
    session: AsyncSession = Depends(get_db),
):
    """Parse a natural-language query and retrieve matching CCTV events."""

    parser = request.app.state.parser
    retriever = request.app.state.retriever

    # Step 1 — Parse natural language → structured query
    parsed = parser.parse(body.query)
    parsed.limit = body.top_k

    # Step 2 — Hybrid retrieval (SQL + vector)
    results = await retriever.retrieve(parsed, session)

    # Step 3 — Build response
    event_responses = []
    for obj, score in results:
        if isinstance(obj, Event):
            clip_url = f"/clips/{obj.id}" if obj.clip_path else None
            event_responses.append(
                EventResponse(
                    event_id=str(obj.id),
                    camera_id=str(obj.camera_id),
                    camera_name=None,
                    timestamp_utc=obj.timestamp_utc,
                    class_name=obj.class_name,
                    attributes=obj.attributes or {},
                    confidence=obj.confidence,
                    caption=obj.caption,
                    clip_url=clip_url,
                    score=score,
                )
            )
        else:
            # SceneDiff result
            event_responses.append(
                EventResponse(
                    event_id=str(obj.id),
                    camera_id=str(obj.camera_id),
                    camera_name=None,
                    timestamp_utc=obj.timestamp_utc,
                    class_name=obj.object_label,
                    attributes={"event_type": obj.event_type},
                    confidence=1.0,
                    caption=f"{obj.object_label} {obj.event_type}",
                    clip_url=f"/clips/{obj.id}" if obj.clip_path else None,
                    score=score,
                )
            )

    return QueryResponse(
        query_interpreted=parsed,
        results=event_responses,
        total=len(event_responses),
    )
