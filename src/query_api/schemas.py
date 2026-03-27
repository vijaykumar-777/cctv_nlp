"""Pydantic v2 schemas for the CCTV Intelligence query API."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class ParsedQuery(BaseModel):
    """Structured representation of a parsed natural-language query."""
    class_filter: list[str] | None = None
    attribute_filters: dict = Field(default_factory=dict)
    camera_ids: list[str] | None = None
    time_range: dict | None = None
    event_type: str = "any"
    text_query: str = ""
    limit: int = 10


class QueryRequest(BaseModel):
    """Incoming query payload."""
    query: str
    top_k: int = 10


class EventResponse(BaseModel):
    """A single search result."""
    event_id: str
    camera_id: str
    camera_name: str | None = None
    timestamp_utc: datetime
    class_name: str
    attributes: dict = Field(default_factory=dict)
    confidence: float
    caption: str | None = None
    clip_url: str | None = None
    score: float


class QueryResponse(BaseModel):
    """Complete query response with interpreted query and results."""
    query_interpreted: ParsedQuery
    results: list[EventResponse]
    total: int
