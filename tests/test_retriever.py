"""Tests for HybridRetriever — mocks Qdrant and database."""

from __future__ import annotations

import pytest
import pytest_asyncio
from unittest.mock import MagicMock, AsyncMock, patch

from src.query_api.nlp.retriever import HybridRetriever
from src.query_api.schemas import ParsedQuery


@pytest.fixture
def mock_embedder():
    """Mock CLIPEmbedder that returns a zero vector."""
    import numpy as np
    embedder = MagicMock()
    embedder.embed_text.return_value = np.zeros(512, dtype="float32")
    return embedder


@pytest.fixture
def mock_qdrant():
    """Mock QdrantWriter that returns empty search results."""
    qdrant = MagicMock()
    qdrant.search.return_value = []
    return qdrant


@pytest.mark.asyncio
async def test_empty_results_returns_list(mock_embedder, mock_qdrant, async_session):
    """When both SQL and Qdrant return nothing, retrieve() returns []."""
    retriever = HybridRetriever(mock_embedder, mock_qdrant)
    parsed = ParsedQuery(text_query="find a car", limit=5)

    results = await retriever.retrieve(parsed, async_session)

    assert isinstance(results, list)
    assert len(results) == 0


@pytest.mark.asyncio
async def test_sql_filter_class(mock_embedder, mock_qdrant, async_session):
    """Verify that class_filter is used in SQL query (no crash, returns list)."""
    retriever = HybridRetriever(mock_embedder, mock_qdrant)
    parsed = ParsedQuery(
        class_filter=["car"],
        text_query="red car",
        limit=10,
    )

    results = await retriever.retrieve(parsed, async_session)

    assert isinstance(results, list)
    # No data in the in-memory DB so results are empty
    assert len(results) == 0


@pytest.mark.asyncio
async def test_scene_diff_retrieval(mock_embedder, mock_qdrant, async_session):
    """Verify scene_diff event_type triggers SceneDiff query (no crash)."""
    retriever = HybridRetriever(mock_embedder, mock_qdrant)
    parsed = ParsedQuery(
        event_type="scene_diff",
        class_filter=["bag"],
        text_query="missing bag",
        limit=5,
    )

    results = await retriever.retrieve(parsed, async_session)

    assert isinstance(results, list)
    assert len(results) == 0
