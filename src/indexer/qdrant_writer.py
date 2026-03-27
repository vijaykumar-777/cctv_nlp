"""Qdrant writer — manages vector upsert and similarity search."""

from __future__ import annotations

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from src.config import settings

COLLECTION = "cctv_events"
VECTOR_SIZE = 512


class QdrantWriter:
    """Writes event embeddings to Qdrant and performs similarity search."""

    def __init__(self) -> None:
        self._client = QdrantClient(
            host=settings.qdrant_host, port=settings.qdrant_port
        )

    def ensure_collection(self) -> None:
        """Create the collection if it does not exist."""
        collections = self._client.get_collections().collections
        existing_names = {c.name for c in collections}

        if COLLECTION not in existing_names:
            self._client.create_collection(
                collection_name=COLLECTION,
                vectors_config=VectorParams(
                    size=VECTOR_SIZE, distance=Distance.COSINE
                ),
            )

    def upsert(self, event_id: str, vector: np.ndarray, payload: dict) -> None:
        """Insert or update a single point in the collection."""
        self._client.upsert(
            collection_name=COLLECTION,
            points=[
                PointStruct(
                    id=event_id,
                    vector=vector.tolist(),
                    payload=payload,
                )
            ],
        )

    def search(
        self,
        query_vector: np.ndarray,
        must_filters: list | None = None,
        top_k: int = 10,
    ) -> list[tuple[str, float, dict]]:
        """Search for the nearest vectors.

        Args:
            query_vector: 512-d float32 array.
            must_filters: Optional list of (field, value) tuples for filtering.
            top_k: Number of results to return.

        Returns:
            List of (id, score, payload) tuples sorted by descending score.
        """
        query_filter = None
        if must_filters:
            conditions = [
                FieldCondition(key=field, match=MatchValue(value=value))
                for field, value in must_filters
            ]
            query_filter = Filter(must=conditions)

        results = self._client.search(
            collection_name=COLLECTION,
            query_vector=query_vector.tolist(),
            query_filter=query_filter,
            limit=top_k,
        )

        return [
            (str(hit.id), hit.score, hit.payload or {})
            for hit in results
        ]
