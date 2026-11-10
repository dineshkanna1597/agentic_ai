
from typing import Any

from qdrant_client import QdrantClient, models

from agentic_ai_job_board.core.config import settings
from agentic_ai_job_board.services.vector_store.qdrant_service import (
    QdrantStorageService,
)


class QdrantRetrievalService:
    """Retrieve relevant document chunks using semantic vector search."""

    def __init__(
        self,
        storage_service: QdrantStorageService | None = None,
    ) -> None:
        self.storage_service = storage_service or QdrantStorageService()
        self.client: QdrantClient = self.storage_service.client
        self.collection_name = settings.QDRANT_COLLECTION_NAME

    def search(
        self,
        query_vector: list[float],
        document_type: str | None = None,
        limit: int = 5,
        score_threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        """Search for semantically similar document chunks."""

        if not query_vector:
            raise ValueError("Query vector cannot be empty.")

        if len(query_vector) != settings.QDRANT_VECTOR_SIZE:
            raise ValueError(
                f"Invalid query vector dimension. "
                f"Expected {settings.QDRANT_VECTOR_SIZE}, "
                f"got {len(query_vector)}."
            )

        if limit < 1:
            raise ValueError("Limit must be greater than zero.")

        if document_type not in (None, "resume", "job_posting"):
            raise ValueError(
                "Document type must be 'resume', "
                "'job_posting', or None."
            )

        query_filter = None

        if document_type:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.type",
                        match=models.MatchValue(value=document_type),
                    )
                ]
            )

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=limit,
            score_threshold=score_threshold,
            with_payload=["page_content", "metadata"],
            with_vectors=False,
        )

        results = []

        for point in response.points:
            payload = point.payload or {}
            metadata = payload.get("metadata") or {}

            results.append(
                {
                    "content": payload.get("page_content", ""),
                    "source": metadata.get(
                        "file_name",
                        metadata.get("source"),
                    ),
                    "document_id": metadata.get("document_id"),
                    "document_type": metadata.get(
                        "document_type",
                        metadata.get("type"),
                    ),
                    "chunk_index": metadata.get("chunk_index"),
                    "metadata": metadata,
                    "score": point.score,
                }
            )

        return results

    def close(self) -> None:
        """Close the underlying Qdrant client."""

        self.client.close()
