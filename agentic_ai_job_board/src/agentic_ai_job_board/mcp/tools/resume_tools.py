
from typing import Any

from agentic_ai_job_board.services.embedding.document_embedding_service import (
    DocumentEmbeddingService,
)
from agentic_ai_job_board.services.retrieval.qdrant_retrieval_service import (
    QdrantRetrievalService,
)


async def search_resumes(
    query: str,
    limit: int = 5,
    score_threshold: float | None = None,
) -> list[dict[str, Any]]:
    """Search candidate resumes using semantic similarity.

    Args:
        query: Natural-language description of the candidate requirements.
        limit: Maximum number of matching resume chunks.
        score_threshold: Optional minimum similarity score.

    Returns:
        Matching resume chunks with metadata and similarity scores.
    """

    embedding_service = DocumentEmbeddingService()
    retrieval_service = QdrantRetrievalService()

    try:
        query_vector = await embedding_service.embed_query(query)

        return retrieval_service.search(
            query_vector=query_vector,
            document_type="resume",
            limit=limit,
            score_threshold=score_threshold,
        )

    finally:
        retrieval_service.close()
