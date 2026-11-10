
from langchain.tools import ToolRuntime, tool

from agentic_ai_job_board.agents.data_ingestion.context import (
    DataIngestionContext,
)
from agentic_ai_job_board.services.vector_store.qdrant_service import (
    QdrantStorageService,
)


@tool
def store_vectors(
    runtime: ToolRuntime[DataIngestionContext],
) -> str:
    """Store the current document's chunks and embeddings in Qdrant.

    Call this tool after chunking and embedding are completed.
    Existing vectors for the document are replaced safely.
    """

    context = runtime.context

    if not context.chunks:
        raise ValueError("Cannot store vectors: document chunks are missing.")

    if not context.embeddings:
        raise ValueError("Cannot store vectors: embeddings are missing.")

    if len(context.chunks) != len(context.embeddings):
        raise ValueError(
            "Chunk count does not match embedding count."
        )

    service = QdrantStorageService()

    try:
        points = service.build_points(
            document_id=context.document_id,
            document_type=context.document_type,
            chunks=context.chunks,
            embeddings=context.embeddings,
        )

        service.replace_document(
            document_id=context.document_id,
            points=points,
        )

        return (
            f"Vector storage completed successfully. "
            f"Document ID: {context.document_id}. "
            f"Document type: {context.document_type}. "
            f"Stored chunks: {len(points)}. "
            f"Collection: {service.collection_name}."
        )

    finally:
        service.client.close()
