from langchain.tools import ToolRuntime, tool

from agentic_ai_job_board.agents.data_ingestion.context import (
    DataIngestionContext,
)
from agentic_ai_job_board.services.embedding.document_embedding_service import (
    DocumentEmbeddingService,
)


@tool
async def generate_embeddings(
    runtime: ToolRuntime[DataIngestionContext],
) -> str:
    """Generate vector embeddings for all chunks of the current document.

    Call this tool after chunk_document has completed successfully.
    """

    context = runtime.context

    if not context.chunks:
        raise ValueError(
            "Cannot generate embeddings. Document chunks are not available."
        )

    embedding_service = DocumentEmbeddingService()

    embeddings = await embedding_service.embed_documents(context.chunks)

    if len(embeddings) != len(context.chunks):
        raise ValueError(
            "Embedding count does not match the number of document chunks."
        )

    context.embeddings = embeddings

    return (
        f"Embedding generation completed successfully. "
        f"Document ID: {context.document_id}. "
        f"Total chunks: {len(context.chunks)}. "
        f"Total embeddings: {len(embeddings)}. "
        f"Embedding dimensions: {len(embeddings[0])}."
    )