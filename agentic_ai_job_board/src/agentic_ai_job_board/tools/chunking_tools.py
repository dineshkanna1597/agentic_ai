
from langchain.tools import ToolRuntime, tool

from agentic_ai_job_board.agents.data_ingestion.context import (
    DataIngestionContext,
)
from agentic_ai_job_board.agents.data_ingestion.schemas import (
    ChunkingPlan,
    ChunkingStrategy,
)
from agentic_ai_job_board.services.chunking.document_chunking_service import (
    DocumentChunkingService,
)


@tool
def chunk_document(
    strategy: ChunkingStrategy,
    chunk_size: int,
    chunk_overlap: int,
    section_headers: list[str],
    reasoning: str,
    runtime: ToolRuntime[DataIngestionContext],
) -> str:
    """Split the current document using the selected chunking strategy.

    Choose section_based for documents with identifiable sections.
    Choose recursive for unstructured documents or when section detection
    is unreliable.
    """
    plan = ChunkingPlan(
        strategy=strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        section_headers=section_headers,
        reasoning=reasoning,
    )

    service = DocumentChunkingService()

    if runtime.context.document is None:
        raise RuntimeError(
            "Chunking failed: prepared document is missing."
        )

    chunks = service.chunk(
        document=runtime.context.document,
        plan=plan,
    )

    runtime.context.chunks = chunks

    sections = sorted(
        {
            chunk.metadata.get("section", "General")
            for chunk in chunks
        }
    )

    return (
        f"Document chunking completed successfully. "
        f"Document ID: {runtime.context.document_id}. "
        f"Document type: {runtime.context.document_type}. "
        f"Total chunks: {len(chunks)}. "
        f"Sections: {', '.join(sections)}."
    )
    