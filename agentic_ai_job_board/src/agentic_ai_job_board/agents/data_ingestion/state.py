
from typing import TypedDict

from agentic_ai_job_board.agents.document_quality.schemas import (
    QualityStatus,
)
from agentic_ai_job_board.agents.document_validation.schemas import (
    DocumentType,
    ValidationStatus,
)


class DataIngestionState(TypedDict, total=False):
    """Safe, checkpointable state for document ingestion."""

    # Safe file metadata
    file_name: str
    file_path: str
    source: str
    event_type: str

    # Document classification
    expected_document_type: DocumentType
    document_type: DocumentType
    document_id: str

    # Processing outcomes
    validation_status: ValidationStatus
    quality_status: QualityStatus

    # Final workflow outcome
    processing_status: str
    error_message: str
    review_reason: str

    # Non-sensitive metrics
    chunks_count: int
    embeddings_count: int
