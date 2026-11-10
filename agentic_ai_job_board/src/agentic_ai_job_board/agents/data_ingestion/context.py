
from dataclasses import dataclass, field

from langchain_core.documents import Document

from agentic_ai_job_board.ingestion.events import FileEvent
from agentic_ai_job_board.services.validation.pii_detection_service import (
    PIIDetectionResult,
)
from agentic_ai_job_board.services.validation.resume_contact_extraction_service import (
    ResumeContactExtractionResult,
)
from agentic_ai_job_board.agents.document_validation.schemas import (
    DocumentValidationExtraction,
    DocumentValidationResult,
)
from agentic_ai_job_board.agents.document_quality.schemas import (
    DocumentQualityExtraction,
    DocumentQualityResult,
)

@dataclass
class DataIngestionContext:
    """Runtime context shared across the document ingestion workflow."""

    event: FileEvent

    # Local document processing
    extracted_documents: list[Document] = field(default_factory=list)
    normalized_documents: list[Document] = field(default_factory=list)

    # Local-only sensitive data; never add these to graph state.
    pii_result: PIIDetectionResult | None = None
    contact_result: ResumeContactExtractionResult | None = None

    # Validation intermediates; runtime-only, never checkpointed.
    validation_extraction: DocumentValidationExtraction | None = None
    validation_result: DocumentValidationResult | None = None

    # Quality intermediates; runtime-only, never checkpointed.
    quality_extraction: DocumentQualityExtraction | None = None
    quality_result: DocumentQualityResult | None = None

    # Redacted document used by downstream agents.
    document: Document | None = None

    # Document metadata
    document_id: str = ""
    document_type: str = ""

    # Data ingestion
    chunks: list[Document] = field(default_factory=list)
    embeddings: list[list[float]] = field(default_factory=list)
