import asyncio
import uuid
from langgraph.runtime import Runtime

from agentic_ai_job_board.agents.data_ingestion.context import (
    DataIngestionContext,
)
from agentic_ai_job_board.agents.data_ingestion.state import (
    DataIngestionState,
)
from agentic_ai_job_board.agents.document_validation.schemas import (
    DocumentType,
)
from agentic_ai_job_board.ingestion.events import (
    FileEventType,
    FileSource,
)
from agentic_ai_job_board.ingestion.extractor import DocumentExtractor
from agentic_ai_job_board.services.embedding.document_embedding_service import (
    DocumentEmbeddingService,
)
from agentic_ai_job_board.services.vector_store.qdrant_service import (
    QdrantStorageService,
)
from agentic_ai_job_board.ingestion.normalizer import DocumentNormalizer
from agentic_ai_job_board.services.validation.pii_detection_service import (
    PIIDetectionService,
)
from langchain_core.documents import Document
from agentic_ai_job_board.services.validation.resume_contact_extraction_service import (
    ResumeContactExtractionService,
)
from agentic_ai_job_board.agents.document_validation.agent import (
    DocumentValidationAgent,
)
from agentic_ai_job_board.services.validation.document_validation_service import (
    DocumentValidationService,
)
from agentic_ai_job_board.agents.document_validation.schemas import (
    DocumentType,
    ValidationStatus,
)
from agentic_ai_job_board.agents.document_quality.agent import (
    DocumentQualityAgent,
)
from agentic_ai_job_board.services.validation.document_quality_service import (
    DocumentQualityService,
)
from agentic_ai_job_board.agents.document_quality.schemas import (
    QualityStatus,
)

async def extract_document(
    state: DataIngestionState,
    runtime: Runtime[DataIngestionContext],
) -> dict:
    """Extract text from the incoming document."""

    context = runtime.context
    event = context.event

    if event.event_type == FileEventType.DELETED:
        raise ValueError("Deleted files cannot enter document ingestion.")

    document_type_mapping = {
        FileSource.RESUME: DocumentType.RESUME,
        FileSource.POSTING: DocumentType.JOB_POSTING,
    }

    expected_document_type = document_type_mapping.get(event.source)

    if expected_document_type is None:
        raise ValueError(
            f"Unsupported document source: {event.source.value}"
        )

    extractor = DocumentExtractor()

    documents = await asyncio.to_thread(
        extractor.extract,
        event.file_path,
    )

    context.extracted_documents = documents

    if not documents:
        raise ValueError("Document extraction returned no content.")

    return {
        "file_name": event.file_path.name,
        "source": event.source.value,
        "event_type": event.event_type.value,
        "expected_document_type": expected_document_type,
        "processing_status": "extracting",
    }

async def normalize_document(
    state: DataIngestionState,
    runtime: Runtime[DataIngestionContext],
) -> dict:
    """Normalize extracted document content."""

    context = runtime.context

    if not context.extracted_documents:
        raise RuntimeError(
            "Normalization failed: no extracted documents available."
        )

    normalizer = DocumentNormalizer()

    normalized_documents = await asyncio.to_thread(
        normalizer.normalize,
        context.extracted_documents,
    )

    context.normalized_documents = normalized_documents

    return {
        "processing_status": "normalizing",
    }

async def detect_and_redact_pii(
    state: DataIngestionState,
    runtime: Runtime[DataIngestionContext],
) -> dict:
    """Detect and redact PII locally before external LLM processing."""

    context = runtime.context

    if not context.normalized_documents:
        raise RuntimeError(
            "PII detection failed: no normalized documents available."
        )

    combined_text = "\n\n".join(
        document.page_content
        for document in context.normalized_documents
    )

    pii_service = PIIDetectionService()

    context.pii_result = await asyncio.to_thread(
        pii_service.detect_and_redact,
        combined_text,
    )

    return {
        "processing_status": "redacting_pii",
    }

async def prepare_redacted_document(
    state: DataIngestionState,
    runtime: Runtime[DataIngestionContext],
) -> dict:
    """Prepare the redacted document for downstream LLM agents."""

    context = runtime.context

    if context.pii_result is None:
        raise RuntimeError(
            "Document preparation failed: PII detection result is missing."
        )

    event = context.event
    expected_type = state["expected_document_type"]

    context.document_type = expected_type.value
    context.document_id = str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            (
                f"agentic-ai-job-board:"
                f"{context.document_type.lower()}:"
                f"{event.file_path.resolve()}"
            ),
        )
    )

    context.document = Document(
        page_content=context.pii_result.redacted_text,
        metadata={
            "source": str(event.file_path.resolve()),
            "file_name": event.file_path.name,
            "file_extension": event.file_path.suffix.lower(),
            "document_id": context.document_id,
            "document_type": context.document_type,
        },
    )

    return {
        "document_id": context.document_id,
        "document_type": expected_type,
        "processing_status": "document_prepared",
    }

async def extract_resume_contacts(
    state: DataIngestionState,
    runtime: Runtime[DataIngestionContext],
) -> dict:
    """Extract and validate resume contact details locally."""

    context = runtime.context

    if state["expected_document_type"] != DocumentType.RESUME:
        return {}

    if context.pii_result is None:
        raise RuntimeError(
            "Contact extraction failed: PII detection result is missing."
        )

    contact_service = ResumeContactExtractionService()

    context.contact_result = await asyncio.to_thread(
        contact_service.extract,
        context.pii_result,
    )

    return {
        "processing_status": "extracting_resume_contacts",
    }

async def run_document_validation_agent(
    state: DataIngestionState,
    runtime: Runtime[DataIngestionContext],
) -> dict:
    """Classify the redacted document and extract structured data."""

    context = runtime.context

    if context.document is None:
        raise RuntimeError(
            "Document validation failed: prepared document is missing."
        )

    validation_agent = DocumentValidationAgent()

    context.validation_extraction = await asyncio.to_thread(
        validation_agent.invoke,
        context.document.page_content,
    )

    return {
        "processing_status": "llm_validation_completed",
    }

async def validate_document(
    state: DataIngestionState,
    runtime: Runtime[DataIngestionContext],
) -> dict:
    """Apply deterministic validation rules to the extracted document."""

    context = runtime.context
    extraction = context.validation_extraction

    if extraction is None:
        raise RuntimeError(
            "Document validation failed: LLM extraction is missing."
        )

    expected_type = state["expected_document_type"]
    validation_service = DocumentValidationService()

    if expected_type == DocumentType.RESUME:
        if extraction.resume_data is None:
            raise RuntimeError(
                "Resume validation failed: extracted resume data is missing."
            )

        context.validation_result = await asyncio.to_thread(
            validation_service.validate_resume,
            extraction.classification,
            extraction.resume_data,
            context.contact_result,
        )

    elif expected_type == DocumentType.JOB_POSTING:
        if extraction.job_posting_data is None:
            raise RuntimeError(
                "Job posting validation failed: extracted posting data is missing."
            )

        context.validation_result = await asyncio.to_thread(
            validation_service.validate_job_posting,
            extraction.classification,
            extraction.job_posting_data,
        )

    else:
        raise ValueError(
            f"Unsupported document type: {expected_type.value}"
        )

    return {
        "validation_status": context.validation_result.status,
        "processing_status": "deterministic_validation_completed",
    }

def route_after_validation(
    state: DataIngestionState,
) -> str:
    """Route the workflow based on deterministic validation status."""

    status = state.get("validation_status")

    if status == ValidationStatus.ACCEPTED:
        return "accepted"

    if status == ValidationStatus.REJECTED:
        return "rejected"

    if status == ValidationStatus.NEEDS_REVIEW:
        return "needs_review"

    raise RuntimeError(
        "Validation routing failed: validation status is missing."
    )

async def run_document_quality_agent(
    state: DataIngestionState,
    runtime: Runtime[DataIngestionContext],
) -> dict:
    """Evaluate the quality of the redacted document using an LLM."""

    context = runtime.context

    if context.document is None:
        raise RuntimeError(
            "Quality assessment failed: prepared document is missing."
        )

    quality_agent = DocumentQualityAgent()

    context.quality_extraction = await asyncio.to_thread(
        quality_agent.invoke,
        context.document.page_content,
    )

    return {
        "processing_status": "llm_quality_assessment_completed",
    }

async def assess_document_quality(
    state: DataIngestionState,
    runtime: Runtime[DataIngestionContext],
) -> dict:
    """Calculate the document quality score and final quality status."""

    context = runtime.context
    extraction = context.quality_extraction

    if extraction is None:
        raise RuntimeError(
            "Quality assessment failed: LLM extraction is missing."
        )

    expected_type = state["expected_document_type"]
    quality_service = DocumentQualityService()

    context.quality_result = await asyncio.to_thread(
        quality_service.assess,
        extraction,
        expected_type,
    )

    return {
        "quality_status": context.quality_result.status,
        "processing_status": "deterministic_quality_assessment_completed",
    }

def route_after_quality_assessment(
    state: DataIngestionState,
) -> str:
    """Route the workflow based on document quality status."""

    status = state.get("quality_status")

    if status == QualityStatus.ACCEPTED:
        return "accepted"

    if status == QualityStatus.REJECTED:
        return "rejected"

    if status == QualityStatus.NEEDS_REVIEW:
        return "needs_review"

    raise RuntimeError(
        "Quality routing failed: quality status is missing."
    )

async def generate_embeddings(
    state: DataIngestionState,
    runtime: Runtime[DataIngestionContext],
) -> dict:
    """Generate embeddings for the document chunks."""

    context = runtime.context

    if not context.chunks:
        raise RuntimeError(
            "Embedding generation failed: no document chunks available."
        )

    embedding_service = DocumentEmbeddingService()

    context.embeddings = await embedding_service.embed_documents(
        context.chunks
    )

    return {
        "embeddings_count": len(context.embeddings),
        "processing_status": "embeddings_generated",
    }

def store_vectors(
    state: DataIngestionState,
    runtime: Runtime[DataIngestionContext],
) -> dict:
    """Store document chunks and embeddings in Qdrant."""

    context = runtime.context

    if not context.chunks or not context.embeddings:
        raise RuntimeError(
            "Vector storage failed: chunks or embeddings are missing."
        )

    storage_service = QdrantStorageService()

    try:
        points = storage_service.build_points(
            document_id=context.document_id,
            document_type=context.document_type,
            chunks=context.chunks,
            embeddings=context.embeddings,
        )

        storage_service.replace_document(
            document_id=context.document_id,
            points=points,
        )
    finally:
        storage_service.client.close()

    return {
        "processing_status": "completed",
        "chunks_count": len(context.chunks),
        "embeddings_count": len(context.embeddings),
    }

def finalize_non_accepted_document(state: DataIngestionState) -> dict:
    """Set the terminal status for rejected or review-required documents."""

    if (
        state.get("validation_status") == ValidationStatus.REJECTED
        or state.get("quality_status") == QualityStatus.REJECTED
    ):
        return {"processing_status": "rejected"}

    if (
        state.get("validation_status") == ValidationStatus.NEEDS_REVIEW
        or state.get("quality_status") == QualityStatus.NEEDS_REVIEW
    ):
        return {"processing_status": "needs_review"}

    raise RuntimeError("Cannot finalize a document without a non-accepted status.")