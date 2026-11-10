
import importlib
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from agentic_ai_job_board.agents.data_ingestion.context import (
    DataIngestionContext,
)
from agentic_ai_job_board.agents.document_quality.schemas import (
    QualityStatus,
)
from agentic_ai_job_board.agents.document_validation.schemas import (
    ValidationStatus,
)
from agentic_ai_job_board.ingestion.events import (
    FileEvent,
    FileEventType,
    FileSource,
)
from langchain_core.documents import Document

@pytest.mark.asyncio
async def test_graph_routes_validation_rejection_to_finalizer(monkeypatch):
    graph_module = importlib.import_module(
        "agentic_ai_job_board.agents.data_ingestion.graph"
    )

    for node_name in (
        "extract_document",
        "normalize_document",
        "detect_and_redact_pii",
        "prepare_redacted_document",
        "extract_resume_contacts",
        "run_document_validation_agent",
    ):
        monkeypatch.setattr(
            graph_module,
            node_name,
            AsyncMock(return_value={}),
        )

    monkeypatch.setattr(
        graph_module,
        "validate_document",
        AsyncMock(
            return_value={
                "validation_status": ValidationStatus.REJECTED,
                "processing_status": "deterministic_validation_completed",
            }
        ),
    )

    quality_agent = AsyncMock(return_value={})
    quality_assessment = AsyncMock(return_value={})

    monkeypatch.setattr(
        graph_module,
        "run_document_quality_agent",
        quality_agent,
    )
    monkeypatch.setattr(
        graph_module,
        "assess_document_quality",
        quality_assessment,
    )

    mock_agent = Mock()
    mock_agent.run = AsyncMock(return_value={})

    monkeypatch.setattr(
        graph_module,
        "DataIngestionAgent",
        Mock(return_value=mock_agent),
    )

    ingestion_graph = graph_module.DataIngestionGraph()

    event = FileEvent.create(
        event_type=FileEventType.ADDED,
        source=FileSource.RESUME,
        file_path=Path("test_resume.txt"),
    )
    context = DataIngestionContext(event=event)

    result = await ingestion_graph.run(context)

    assert result["validation_status"] == ValidationStatus.REJECTED
    assert result["processing_status"] == "rejected"

    quality_agent.assert_not_awaited()
    quality_assessment.assert_not_awaited()
    mock_agent.run.assert_not_awaited()


@pytest.mark.asyncio
async def test_graph_routes_validation_needs_review_to_finalizer(monkeypatch):
    graph_module = importlib.import_module(
        "agentic_ai_job_board.agents.data_ingestion.graph"
    )

    for node_name in (
        "extract_document",
        "normalize_document",
        "detect_and_redact_pii",
        "prepare_redacted_document",
        "extract_resume_contacts",
        "run_document_validation_agent",
    ):
        monkeypatch.setattr(
            graph_module,
            node_name,
            AsyncMock(return_value={}),
        )

    monkeypatch.setattr(
        graph_module,
        "validate_document",
        AsyncMock(
            return_value={
                "validation_status": ValidationStatus.NEEDS_REVIEW,
                "processing_status": "deterministic_validation_completed",
            }
        ),
    )

    quality_agent = AsyncMock(return_value={})
    quality_assessment = AsyncMock(return_value={})

    monkeypatch.setattr(
        graph_module,
        "run_document_quality_agent",
        quality_agent,
    )
    monkeypatch.setattr(
        graph_module,
        "assess_document_quality",
        quality_assessment,
    )

    mock_agent = Mock()
    mock_agent.run = AsyncMock(return_value={})

    monkeypatch.setattr(
        graph_module,
        "DataIngestionAgent",
        Mock(return_value=mock_agent),
    )

    ingestion_graph = graph_module.DataIngestionGraph()

    event = FileEvent.create(
        event_type=FileEventType.ADDED,
        source=FileSource.RESUME,
        file_path=Path("test_resume.txt"),
    )
    context = DataIngestionContext(event=event)

    result = await ingestion_graph.run(context)

    assert result["validation_status"] == ValidationStatus.NEEDS_REVIEW
    assert result["processing_status"] == "needs_review"

    quality_agent.assert_not_awaited()
    quality_assessment.assert_not_awaited()
    mock_agent.run.assert_not_awaited()


@pytest.mark.asyncio
async def test_graph_routes_quality_rejection_to_finalizer(monkeypatch):
    graph_module = importlib.import_module(
        "agentic_ai_job_board.agents.data_ingestion.graph"
    )

    for node_name in (
        "extract_document",
        "normalize_document",
        "detect_and_redact_pii",
        "prepare_redacted_document",
        "extract_resume_contacts",
        "run_document_validation_agent",
    ):
        monkeypatch.setattr(
            graph_module,
            node_name,
            AsyncMock(return_value={}),
        )

    monkeypatch.setattr(
        graph_module,
        "validate_document",
        AsyncMock(
            return_value={
                "validation_status": ValidationStatus.ACCEPTED,
                "processing_status": "deterministic_validation_completed",
            }
        ),
    )

    quality_agent = AsyncMock(return_value={})
    quality_assessment = AsyncMock(
        return_value={
            "quality_status": QualityStatus.REJECTED,
            "processing_status": "quality_assessment_completed",
        }
    )

    monkeypatch.setattr(
        graph_module,
        "run_document_quality_agent",
        quality_agent,
    )
    monkeypatch.setattr(
        graph_module,
        "assess_document_quality",
        quality_assessment,
    )

    mock_agent = Mock()
    mock_agent.run = AsyncMock(return_value={})

    monkeypatch.setattr(
        graph_module,
        "DataIngestionAgent",
        Mock(return_value=mock_agent),
    )

    embedding_node = AsyncMock(return_value={})
    storage_node = AsyncMock(return_value={})

    monkeypatch.setattr(
        graph_module,
        "generate_embeddings",
        embedding_node,
    )
    monkeypatch.setattr(
        graph_module,
        "store_vectors",
        storage_node,
    )

    ingestion_graph = graph_module.DataIngestionGraph()

    event = FileEvent.create(
        event_type=FileEventType.ADDED,
        source=FileSource.RESUME,
        file_path=Path("test_resume.txt"),
    )
    context = DataIngestionContext(event=event)

    result = await ingestion_graph.run(context)

    assert result["validation_status"] == ValidationStatus.ACCEPTED
    assert result["quality_status"] == QualityStatus.REJECTED
    assert result["processing_status"] == "rejected"

    quality_agent.assert_awaited_once()
    quality_assessment.assert_awaited_once()

    mock_agent.run.assert_not_awaited()
    embedding_node.assert_not_awaited()
    storage_node.assert_not_awaited()


@pytest.mark.asyncio
async def test_graph_routes_quality_needs_review_to_finalizer(monkeypatch):
    graph_module = importlib.import_module(
        "agentic_ai_job_board.agents.data_ingestion.graph"
    )

    for node_name in (
        "extract_document",
        "normalize_document",
        "detect_and_redact_pii",
        "prepare_redacted_document",
        "extract_resume_contacts",
        "run_document_validation_agent",
    ):
        monkeypatch.setattr(
            graph_module,
            node_name,
            AsyncMock(return_value={}),
        )

    monkeypatch.setattr(
        graph_module,
        "validate_document",
        AsyncMock(
            return_value={
                "validation_status": ValidationStatus.ACCEPTED,
                "processing_status": "deterministic_validation_completed",
            }
        ),
    )

    quality_agent = AsyncMock(return_value={})
    quality_assessment = AsyncMock(
        return_value={
            "quality_status": QualityStatus.NEEDS_REVIEW,
            "processing_status": "quality_assessment_completed",
        }
    )

    monkeypatch.setattr(
        graph_module,
        "run_document_quality_agent",
        quality_agent,
    )
    monkeypatch.setattr(
        graph_module,
        "assess_document_quality",
        quality_assessment,
    )

    mock_agent = Mock()
    mock_agent.run = AsyncMock(return_value={})

    monkeypatch.setattr(
        graph_module,
        "DataIngestionAgent",
        Mock(return_value=mock_agent),
    )

    embedding_node = AsyncMock(return_value={})
    storage_node = AsyncMock(return_value={})

    monkeypatch.setattr(
        graph_module,
        "generate_embeddings",
        embedding_node,
    )
    monkeypatch.setattr(
        graph_module,
        "store_vectors",
        storage_node,
    )

    ingestion_graph = graph_module.DataIngestionGraph()

    event = FileEvent.create(
        event_type=FileEventType.ADDED,
        source=FileSource.RESUME,
        file_path=Path("test_resume.txt"),
    )
    context = DataIngestionContext(event=event)

    result = await ingestion_graph.run(context)

    assert result["validation_status"] == ValidationStatus.ACCEPTED
    assert result["quality_status"] == QualityStatus.NEEDS_REVIEW
    assert result["processing_status"] == "needs_review"

    quality_agent.assert_awaited_once()
    quality_assessment.assert_awaited_once()

    mock_agent.run.assert_not_awaited()
    embedding_node.assert_not_awaited()
    storage_node.assert_not_awaited()


@pytest.mark.asyncio
async def test_graph_routes_accepted_document_to_vector_storage(monkeypatch):
    graph_module = importlib.import_module(
        "agentic_ai_job_board.agents.data_ingestion.graph"
    )
    nodes_module = importlib.import_module(
       "agentic_ai_job_board.agents.data_ingestion.nodes"
    )

    # Mock extraction and preparation nodes.
    for node_name in (
        "extract_document",
        "normalize_document",
        "detect_and_redact_pii",
        "prepare_redacted_document",
        "extract_resume_contacts",
        "run_document_validation_agent",
    ):
        monkeypatch.setattr(
            graph_module,
            node_name,
            AsyncMock(return_value={}),
        )

    # Simulate successful validation.
    monkeypatch.setattr(
        graph_module,
        "validate_document",
        AsyncMock(
            return_value={
                "validation_status": ValidationStatus.ACCEPTED,
                "processing_status": "deterministic_validation_completed",
            }
        ),
    )

    # Simulate successful quality assessment.
    quality_agent = AsyncMock(return_value={})
    quality_assessment = AsyncMock(
        return_value={
            "quality_status": QualityStatus.ACCEPTED,
            "processing_status": "quality_assessment_completed",
        }
    )

    monkeypatch.setattr(
        graph_module,
        "run_document_quality_agent",
        quality_agent,
    )
    monkeypatch.setattr(
        graph_module,
        "assess_document_quality",
        quality_assessment,
    )

    # Mock the ingestion agent and simulate generated chunks.
    mock_agent = Mock()

    async def mock_ingestion_run(state):
        context.chunks = [
            Document(page_content="First document chunk"),
            Document(page_content="Second document chunk"),
        ]
        context.document_id = "test-document-id"
        context.document_type = "resume"
        return {}

    mock_agent.run = AsyncMock(side_effect=mock_ingestion_run)

    monkeypatch.setattr(
        graph_module,
        "DataIngestionAgent",
        Mock(return_value=mock_agent),
    )

    # Mock embedding generation.
    embeddings = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
    ]

    embedding_service = Mock()
    embedding_service.embed_documents = AsyncMock(
        return_value=embeddings,
    )

    monkeypatch.setattr(
        nodes_module,
        "DocumentEmbeddingService",
        Mock(return_value=embedding_service),
    )

    # Mock Qdrant storage.
    storage_service = Mock()
    storage_service.build_points.return_value = ["mock-point-1", "mock-point-2"]

    monkeypatch.setattr(
        nodes_module,
        "QdrantStorageService",
        Mock(return_value=storage_service),
    )

    ingestion_graph = graph_module.DataIngestionGraph()

    event = FileEvent.create(
        event_type=FileEventType.ADDED,
        source=FileSource.RESUME,
        file_path=Path("test_resume.txt"),
    )
    context = DataIngestionContext(event=event)

    # The ingestion agent requires a prepared document.
    context.document = Document(
        page_content="Sample resume content",
    )
    context.document_id = "test-document-id"
    context.document_type = "resume"

    result = await ingestion_graph.run(context)

    assert result["validation_status"] == ValidationStatus.ACCEPTED
    assert result["quality_status"] == QualityStatus.ACCEPTED
    assert result["processing_status"] == "completed"

    assert result["chunks_count"] == 2
    assert result["embeddings_count"] == 2

    quality_agent.assert_awaited_once()
    quality_assessment.assert_awaited_once()
    mock_agent.run.assert_awaited_once()

    embedding_service.embed_documents.assert_awaited_once()
    storage_service.build_points.assert_called_once()
    storage_service.replace_document.assert_called_once()
    storage_service.client.close.assert_called_once()

def test_data_ingestion_graph_compiles():
    mock_agent = Mock()

    with (
        patch(
            "agentic_ai_job_board.gateway.router.get_chat_model",
            return_value=Mock(),
        ),
        patch(
            "langchain.agents.create_agent",
            return_value=mock_agent,
        ),
    ):
        from agentic_ai_job_board.agents.data_ingestion.graph import (
            DataIngestionGraph,
        )

        ingestion_graph = DataIngestionGraph()

    assert ingestion_graph.graph is not None
    assert ingestion_graph.agent is not None


def test_validation_rejection_finalizes_document():
    from agentic_ai_job_board.agents.data_ingestion.nodes import (
        finalize_non_accepted_document,
    )

    state = {
        "validation_status": ValidationStatus.REJECTED,
    }

    result = finalize_non_accepted_document(state)

    assert result == {"processing_status": "rejected"}


def test_validation_needs_review_finalizes_document():
    from agentic_ai_job_board.agents.data_ingestion.nodes import (
        finalize_non_accepted_document,
    )

    state = {
        "validation_status": ValidationStatus.NEEDS_REVIEW,
    }

    result = finalize_non_accepted_document(state)

    assert result == {"processing_status": "needs_review"}


def test_quality_rejection_finalizes_document():
    from agentic_ai_job_board.agents.data_ingestion.nodes import (
        finalize_non_accepted_document,
    )

    state = {
        "quality_status": QualityStatus.REJECTED,
    }

    result = finalize_non_accepted_document(state)

    assert result == {"processing_status": "rejected"}


def test_quality_needs_review_finalizes_document():
    from agentic_ai_job_board.agents.data_ingestion.nodes import (
        finalize_non_accepted_document,
    )

    state = {
        "quality_status": QualityStatus.NEEDS_REVIEW,
    }

    result = finalize_non_accepted_document(state)

    assert result == {"processing_status": "needs_review"}
