
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from agentic_ai_job_board.tools import vector_store_tools


@pytest.fixture
def context():
    return SimpleNamespace(
        document_id="resume-123",
        document_type="resume",
        chunks=[
            Document(
                page_content="Python developer with Django experience.",
                metadata={"source": "resume.pdf"},
            ),
            Document(
                page_content="Built REST APIs using FastAPI.",
                metadata={"source": "resume.pdf"},
            ),
        ],
        embeddings=[
            [0.1] * 4096,
            [0.2] * 4096,
        ],
    )


def test_store_vectors_successfully(monkeypatch, context):
    mock_service = MagicMock()
    mock_service.collection_name = "career_portal"
    mock_service.build_points.return_value = ["point-1", "point-2"]

    monkeypatch.setattr(
        vector_store_tools,
        "QdrantStorageService",
        lambda: mock_service,
    )

    runtime = SimpleNamespace(context=context)

    result = vector_store_tools.store_vectors.func(runtime=runtime)

    mock_service.build_points.assert_called_once_with(
        document_id="resume-123",
        document_type="resume",
        chunks=context.chunks,
        embeddings=context.embeddings,
    )

    mock_service.replace_document.assert_called_once_with(
        document_id="resume-123",
        points=["point-1", "point-2"],
    )

    mock_service.client.close.assert_called_once()

    assert "Vector storage completed successfully" in result
    assert "Stored chunks: 2" in result


def test_store_vectors_rejects_missing_chunks(context):
    context.chunks = []
    runtime = SimpleNamespace(context=context)

    with pytest.raises(ValueError, match="chunks are missing"):
        vector_store_tools.store_vectors.func(runtime=runtime)


def test_store_vectors_rejects_missing_embeddings(context):
    context.embeddings = []
    runtime = SimpleNamespace(context=context)

    with pytest.raises(ValueError, match="embeddings are missing"):
        vector_store_tools.store_vectors.func(runtime=runtime)


def test_store_vectors_rejects_mismatched_counts(context):
    context.embeddings = [[0.1] * 4096]
    runtime = SimpleNamespace(context=context)

    with pytest.raises(ValueError, match="does not match"):
        vector_store_tools.store_vectors.func(runtime=runtime)
