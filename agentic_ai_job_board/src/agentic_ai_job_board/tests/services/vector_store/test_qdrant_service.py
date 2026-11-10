
import pytest

from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from agentic_ai_job_board.services.vector_store.qdrant_service import (
    QdrantStorageService,
)


@pytest.fixture
def service():
    """Create an isolated in-memory Qdrant service."""

    storage = QdrantStorageService.__new__(QdrantStorageService)
    storage.client = QdrantClient(":memory:")
    storage.collection_name = "test_collection"

    storage.client.create_collection(
        collection_name=storage.collection_name,
        vectors_config=VectorParams(
            size=4096,
            distance=Distance.COSINE,
        ),
    )

    yield storage

    storage.client.close()


@pytest.fixture
def sample_chunks():
    return [
        Document(
            page_content="Python developer with Django experience.",
            metadata={"source": "resume.pdf", "section": "Skills"},
        ),
        Document(
            page_content="Built REST APIs using FastAPI.",
            metadata={"source": "resume.pdf", "section": "Projects"},
        ),
    ]


@pytest.fixture
def sample_embeddings():
    return [
        [0.1] * 4096,
        [0.2] * 4096,
    ]


def test_generate_chunk_id_is_deterministic():
    first = QdrantStorageService.generate_chunk_id("resume-123", 0)
    second = QdrantStorageService.generate_chunk_id("resume-123", 0)
    third = QdrantStorageService.generate_chunk_id("resume-123", 1)

    assert first == second
    assert first != third


def test_build_points(service, sample_chunks, sample_embeddings):
    points = service.build_points(
        document_id="resume-123",
        document_type="resume",
        chunks=sample_chunks,
        embeddings=sample_embeddings,
    )

    assert len(points) == 2
    assert points[0].payload["page_content"] == sample_chunks[0].page_content
    assert points[0].payload["metadata"]["document_id"] == "resume-123"
    assert points[0].payload["metadata"]["type"] == "resume"
    assert points[0].payload["metadata"]["chunk_index"] == 0
    assert points[0].payload["metadata"]["source"] == "resume.pdf"
    assert points[0].payload["metadata"]["section"] == "Skills"
    assert len(points[0].vector) == 4096


def test_build_points_rejects_empty_chunks(service, sample_embeddings):
    with pytest.raises(ValueError, match="empty chunks"):
        service.build_points(
            document_id="resume-123",
            document_type="resume",
            chunks=[],
            embeddings=sample_embeddings,
        )


def test_build_points_rejects_mismatched_counts(service, sample_chunks):
    with pytest.raises(ValueError, match="count does not match"):
        service.build_points(
            document_id="resume-123",
            document_type="resume",
            chunks=sample_chunks,
            embeddings=[[0.1] * 4096],
        )


def test_build_points_rejects_invalid_embedding_dimension(
    service,
    sample_chunks,
):
    embeddings = [
        [0.1] * 4095,
        [0.2] * 4096,
    ]

    with pytest.raises(ValueError, match="Invalid embedding dimension"):
        service.build_points(
            document_id="resume-123",
            document_type="resume",
            chunks=sample_chunks,
            embeddings=embeddings,
        )


def test_store_points_and_document_exists(
    service,
    sample_chunks,
    sample_embeddings,
):
    points = service.build_points(
        document_id="resume-123",
        document_type="resume",
        chunks=sample_chunks,
        embeddings=sample_embeddings,
    )

    service.store_points(points)

    assert service.document_exists("resume-123")
    assert service.client.count(
        service.collection_name
    ).count == 2


def test_document_exists_returns_false_for_unknown_document(service):
    assert not service.document_exists("unknown-document")


def test_replace_document_removes_outdated_chunks(service):
    initial_chunks = [
        Document(page_content=f"Initial chunk {i}")
        for i in range(3)
    ]
    initial_embeddings = [[0.1] * 4096 for _ in initial_chunks]

    initial_points = service.build_points(
        document_id="resume-123",
        document_type="resume",
        chunks=initial_chunks,
        embeddings=initial_embeddings,
    )

    service.replace_document("resume-123", initial_points)

    assert service.client.count(
        service.collection_name
    ).count == 3

    updated_chunks = [
        Document(page_content=f"Updated chunk {i}")
        for i in range(2)
    ]
    updated_embeddings = [[0.2] * 4096 for _ in updated_chunks]

    updated_points = service.build_points(
        document_id="resume-123",
        document_type="resume",
        chunks=updated_chunks,
        embeddings=updated_embeddings,
    )

    service.replace_document("resume-123", updated_points)

    assert service.client.count(
        service.collection_name
    ).count == 2

    points, _ = service.client.scroll(
        collection_name=service.collection_name,
        limit=10,
        with_payload=True,
        with_vectors=False,
    )

    contents = sorted(
        point.payload["page_content"]
        for point in points
    )

    assert contents == ["Updated chunk 0", "Updated chunk 1"]


def test_delete_document_does_not_affect_other_documents(service):
    for document_id in ("resume-123", "resume-456"):
        chunks = [
            Document(page_content=f"Content for {document_id}")
        ]

        points = service.build_points(
            document_id=document_id,
            document_type="resume",
            chunks=chunks,
            embeddings=[[0.1] * 4096],
        )

        service.store_points(points)

    service.delete_document("resume-123")

    assert not service.document_exists("resume-123")
    assert service.document_exists("resume-456")
    assert service.client.count(
        service.collection_name
    ).count == 1
