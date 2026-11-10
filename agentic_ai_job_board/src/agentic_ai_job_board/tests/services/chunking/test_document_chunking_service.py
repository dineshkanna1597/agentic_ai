
import pytest
from langchain_core.documents import Document

from agentic_ai_job_board.agents.data_ingestion.schemas import (
    ChunkingPlan,
    ChunkingStrategy,
)
from agentic_ai_job_board.services.chunking.document_chunking_service import (
    DocumentChunkingService,
)


@pytest.fixture
def service():
    return DocumentChunkingService()


@pytest.fixture
def document():
    return Document(
        page_content=(
            "John Doe\n"
            "SKILLS\n"
            "Python, Django, PostgreSQL\n"
            "EXPERIENCE\n"
            "Worked as a Python developer building REST APIs.\n"
            "EDUCATION\n"
            "Bachelor of Computer Science."
        ),
        metadata={
            "source": "resume",
            "file_name": "john_doe.pdf",
            "document_type": "resume",
        },
    )


@pytest.fixture
def section_plan():
    return ChunkingPlan(
        strategy=ChunkingStrategy.SECTION_BASED,
        chunk_size=200,
        chunk_overlap=20,
        section_headers=["SKILLS", "EXPERIENCE", "EDUCATION"],
        reasoning="The document contains clearly identifiable sections.",
    )


@pytest.fixture
def recursive_plan():
    return ChunkingPlan(
        strategy=ChunkingStrategy.RECURSIVE,
        chunk_size=100,
        chunk_overlap=20,
        section_headers=[],
        reasoning="The document requires recursive splitting.",
    )


def test_chunk_document_by_sections(service, document, section_plan):
    chunks = service.chunk(document, section_plan)

    assert len(chunks) == 4

    sections = [chunk.metadata["section"] for chunk in chunks]

    assert sections == [
        "General",
        "SKILLS",
        "EXPERIENCE",
        "EDUCATION",
    ]

    assert chunks[0].page_content == "John Doe"


def test_section_chunks_preserve_metadata(service, document, section_plan):
    chunks = service.chunk(document, section_plan)

    for chunk in chunks:
        assert chunk.metadata["source"] == "resume"
        assert chunk.metadata["file_name"] == "john_doe.pdf"
        assert chunk.metadata["document_type"] == "resume"
        assert "section" in chunk.metadata
        assert "chunk_index" in chunk.metadata


def test_section_chunking_falls_back_without_headers(
    service,
    document,
):
    plan = ChunkingPlan(
        strategy=ChunkingStrategy.SECTION_BASED,
        chunk_size=100,
        chunk_overlap=20,
        section_headers=[],
        reasoning="No section headings were identified.",
    )

    chunks = service.chunk(document, plan)

    assert chunks
    assert all("chunk_index" in chunk.metadata for chunk in chunks)


def test_recursive_chunking(service, document, recursive_plan):
    chunks = service.chunk(document, recursive_plan)

    assert chunks
    assert all(chunk.page_content.strip() for chunk in chunks)
    assert all("chunk_index" in chunk.metadata for chunk in chunks)


def test_chunk_indexes_are_sequential(
    service,
    document,
    section_plan,
):
    chunks = service.chunk(document, section_plan)

    indexes = [chunk.metadata["chunk_index"] for chunk in chunks]

    assert indexes == list(range(len(chunks)))


def test_empty_document_raises_error(service, section_plan):
    document = Document(page_content="   ", metadata={})

    with pytest.raises(ValueError, match="Document content cannot be empty"):
        service.chunk(document, section_plan)


def test_section_heading_is_preserved_in_every_chunk(service):
    experience = "Built REST APIs using Python and Django. " * 30

    document = Document(
        page_content=f"EXPERIENCE\n{experience}",
        metadata={"source": "resume"},
    )

    plan = ChunkingPlan(
        strategy=ChunkingStrategy.SECTION_BASED,
        chunk_size=100,
        chunk_overlap=20,
        section_headers=["EXPERIENCE"],
        reasoning="The experience section is too long for one chunk.",
    )

    chunks = service.chunk(document, plan)

    experience_chunks = [
        chunk
        for chunk in chunks
        if chunk.metadata["section"] == "EXPERIENCE"
    ]

    assert len(experience_chunks) > 1

    for chunk in experience_chunks:
        assert chunk.page_content.startswith("EXPERIENCE")
