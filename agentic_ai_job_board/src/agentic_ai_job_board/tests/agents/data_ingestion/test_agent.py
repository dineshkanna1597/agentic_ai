from pathlib import Path

import pytest
from langchain_core.documents import Document
from langgraph.runtime import Runtime

from agentic_ai_job_board.agents.data_ingestion.agent import (
    DataIngestionAgent,
)
from agentic_ai_job_board.agents.data_ingestion.context import (
    DataIngestionContext,
)
from agentic_ai_job_board.ingestion.events import (
    FileEvent,
    FileEventType,
    FileSource,
)


def create_context(document: Document) -> DataIngestionContext:
    event = FileEvent.create(
        event_type=FileEventType.ADDED,
        source=FileSource.RESUME,
        file_path=Path("test_resume.txt"),
    )

    return DataIngestionContext(
        event=event,
        document=document,
        document_id="test-document-001",
        document_type="resume",
    )


@pytest.mark.asyncio
async def test_agent_chunks_structured_resume():
    document = Document(
        page_content="""
        John Doe

        SKILLS
        Python, Django, FastAPI, PostgreSQL

        EXPERIENCE
        Python Developer at ABC Technologies.
        Developed REST APIs using Django REST Framework.

        EDUCATION
        Bachelor of Computer Science.
        """,
        metadata={"source": "test_resume.txt"},
    )

    context = create_context(document)
    agent = DataIngestionAgent()

    await agent.run({}, Runtime(context=context))

    assert context.chunks
    assert all(chunk.page_content.strip() for chunk in context.chunks)

    assert all(
        chunk.metadata.get("chunk_index") == index
        for index, chunk in enumerate(context.chunks)
    )


@pytest.mark.asyncio
async def test_agent_chunks_unstructured_document():
    document = Document(
        page_content=(
            "Python developer with experience in Django and FastAPI. "
            "Worked on REST APIs, PostgreSQL databases, and backend systems. "
            "Interested in scalable software architecture and distributed systems."
        ),
        metadata={"source": "unstructured.txt"},
    )

    context = create_context(document)
    agent = DataIngestionAgent()

    await agent.run({}, Runtime(context=context))

    assert context.chunks
    assert all(chunk.page_content.strip() for chunk in context.chunks)