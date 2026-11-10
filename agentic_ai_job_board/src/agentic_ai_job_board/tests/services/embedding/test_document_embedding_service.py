from unittest.mock import AsyncMock

import pytest
from langchain_core.documents import Document

from agentic_ai_job_board.services.embedding.document_embedding_service import (
    DocumentEmbeddingService,
)


@pytest.fixture
def embedding_model():
    from unittest.mock import MagicMock

    model = MagicMock()
    model.aembed_documents = AsyncMock(
        return_value=[
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ]
    )
    return model


@pytest.fixture
def embedding_service(embedding_model):
    return DocumentEmbeddingService(embedding_model=embedding_model)


@pytest.mark.asyncio
async def test_embed_documents(embedding_service, embedding_model):
    chunks = [
        Document(page_content="Python and Django"),
        Document(page_content="FastAPI and PostgreSQL"),
    ]

    embeddings = await embedding_service.embed_documents(chunks)

    assert len(embeddings) == 2
    assert embeddings[0] == [0.1, 0.2, 0.3]
    assert embeddings[1] == [0.4, 0.5, 0.6]

    embedding_model.aembed_documents.assert_awaited_once_with(
        ["Python and Django", "FastAPI and PostgreSQL"]
    )


@pytest.mark.asyncio
async def test_embed_documents_rejects_empty_chunks(embedding_service):
    with pytest.raises(ValueError, match="empty chunks"):
        await embedding_service.embed_documents([])


@pytest.mark.asyncio
async def test_embed_documents_rejects_empty_chunk_content(
    embedding_service,
    embedding_model,
):
    chunks = [
        Document(page_content="Python"),
        Document(page_content="   "),
    ]

    with pytest.raises(ValueError, match="empty text"):
        await embedding_service.embed_documents(chunks)

    embedding_model.aembed_documents.assert_not_awaited()