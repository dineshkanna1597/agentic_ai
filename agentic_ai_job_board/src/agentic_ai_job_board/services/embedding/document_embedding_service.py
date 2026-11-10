
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from agentic_ai_job_board.gateway.router import get_embedding_model


class DocumentEmbeddingService:
    """Generate embeddings for documents and user queries."""

    def __init__(self, embedding_model: Embeddings | None = None) -> None:
        self.embedding_model = embedding_model or get_embedding_model()

    async def embed_documents(
        self,
        chunks: list[Document],
    ) -> list[list[float]]:
        """Generate embeddings for document chunks."""

        if not chunks:
            raise ValueError("Cannot generate embeddings for empty chunks.")

        texts = [chunk.page_content for chunk in chunks]

        if any(not text.strip() for text in texts):
            raise ValueError("Document chunks cannot contain empty text.")

        return await self.embedding_model.aembed_documents(texts)

    async def embed_query(self, query: str) -> list[float]:
        """Generate an embedding for a user query."""

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        return await self.embedding_model.aembed_query(query)
