from qdrant_client import QdrantClient
from uuid import NAMESPACE_URL, uuid5
from agentic_ai_job_board.core.config import settings
from qdrant_client.models import PointStruct
from langchain_core.documents import Document
from qdrant_client import models

class QdrantStorageService:
    """Manage document vectors in the local Qdrant collection."""

    def __init__(self) -> None:
        self.client = QdrantClient(
            path=str(settings.QDRANT_STORAGE_PATH)
        )
        self.collection_name = settings.QDRANT_COLLECTION_NAME

    def validate_collection(self) -> None:
        """Validate that the collection uses the configured vector size."""

        collection_info = self.client.get_collection(
            collection_name=self.collection_name
        )

        vector_config = collection_info.config.params.vectors

        if vector_config.size != settings.QDRANT_VECTOR_SIZE:
            raise ValueError(
                f"Invalid vector size for collection "
                f"'{self.collection_name}'. "
                f"Expected {settings.QDRANT_VECTOR_SIZE}, "
                f"got {vector_config.size}."
            )

    @staticmethod
    def generate_chunk_id(
        document_id: str,
        chunk_index: int,
    ) -> str:
        """Generate a deterministic UUID for a document chunk."""

        return str(
            uuid5(
                NAMESPACE_URL,
                f"{document_id}:{chunk_index}",
            )
        )

    def build_points(
        self,
        document_id: str,
        document_type: str,
        chunks: list[Document],
        embeddings: list[list[float]],
    ) -> list[PointStruct]:
        """Build Qdrant points from document chunks and embeddings."""

        if not chunks:
            raise ValueError("Cannot build points from empty chunks.")

        if len(chunks) != len(embeddings):
            raise ValueError(
                "Chunk count does not match embedding count."
            )

        points = []

        for index, (chunk, embedding) in enumerate(
            zip(chunks, embeddings, strict=True)
        ):
            if len(embedding) != settings.QDRANT_VECTOR_SIZE:
                raise ValueError(
                    f"Invalid embedding dimension at chunk {index}. "
                    f"Expected {settings.QDRANT_VECTOR_SIZE}, "
                    f"got {len(embedding)}."
                )

            metadata = {
                **chunk.metadata,
                "document_id": document_id,
                "type": document_type,
                "chunk_index": index,
            }

            points.append(
                PointStruct(
                    id=self.generate_chunk_id(document_id, index),
                    vector=embedding,
                    payload={
                        "page_content": chunk.page_content,
                        "metadata": metadata,
                    },
                )
            )

        return points
    
    def store_points(self, points: list[PointStruct]) -> None:
        """Store document chunk vectors in the Qdrant collection."""

        if not points:
            raise ValueError("Cannot store an empty list of points.")

        self.validate_collection()

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )        

    def delete_document(self, document_id: str) -> None:
        """Delete all vectors belonging to a document."""

        if not document_id.strip():
            raise ValueError("Document ID cannot be empty.")

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="metadata.document_id",
                            match=models.MatchValue(value=document_id),
                        )
                    ]
                )
            ),
            wait=True,
        )

    def document_exists(self, document_id: str) -> bool:
        """Check whether vectors exist for a document."""

        if not document_id.strip():
            raise ValueError("Document ID cannot be empty.")

        result = self.client.count(
            collection_name=self.collection_name,
            count_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.document_id",
                        match=models.MatchValue(value=document_id),
                    )
                ]
            ),
            exact=True,
        )

        return result.count > 0

    def replace_document(
        self,
        document_id: str,
        points: list[PointStruct],
    ) -> None:
        """Replace a document's vectors while preserving existing data on failure."""

        if not document_id.strip():
            raise ValueError("Document ID cannot be empty.")

        if not points:
            raise ValueError("Cannot replace a document with empty points.")

        for point in points:
            if point.payload["metadata"].get("document_id") != document_id:
                raise ValueError(
                    "All points must belong to the specified document."
                )

        # Insert new points first. Existing IDs are updated.
        self.store_points(points)

        # Remove outdated chunks, keeping the newly stored point IDs.
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="metadata.document_id",
                            match=models.MatchValue(value=document_id),
                        )
                    ],
                    must_not=[
                        models.HasIdCondition(
                            has_id=[point.id for point in points]
                        )
                    ],
                )
            ),
            wait=True,
        )