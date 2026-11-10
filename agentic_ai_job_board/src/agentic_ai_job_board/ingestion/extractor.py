from pathlib import Path

from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyMuPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document

from agentic_ai_job_board.core.logging import get_logger

logger = get_logger(__name__)


class DocumentExtractor:
    """Extract text and metadata from supported document formats."""

    def extract(self, file_path: Path) -> list[Document]:
        """Extract a document and return LangChain Document objects."""

        if not file_path.is_file():
            raise FileNotFoundError(f"Document not found: {file_path}")

        extension = file_path.suffix.lower()

        loaders = {
            ".pdf": PyMuPDFLoader,
            ".docx": Docx2txtLoader,
            ".txt": TextLoader,
        }

        loader_class = loaders.get(extension)

        if loader_class is None:
            raise ValueError(
                f"Unsupported document format: {extension}"
            )

        try:
            if extension == ".txt":
                loader = loader_class(str(file_path), encoding="utf-8")
            else:
                loader = loader_class(str(file_path))

            documents = loader.load()

            if not documents:
                raise ValueError("No content extracted from document")

            for document in documents:
                document.metadata.update(
                    {
                        "source": file_path.name,
                        "file_name": file_path.name,
                        "file_type": extension.lstrip("."),
                    }
                )

            if not any(document.page_content.strip() for document in documents):
                raise ValueError("Document contains no extractable text")

            logger.info(
                "Document extraction successful | file=%s | documents=%d",
                file_path.name,
                len(documents),
            )

            return documents

        except Exception:
            logger.exception(
                "Document extraction failed | file=%s",
                file_path.name,
            )
            raise