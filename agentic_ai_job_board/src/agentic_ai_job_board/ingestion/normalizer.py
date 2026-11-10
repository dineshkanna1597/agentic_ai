
import re

from langchain_core.documents import Document

from agentic_ai_job_board.core.logging import get_logger

logger = get_logger(__name__)


class DocumentNormalizer:
    """Normalize extracted document text while preserving its structure."""

    def normalize(self, documents: list[Document]) -> list[Document]:
        """Clean document text and preserve its metadata."""

        if not documents:
            raise ValueError("No documents provided for normalization")

        normalized_documents: list[Document] = []

        for document in documents:
            content = self._normalize_text(document.page_content)

            if not content.strip():
                raise ValueError(
                    "Document contains no content after normalization"
                )

            normalized_documents.append(
                Document(
                    page_content=content,
                    metadata=document.metadata.copy(),
                )
            )

        logger.info(
            "Document normalization successful | documents=%d",
            len(normalized_documents),
        )

        return normalized_documents

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Clean common extraction artifacts without flattening the text."""

        # Normalize line endings and non-breaking spaces.
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = text.replace("\u00a0", " ")

        # Remove control characters while preserving tabs and newlines.
        text = "".join(
            char
            for char in text
            if char in "\n\t" or char.isprintable()
        )

        # Remove trailing whitespace from individual lines.
        lines = [line.rstrip() for line in text.split("\n")]

        # Collapse repeated horizontal spaces without removing indentation.
        lines = [
            re.sub(r"[^\S\n]+", " ", line).strip()
            for line in lines
        ]

        # Collapse excessive blank lines while retaining section separation.
        text = "\n".join(lines)
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()
