
import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from agentic_ai_job_board.agents.data_ingestion.schemas import (
    ChunkingPlan,
    ChunkingStrategy,
)


class DocumentChunkingService:
    """Split documents using an agent-selected chunking strategy."""

    def chunk(
        self,
        document: Document,
        plan: ChunkingPlan,
    ) -> list[Document]:
        if not document.page_content.strip():
            raise ValueError("Document content cannot be empty.")

        if plan.strategy == ChunkingStrategy.SECTION_BASED:
            return self._chunk_by_sections(document, plan)

        return self._chunk_recursively(document, plan)

    def _chunk_by_sections(
        self,
        document: Document,
        plan: ChunkingPlan,
    ) -> list[Document]:
        if not plan.section_headers:
            return self._chunk_recursively(document, plan)

        pattern = "|".join(
            re.escape(header)
            for header in sorted(
                plan.section_headers,
                key=len,
                reverse=True,
            )
        )

        matches = list(
            re.finditer(
                rf"(?im)^\s*({pattern})\s*:?\s*$",
                document.page_content,
            )
        )

        if not matches:
            return self._chunk_recursively(document, plan)

        sections: list[tuple[str, str]] = []

        preamble = document.page_content[:matches[0].start()].strip()
        if preamble:
            sections.append(("General", preamble))

        for index, match in enumerate(matches):
            start = match.start()
            end = (
                matches[index + 1].start()
                if index + 1 < len(matches)
                else len(document.page_content)
            )

            section_name = match.group(1).strip()
            section_content = document.page_content[start:end].strip()

            if section_content:
                sections.append((section_name, section_content))

        chunks: list[Document] = []

        for section_name, section_content in sections:
            section_document = Document(
                page_content=section_content,
                metadata={
                    **document.metadata,
                    "section": section_name,
                },
            )

            section_chunks = self._chunk_recursively(
                section_document,
                plan,
            )

            for chunk in section_chunks:
                if section_name != "General":
                    chunk.page_content = (
                        f"{section_name}\n{chunk.page_content}"
                    )

            chunks.extend(section_chunks)

        return self._add_chunk_indexes(chunks)

    def _chunk_recursively(
        self,
        document: Document,
        plan: ChunkingPlan,
    ) -> list[Document]:
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=plan.chunk_size,
            chunk_overlap=plan.chunk_overlap,
        )

        chunks = splitter.split_documents([document])

        return self._add_chunk_indexes(chunks)

    @staticmethod
    def _add_chunk_indexes(
        chunks: list[Document],
    ) -> list[Document]:
        for index, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = index

        return chunks
