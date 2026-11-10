from langchain_core.messages import HumanMessage, SystemMessage

from agentic_ai_job_board.agents.document_quality.prompts import (
    DOCUMENT_QUALITY_SYSTEM_PROMPT,
)
from agentic_ai_job_board.agents.document_quality.schemas import (
    DocumentQualityExtraction,
)
from agentic_ai_job_board.gateway.router import get_chat_model


class DocumentQualityAgent:
    """Evaluate the quality of resumes and job postings using an LLM."""

    def __init__(self) -> None:
        self.llm = get_chat_model()
        self.structured_llm = self.llm.with_structured_output(
            DocumentQualityExtraction
        )

    def invoke(self, document_text: str) -> DocumentQualityExtraction:
        """Evaluate document quality and return structured results."""

        if not document_text or not document_text.strip():
            raise ValueError("Document text cannot be empty.")

        messages = [
            SystemMessage(content=DOCUMENT_QUALITY_SYSTEM_PROMPT),
            HumanMessage(
                content=f"""
Evaluate the quality of the following document.

DOCUMENT CONTENT:
{document_text}
"""
            ),
        ]

        return self.structured_llm.invoke(messages)