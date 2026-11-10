
from langchain_core.messages import HumanMessage, SystemMessage

from agentic_ai_job_board.agents.document_validation.prompts import (
    DOCUMENT_VALIDATION_SYSTEM_PROMPT,
)
from agentic_ai_job_board.agents.document_validation.schemas import (
    DocumentValidationExtraction,
)
from agentic_ai_job_board.gateway.router import get_chat_model


class DocumentValidationAgent:
    """Classify documents and extract structured information."""

    def __init__(self) -> None:
        self.llm = get_chat_model()

        self.structured_llm = self.llm.with_structured_output(
            DocumentValidationExtraction
        )

    def invoke(self, document_text: str) -> DocumentValidationExtraction:
        """Classify and extract information from document text."""

        if not document_text or not document_text.strip():
            raise ValueError("Document text cannot be empty.")

        messages = [
            SystemMessage(content=DOCUMENT_VALIDATION_SYSTEM_PROMPT),
            HumanMessage(
                content=f"""
Analyze the following document.

DOCUMENT CONTENT:
{document_text}
"""
            ),
        ]

        result = self.structured_llm.invoke(messages)

        # Enforce the relationship between classification and extracted data.
        document_type = result.classification.document_type

        if document_type.value == "resume":
            if result.resume_data is None:
                raise ValueError(
                    "Resume classification requires resume_data."
                )

            if result.job_posting_data is not None:
                raise ValueError(
                    "Resume classification must not contain job_posting_data."
                )

        elif document_type.value == "job_posting":
            if result.job_posting_data is None:
                raise ValueError(
                    "Job posting classification requires job_posting_data."
                )

            if result.resume_data is not None:
                raise ValueError(
                    "Job posting classification must not contain resume_data."
                )

        elif document_type.value == "unknown":
            if (
                result.resume_data is not None
                or result.job_posting_data is not None
            ):
                raise ValueError(
                    "Unknown documents must not contain extracted document data."
                )

        return result