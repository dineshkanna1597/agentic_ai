from langchain.agents import create_agent
from langgraph.runtime import Runtime

from agentic_ai_job_board.agents.data_ingestion.context import (
    DataIngestionContext,
)
from agentic_ai_job_board.agents.data_ingestion.prompts import (
    DATA_INGESTION_SYSTEM_PROMPT,
)
from agentic_ai_job_board.agents.data_ingestion.state import (
    DataIngestionState,
)
from agentic_ai_job_board.gateway.router import get_chat_model
from agentic_ai_job_board.tools.chunking_tools import chunk_document


class DataIngestionAgent:
    """Agent responsible for document chunking decisions."""

    def __init__(self) -> None:
        self.agent = create_agent(
            model=get_chat_model(),
            tools=[chunk_document],
            system_prompt=DATA_INGESTION_SYSTEM_PROMPT,
            context_schema=DataIngestionContext,
        )

    async def run(
        self,
        state: DataIngestionState,
        runtime: Runtime[DataIngestionContext],
    ) -> dict:
        """Analyze the document and generate chunks."""

        if runtime.context.document is None:
            raise RuntimeError(
                "Document chunking failed: prepared document is missing."
            )

        response = await self.agent.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            f"Document type: {runtime.context.document_type}\n"
                            f"Document ID: {runtime.context.document_id}\n\n"
                            "Analyze this document and choose the best "
                            "chunking strategy:\n\n"
                            f"{runtime.context.document.page_content}"
                        ),
                    }
                ]
            },
            context=runtime.context,
        )

        if not runtime.context.chunks:
            raise RuntimeError(
                "Document chunking failed: no chunks were generated."
            )

        return {}