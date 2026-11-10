from langgraph.graph import END, START, StateGraph

from agentic_ai_job_board.agents.data_ingestion.agent import (
    DataIngestionAgent,
)
from agentic_ai_job_board.agents.data_ingestion.context import (
    DataIngestionContext,
)
from agentic_ai_job_board.agents.data_ingestion.nodes import (
    extract_document,
    normalize_document,
    detect_and_redact_pii,
    prepare_redacted_document,
    extract_resume_contacts,
    run_document_validation_agent,
    validate_document,
    route_after_validation,
    run_document_quality_agent,
    assess_document_quality,
    route_after_quality_assessment,
    generate_embeddings,
    store_vectors,
    finalize_non_accepted_document,
)
from agentic_ai_job_board.agents.data_ingestion.state import (
    DataIngestionState,
)


class DataIngestionGraph:
    """LangGraph workflow for document ingestion."""

    def __init__(self) -> None:
        self.agent = DataIngestionAgent()

        builder = StateGraph(
            DataIngestionState,
            context_schema=DataIngestionContext,
        )

        builder.add_node("extract_document", extract_document)
        builder.add_node("normalize_document", normalize_document)
        builder.add_node("detect_and_redact_pii", detect_and_redact_pii)
        builder.add_node("prepare_redacted_document", prepare_redacted_document)
        builder.add_node("extract_resume_contacts", extract_resume_contacts)
        builder.add_node(
            "run_document_validation_agent",
            run_document_validation_agent,
        )
        builder.add_node("validate_document", validate_document)
        builder.add_node(
            "run_document_quality_agent",
            run_document_quality_agent,
        )
        builder.add_node("assess_document_quality", assess_document_quality)

        builder.add_node("data_ingestion_agent", self.agent.run)
        builder.add_node("generate_embeddings", generate_embeddings)
        builder.add_node("store_vectors", store_vectors)
        builder.add_edge(START, "extract_document")

        builder.add_edge("extract_document", "normalize_document")
        builder.add_edge("normalize_document", "detect_and_redact_pii")
        builder.add_node(
            "finalize_non_accepted_document",
            finalize_non_accepted_document,
        )
        builder.add_edge("detect_and_redact_pii", "prepare_redacted_document")
        builder.add_edge("prepare_redacted_document", "extract_resume_contacts")
        builder.add_edge("extract_resume_contacts", "run_document_validation_agent")
        builder.add_edge(
            "run_document_validation_agent",
            "validate_document",
        )
        builder.add_conditional_edges(
            "validate_document",
            route_after_validation,
            {
                "accepted": "run_document_quality_agent",
                "rejected": "finalize_non_accepted_document",
                "needs_review": "finalize_non_accepted_document",
            },
        )
        builder.add_edge(
            "run_document_quality_agent",
            "assess_document_quality",
        )

        builder.add_conditional_edges(
            "assess_document_quality",
            route_after_quality_assessment,
            {
                "accepted": "data_ingestion_agent",
                "rejected": "finalize_non_accepted_document",
                "needs_review": "finalize_non_accepted_document",
            },
        )
        builder.add_edge("finalize_non_accepted_document", END)

        builder.add_edge(
            "data_ingestion_agent",
            "generate_embeddings",
        )

        builder.add_edge(
            "generate_embeddings",
            "store_vectors",
        )

        builder.add_edge("store_vectors", END)

        self.graph = builder.compile()

    async def run(self, context: DataIngestionContext) -> dict:
        """Execute the document ingestion workflow."""

        return await self.graph.ainvoke(
            {},
            context=context,
        )
graph = DataIngestionGraph().graph