
from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgenticRAGState(TypedDict, total=False):
    """Checkpointable state for the Agentic RAG workflow."""

    query: str

    messages: Annotated[
        list[BaseMessage],
        add_messages,
    ]

    selected_tools: list[str]
    tool_results: list[dict[str, Any]]

    # Skills explicitly requested in the user's query.
    requested_skills: list[str]

    # Individual chunks retrieved from the knowledge base.
    retrieved_context: list[dict[str, Any]]

    # Unique document-level source information.
    sources: list[dict[str, Any]]

    # Deterministic document-level skill verification results.
    verification_results: list[dict[str, Any]]

    verification_complete: bool

    tool_call_count: int

    final_response: str
    error_message: str
