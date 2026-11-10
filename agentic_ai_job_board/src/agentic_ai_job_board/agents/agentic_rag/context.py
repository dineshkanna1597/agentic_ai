
from dataclasses import dataclass
from typing import Any


@dataclass
class AgenticRAGContext:
    """Runtime dependencies for the Agentic RAG workflow."""

    mcp_client: Any
    max_tool_calls: int = 8
    max_retrieval_results: int = 10
