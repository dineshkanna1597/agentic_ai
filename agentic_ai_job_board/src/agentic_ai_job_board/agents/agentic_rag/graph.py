from agentic_ai_job_board.agents.agentic_rag.agent import (
    AgenticRAGAgent,
)


async def make_graph():
    """Build and return the Agentic RAG graph."""

    agent = AgenticRAGAgent(
        max_tool_calls=3,
    )

    return await agent.build()