
import asyncio

from mcp.server.mcpserver import MCPServer

from agentic_ai_job_board.mcp.tools.posting_tools import (
    search_job_postings,
)
from agentic_ai_job_board.mcp.tools.resume_tools import (
    search_resumes,
)


server = MCPServer(
    name="agentic-rag-mcp-server",
    instructions=(
        "Provides semantic search tools for candidate resumes "
        "and job postings in the recruitment knowledge base."
    ),
)

server.add_tool(
    fn=search_resumes,
    name="search_resumes",
    description=(
        "Search candidate resumes using a natural-language query. "
        "Returns relevant resume chunks with metadata and similarity scores."
    ),
)

server.add_tool(
    fn=search_job_postings,
    name="search_job_postings",
    description=(
        "Search job postings using a natural-language query. "
        "Returns relevant job posting chunks with metadata and similarity scores."
    ),
)


async def main() -> None:
    """Start the MCP server using Streamable HTTP."""

    await server.run_streamable_http_async(
        host="127.0.0.1",
        port=8001,
        streamable_http_path="/mcp",
    )


if __name__ == "__main__":
    asyncio.run(main())
