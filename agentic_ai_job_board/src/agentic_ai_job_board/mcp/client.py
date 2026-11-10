
import json
from typing import Any

from langchain_core.tools import StructuredTool
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


class MCPToolClient:
    """Discover MCP tools and expose them as LangChain tools."""

    def __init__(
        self,
        server_url: str = "http://127.0.0.1:8001/mcp",
    ) -> None:
        self.server_url = server_url

    async def get_tools(self) -> list[StructuredTool]:
        """Discover MCP tools and convert them into LangChain tools."""

        async with streamable_http_client(
            self.server_url
        ) as (read_stream, write_stream):

            async with ClientSession(
                read_stream,
                write_stream,
            ) as session:

                await session.initialize()
                mcp_tools = await session.list_tools()

        langchain_tools = []

        for mcp_tool in mcp_tools.tools:
            tool_name = mcp_tool.name

            async def call_mcp_tool(
                _tool_name: str = tool_name,
                **arguments: Any,
            ) -> list[Any]:
                """Execute an MCP tool and return its results."""

                async with streamable_http_client(
                    self.server_url
                ) as (read_stream, write_stream):

                    async with ClientSession(
                        read_stream,
                        write_stream,
                    ) as session:

                        await session.initialize()

                        result = await session.call_tool(
                            _tool_name,
                            arguments=arguments,
                        )

                if result.is_error:
                    raise RuntimeError(
                        f"MCP tool '{_tool_name}' execution failed."
                    )

                parsed_results = []

                for content in result.content:
                    if hasattr(content, "text"):
                        try:
                            parsed_results.append(
                                json.loads(content.text)
                            )
                        except json.JSONDecodeError:
                            parsed_results.append(content.text)

                return parsed_results
            input_schema = dict(mcp_tool.input_schema)
            input_schema["properties"] = {
                name: schema
                for name, schema in input_schema.get("properties", {}).items()
                if name != "score_threshold"
            }

            langchain_tool = StructuredTool.from_function(
                coroutine=call_mcp_tool,
                name=tool_name,
                description=mcp_tool.description or tool_name,
                args_schema=input_schema,
            )

            langchain_tools.append(langchain_tool)

        return langchain_tools
