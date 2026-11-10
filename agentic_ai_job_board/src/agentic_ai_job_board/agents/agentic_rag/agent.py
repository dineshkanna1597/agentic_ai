
import json
from dataclasses import asdict
from typing import Any
from langchain_core.messages import AIMessage
from langchain_core.messages import SystemMessage
from langchain_core.messages import ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from agentic_ai_job_board.agents.agentic_rag.prompts import (
    AGENTIC_RAG_SYSTEM_PROMPT,
)
from agentic_ai_job_board.agents.agentic_rag.state import (
    AgenticRAGState,
)
from agentic_ai_job_board.gateway.router import get_chat_model
from agentic_ai_job_board.mcp.client import MCPToolClient

from agentic_ai_job_board.services.verification.skill_extraction_service import (
    SkillExtractionService,
)
from agentic_ai_job_board.services.verification.skill_verification_service import (
    SkillVerificationService,
)
from langchain_core.messages import HumanMessage

class AgenticRAGAgent:
    """Agentic RAG agent using LangGraph and MCP tools."""

    def __init__(
        self,
        mcp_server_url: str = "http://127.0.0.1:8001/mcp",
        max_tool_calls: int = 3,
    ) -> None:
        if max_tool_calls < 1:
            raise ValueError("max_tool_calls must be at least 1.")

        self.mcp_client = MCPToolClient(
            server_url=mcp_server_url,
        )

        self.llm = get_chat_model()
        self.max_tool_calls = max_tool_calls

    @staticmethod
    def sanitize_messages(messages: list) -> list:
        """Remove provider-specific fields unsupported by Groq."""

        sanitized_messages = []

        for message in messages:
            if (
                isinstance(message, AIMessage)
                and "reasoning_content" in message.additional_kwargs
            ):
                additional_kwargs = dict(message.additional_kwargs)
                additional_kwargs.pop("reasoning_content", None)

                message = message.model_copy(
                    update={
                        "additional_kwargs": additional_kwargs,
                    }
                )

            sanitized_messages.append(message)

        return sanitized_messages

    async def agent_node(
        self,
        state: AgenticRAGState,
    ) -> dict:
        """Select tools or generate the final response."""

        messages = state.get("messages", [])
        if not messages and state.get("query"):
            messages = [
                HumanMessage(content=state["query"])
            ]
        messages = self.sanitize_messages(messages)

        is_new_user_turn = bool(
            messages and isinstance(messages[-1], HumanMessage)
        )

        turn_state = {}

        if is_new_user_turn:
            turn_state = {
                "query": messages[-1].content,
                "requested_skills": [],
                "retrieved_context": [],
                "sources": [],
                "verification_results": [],
                "verification_complete": False,
                "tool_call_count": 0,
                "tool_results": [],
                "selected_tools": [],
                "final_response": "",
        }

        tool_call_count = (
            0 if is_new_user_turn
            else state.get("tool_call_count", 0)
        )

        system_message = SystemMessage(
            content=AGENTIC_RAG_SYSTEM_PROMPT,
        )

        if state.get("verification_complete", False) and not is_new_user_turn:
            requested_skills = state.get("requested_skills", [])
            verification_results = state.get(
                "verification_results", []
            )

            if requested_skills:
                final_response = self.format_skill_verification(
                    requested_skills=requested_skills,
                    verification_results=verification_results,
                )

                return {
                    **turn_state,
                    "final_response": final_response,
                }

            print("\n========== FINAL ANSWER MESSAGE HISTORY ==========")

            for index, message in enumerate(messages):
                print(f"\nMESSAGE {index}")
                print("TYPE:", type(message).__name__)
                print("CONTENT:", message.content)
                print("TOOL CALLS:", getattr(message, "tool_calls", None))

            response = await self.llm.ainvoke(
                [
                    system_message,
                    *messages,
                ]
            )

            return {
                **turn_state,
                "messages": [response],
                "final_response": response.content,
            }
        
        if tool_call_count >= self.max_tool_calls:
            final_instruction = SystemMessage(
                content=(
                    AGENTIC_RAG_SYSTEM_PROMPT
                    + "\n\n"
                    + "The tool execution limit has been reached. "
                    + "Do not request more tools. Generate your final "
                    + "answer using the information already retrieved. "
                    + "Clearly identify any missing information."
                ),
            )

            response = await self.llm.ainvoke(
                [
                    final_instruction,
                    *messages,
                ]
            )

        else:
            response = await self.llm_with_tools.ainvoke(
                [
                    system_message,
                    *messages,
                ]
            )

        result = {
                **turn_state,
                "messages": [response],
            }

        if not response.tool_calls:
            result["final_response"] = response.content

        return result

    @staticmethod
    def route_after_agent(
        state: AgenticRAGState,
    ) -> str:
        """Route to tools, verification, or final completion."""

        last_message = state["messages"][-1]

        if last_message.tool_calls:
            return "tools"

        if not state.get("verification_complete", False):
            return "verify"

        return END

    @staticmethod
    def increment_tool_call_count(
        state: AgenticRAGState,
    ) -> dict:
        """Count requested tool executions."""

        last_message = state["messages"][-1]

        return {
            "tool_call_count": (
                state.get("tool_call_count", 0)
                + len(last_message.tool_calls)
            )
        }

    
    @staticmethod
    def collect_sources(
        state: AgenticRAGState,
    ) -> dict[str, Any]:
        """Collect retrieved chunks and deduplicate document sources."""

        retrieved_context = list(
            state.get("retrieved_context", [])
        )

        sources_by_key = {}

        # Preserve sources already collected in previous iterations.
        for source in state.get("sources", []):
            document_id = source.get("document_id")
            source_name = source.get("source")

            key = (
                f"id:{document_id}"
                if document_id
                else f"source:{source_name}"
                if source_name
                else None
            )

            if key:
                sources_by_key[key] = source

        # Process the latest consecutive tool messages.
        for message in reversed(state.get("messages", [])):
            if not isinstance(message, ToolMessage):
                break

            content = message.content

            if isinstance(content, str):
                try:
                    content = json.loads(content)
                except json.JSONDecodeError:
                    continue

            if isinstance(content, dict):
                content = [content]

            if not isinstance(content, list):
                continue

            for item in content:
                if not isinstance(item, dict):
                    continue

                if "content" not in item or "metadata" not in item:
                    continue

                retrieved_context.append(item)

                document_id = item.get("document_id")
                source_name = item.get("source")

                key = (
                    f"id:{document_id}"
                    if document_id
                    else f"source:{source_name}"
                    if source_name
                    else None
                )

                if not key:
                    continue

                sources_by_key[key] = {
                    "document_id": document_id,
                    "source": source_name,
                    "document_type": item.get("document_type"),
                }

        return {
            "retrieved_context": retrieved_context,
            "sources": list(sources_by_key.values()),
        }

    @staticmethod
    def format_skill_verification(
        requested_skills: list[str],
        verification_results: list[dict[str, Any]],
    ) -> str:
        """Format deterministic document skill verification results."""

        if not requested_skills:
            return (
                "I couldn't identify specific skills in your query "
                "to verify against the retrieved documents."
            )

        full_matches = []
        partial_matches = []

        for result in verification_results:
            verification = result.get("verification", {})
            source = result.get("source") or "Unknown source"

            candidate = {
                "source": source,
                "missing_skills": verification.get(
                    "missing_skills", []
                ),
            }

            if verification.get("is_full_match", False):
                full_matches.append(candidate)
            else:
                partial_matches.append(candidate)

        lines = [
            f"**Requested skills:** {', '.join(requested_skills)}",
            "",
            "### Full matches",
        ]

        if full_matches:
            for candidate in full_matches:
                lines.append(f"- **{candidate['source']}**")
        else:
            lines.append("- No full matches found.")

        lines.extend(["", "### Partial matches"])

        if partial_matches:
            for candidate in partial_matches:
                missing = ", ".join(candidate["missing_skills"])
                lines.append(
                    f"- **{candidate['source']}** — "
                    f"Missing: {missing or 'Not specified'}"
                )
        else:
            lines.append("- No partial matches found.")

        return "\n".join(lines)

    def verify_retrieved_documents(
        self,
        state: AgenticRAGState,
    ) -> dict[str, Any]:
        """Extract requested skills and verify retrieved documents."""

        query = state.get("query", "")

        extraction_service = SkillExtractionService()
        verification_service = SkillVerificationService()

        requested_skills = extraction_service.extract(query)

        verification_results = verification_service.verify_documents(
            retrieved_chunks=state.get("retrieved_context", []),
            requested_skills=requested_skills,
        )

        return {
            "requested_skills": requested_skills,
            "verification_results": [
                asdict(result)
                for result in verification_results
            ],
            "verification_complete": True,
        }

    async def build(self):
        """Discover tools and compile the Agentic RAG graph."""

        tools = await self.mcp_client.get_tools()

        self.llm_with_tools = self.llm.bind_tools(tools)

        tool_node = ToolNode(tools)

        graph = StateGraph(AgenticRAGState)

        graph.add_node("agent", self.agent_node)
        graph.add_node(
            "increment_tool_count",
            self.increment_tool_call_count,
        )
        graph.add_node("tools", tool_node)
        graph.add_node("collect_sources", self.collect_sources)
        graph.add_node(
            "verify_skills",
            self.verify_retrieved_documents,
        )

        graph.add_edge(START, "agent")

        graph.add_conditional_edges(
            "agent",
            self.route_after_agent,
            {
                "tools": "increment_tool_count",
                "verify": "verify_skills",
                END: END,
            },
        )

        graph.add_edge("increment_tool_count", "tools")
        graph.add_edge("tools", "collect_sources")
        graph.add_edge("collect_sources", "agent")
        graph.add_edge("verify_skills", "agent")

        return graph.compile()
