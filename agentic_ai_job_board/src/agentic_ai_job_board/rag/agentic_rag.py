from agentic_ai_job_board.gateway.router import get_embedding_model, get_chat_model
from langchain_qdrant import QdrantVectorStore
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from qdrant_client import QdrantClient, models
from typing import Annotated, Literal, TypedDict
from langchain_core.messages import BaseMessage
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STORAGE_PATH = os.path.join(BASE_DIR, "qdrant_local_storage")

qdrant_client = QdrantClient(path=STORAGE_PATH)
collection_name = 'career_portal'

embeddings = get_embedding_model()

vector_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name=collection_name,
    embedding=embeddings,
)

retriever = vector_store.as_retriever(
    search_kwargs={
        "k": 10,
        "filter": models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.type",
                    match=models.MatchValue(value="resume")
                )
            ]
        )
    }
)

@tool
def search_resumes(query: str) -> str:
    """Search only candidate resumes stored in Qdrant."""

    docs = retriever.invoke(query)

    if not docs:
        return "No matching resumes found."

    results = []

    for doc in docs:
        source = doc.metadata.get("source", "Unknown file")

        results.append(
            f"Resume: {source}\n"
            f"Content:\n{doc.page_content}"
        )

    return "\n\n---\n\n".join(results)

tools = [search_resumes]
llm = get_chat_model().bind_tools(tools)
tool_node = ToolNode(tools)

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def call_model(state: State):
    system_message = SystemMessage(
        content="""
        You are a resume search assistant.

        Always use search_resumes for candidate searches.
        Use only retrieved resume information.
        Never invent candidate details.
        Keep responses concise.
        """
    )

    messages = [system_message] + state["messages"]

    sanitized_messages = []

    for message in messages:
        message = message.model_copy(deep=True)

        message.additional_kwargs.pop("reasoning_content", None)
        message.response_metadata.pop("reasoning_content", None)

        sanitized_messages.append(message)

    response = llm.invoke(sanitized_messages)

    return {"messages": [response]}

def should_continue(state: State) -> Literal["tools", END]: # type: ignore[reportInvalidTypeForm]
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return END

workflow = StateGraph(State)
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "agent")

# Compile graph exportable for LangGraph server (`langgraph.json`)
graph = workflow.compile()