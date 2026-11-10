from langchain_core.messages import AIMessage, HumanMessage

from agentic_ai_job_board.agents.agentic_rag.agent import (
    AgenticRAGAgent,
)


def test_sanitize_messages_removes_reasoning_content():
    message = AIMessage(
        content="Here are the matching candidates.",
        additional_kwargs={
            "reasoning_content": "Internal reasoning",
            "other_field": "preserve this",
        },
    )

    sanitized = AgenticRAGAgent.sanitize_messages([message])

    assert len(sanitized) == 1
    assert sanitized[0].content == message.content
    assert "reasoning_content" not in sanitized[0].additional_kwargs
    assert sanitized[0].additional_kwargs["other_field"] == "preserve this"


def test_sanitize_messages_preserves_messages_without_reasoning_content():
    message = HumanMessage(
        content="Find Python developers with Django experience."
    )

    sanitized = AgenticRAGAgent.sanitize_messages([message])

    assert len(sanitized) == 1
    assert sanitized[0] is message


def test_sanitize_messages_does_not_mutate_original_message():
    message = AIMessage(
        content="Candidate results.",
        additional_kwargs={
            "reasoning_content": "Internal reasoning",
        },
    )

    sanitized = AgenticRAGAgent.sanitize_messages([message])

    assert "reasoning_content" in message.additional_kwargs
    assert "reasoning_content" not in sanitized[0].additional_kwargs
    assert sanitized[0] is not message
