from unittest.mock import MagicMock, patch

import pytest

from agentic_ai_job_board.agents.document_quality.agent import (
    DocumentQualityAgent,
)


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    structured_llm = MagicMock()
    llm.with_structured_output.return_value = structured_llm
    return llm, structured_llm


@patch("agentic_ai_job_board.agents.document_quality.agent.get_chat_model")
def test_agent_initialization(mock_get_chat_model, mock_llm):
    llm, structured_llm = mock_llm
    mock_get_chat_model.return_value = llm

    agent = DocumentQualityAgent()

    mock_get_chat_model.assert_called_once()
    llm.with_structured_output.assert_called_once()

    assert agent.structured_llm is structured_llm


@patch("agentic_ai_job_board.agents.document_quality.agent.get_chat_model")
def test_agent_invokes_structured_llm(mock_get_chat_model, mock_llm):
    llm, structured_llm = mock_llm
    mock_get_chat_model.return_value = llm

    expected_result = MagicMock()
    structured_llm.invoke.return_value = expected_result

    agent = DocumentQualityAgent()

    result = agent.invoke("Python Developer with Django experience")

    assert result is expected_result
    structured_llm.invoke.assert_called_once()

    messages = structured_llm.invoke.call_args.args[0]

    assert len(messages) == 2
    assert "quality assessment specialist" in messages[0].content.lower()
    assert "Python Developer with Django experience" in messages[1].content


@patch("agentic_ai_job_board.agents.document_quality.agent.get_chat_model")
def test_agent_rejects_empty_document(mock_get_chat_model, mock_llm):
    llm, _ = mock_llm
    mock_get_chat_model.return_value = llm

    agent = DocumentQualityAgent()

    with pytest.raises(ValueError, match="Document text cannot be empty"):
        agent.invoke("")


@patch("agentic_ai_job_board.agents.document_quality.agent.get_chat_model")
def test_agent_rejects_whitespace_document(mock_get_chat_model, mock_llm):
    llm, _ = mock_llm
    mock_get_chat_model.return_value = llm

    agent = DocumentQualityAgent()

    with pytest.raises(ValueError, match="Document text cannot be empty"):
        agent.invoke("   ")