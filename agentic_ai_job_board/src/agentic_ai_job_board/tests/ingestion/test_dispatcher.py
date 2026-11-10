import asyncio
from pathlib import Path
from unittest.mock import Mock

from agentic_ai_job_board.ingestion.dispatcher import IngestionDispatcher
from agentic_ai_job_board.ingestion.events import (
    FileEvent,
    FileEventType,
    FileSource,
)


def create_event(
    source: FileSource,
    event_type: FileEventType = FileEventType.ADDED,
) -> FileEvent:
    return FileEvent.create(
        event_type=event_type,
        source=source,
        file_path=Path("test_file.txt"),
    )


def test_resume_event_is_dispatched(monkeypatch):
    mock_apply_async = Mock(return_value=Mock(id="resume-task-123"))

    from agentic_ai_job_board.ingestion import dispatcher

    monkeypatch.setattr(
        dispatcher.process_document,
        "apply_async",
        mock_apply_async,
    )

    event = create_event(FileSource.RESUME)

    asyncio.run(IngestionDispatcher().dispatch(event))

    mock_apply_async.assert_called_once()

    args = mock_apply_async.call_args.kwargs["args"]

    assert args[0] == "test_file.txt"
    assert args[1] == "resume"
    assert args[2] == "added"


def test_posting_event_is_dispatched(monkeypatch):
    mock_apply_async = Mock(return_value=Mock(id="posting-task-123"))

    from agentic_ai_job_board.ingestion import dispatcher

    monkeypatch.setattr(
        dispatcher.process_document,
        "apply_async",
        mock_apply_async,
    )

    event = create_event(FileSource.POSTING)

    asyncio.run(IngestionDispatcher().dispatch(event))

    mock_apply_async.assert_called_once()

    args = mock_apply_async.call_args.kwargs["args"]

    assert args[0] == "test_file.txt"
    assert args[1] == "posting"
    assert args[2] == "added"


def test_deleted_event_is_not_dispatched(monkeypatch):
    from agentic_ai_job_board.ingestion import dispatcher

    mock_apply_async = Mock()

    monkeypatch.setattr(
        dispatcher.process_document,
        "apply_async",
        mock_apply_async,
    )

    event = create_event(
        FileSource.RESUME,
        FileEventType.DELETED,
    )

    asyncio.run(IngestionDispatcher().dispatch(event))

    mock_apply_async.assert_not_called()