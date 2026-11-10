from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest
from watchfiles import Change

from agentic_ai_job_board.ingestion.events import FileSource
from agentic_ai_job_board.ingestion.event_handler import FileEventHandler
from agentic_ai_job_board.ingestion.file_validator import ValidationResult


@pytest.fixture
def handler(tmp_path: Path) -> FileEventHandler:
    """Create a handler with isolated temporary directories."""
    event_handler = FileEventHandler()

    event_handler.resume_dir = (tmp_path / "resumes").resolve()
    event_handler.posting_dir = (tmp_path / "postings").resolve()

    event_handler.resume_dir.mkdir(parents=True)
    event_handler.posting_dir.mkdir(parents=True)

    event_handler.validator.validate = AsyncMock()
    event_handler.quarantine_service.quarantine = Mock()
    event_handler.dispatcher.dispatch = AsyncMock()

    return event_handler


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("directory", "source"),
    [
        ("resumes", FileSource.RESUME),
        ("postings", FileSource.POSTING),
    ],
)
async def test_valid_file_is_dispatched(
    handler: FileEventHandler,
    tmp_path: Path,
    directory: str,
    source: FileSource,
) -> None:
    file_path = tmp_path / directory / "document.pdf"
    file_path.touch()

    handler.validator.validate.return_value = ValidationResult(
        is_valid=True,
        reason="File validation successful",
        retryable=False,
    )

    await handler.handle(Change.added, str(file_path))

    handler.validator.validate.assert_awaited_once_with(file_path)

    handler.dispatcher.dispatch.assert_awaited_once()
    event = handler.dispatcher.dispatch.await_args.args[0]

    assert event.source == source
    assert event.file_path == file_path
    assert event.event_type.value == "added"

    handler.quarantine_service.quarantine.assert_not_called()


@pytest.mark.asyncio
async def test_permanent_validation_failure_quarantines_file(
    handler: FileEventHandler,
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "resumes" / "invalid.csv"
    file_path.touch()

    handler.validator.validate.return_value = ValidationResult(
        is_valid=False,
        reason="Unsupported file extension: .csv",
        retryable=False,
    )

    await handler.handle(Change.added, str(file_path))

    handler.quarantine_service.quarantine.assert_called_once_with(
        file_path=file_path,
        source=FileSource.RESUME,
        reason="Unsupported file extension: .csv",
    )

    handler.dispatcher.dispatch.assert_not_awaited()


@pytest.mark.asyncio
async def test_retryable_validation_failure_does_not_quarantine(
    handler: FileEventHandler,
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "postings" / "job.pdf"
    file_path.touch()

    handler.validator.validate.return_value = ValidationResult(
        is_valid=False,
        reason="File did not become stable within the timeout",
        retryable=True,
    )

    await handler.handle(Change.added, str(file_path))

    handler.quarantine_service.quarantine.assert_not_called()
    handler.dispatcher.dispatch.assert_not_awaited()


@pytest.mark.asyncio
async def test_deleted_file_is_ignored(
    handler: FileEventHandler,
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "resumes" / "deleted.pdf"

    await handler.handle(Change.deleted, str(file_path))

    handler.validator.validate.assert_not_awaited()
    handler.quarantine_service.quarantine.assert_not_called()
    handler.dispatcher.dispatch.assert_not_awaited()


@pytest.mark.asyncio
async def test_file_outside_monitored_directories_is_ignored(
    handler: FileEventHandler,
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "other" / "document.pdf"

    await handler.handle(Change.added, str(file_path))

    handler.validator.validate.assert_not_awaited()
    handler.quarantine_service.quarantine.assert_not_called()
    handler.dispatcher.dispatch.assert_not_awaited()


@pytest.mark.asyncio
async def test_unsupported_change_is_ignored(
    handler: FileEventHandler,
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "resumes" / "document.pdf"

    await handler.handle(object(), str(file_path))  # type: ignore[arg-type]

    handler.validator.validate.assert_not_awaited()
    handler.quarantine_service.quarantine.assert_not_called()
    handler.dispatcher.dispatch.assert_not_awaited()