import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest
from watchfiles import Change

import agentic_ai_job_board.ingestion.watcher as watcher_module
from agentic_ai_job_board.ingestion.watcher import FileWatcher


@pytest.fixture
def watcher_setup(tmp_path: Path, monkeypatch):
    """Create a watcher with temporary directories and a mocked handler."""
    resume_dir = tmp_path / "resumes"
    posting_dir = tmp_path / "postings"

    monkeypatch.setattr(watcher_module.settings, "RESUME_DIR", resume_dir)
    monkeypatch.setattr(watcher_module.settings, "POSTING_DIR", posting_dir)

    event_handler = Mock()
    event_handler.handle = AsyncMock()

    monkeypatch.setattr(
        watcher_module,
        "FileEventHandler",
        lambda: event_handler,
    )

    watcher = FileWatcher()

    return watcher, event_handler, resume_dir, posting_dir


def test_watcher_initializes_directories(watcher_setup):
    watcher, _, resume_dir, posting_dir = watcher_setup

    assert resume_dir.is_dir()
    assert posting_dir.is_dir()

    assert watcher.watch_dirs == (
        resume_dir.resolve(),
        posting_dir.resolve(),
    )


@pytest.mark.asyncio
async def test_watcher_forwards_filesystem_events(
    watcher_setup,
    monkeypatch,
    tmp_path: Path,
):
    watcher, event_handler, _, _ = watcher_setup

    file_path = str(tmp_path / "resumes" / "resume.pdf")
    changes = {
        (Change.added, file_path),
        (Change.modified, file_path),
    }

    async def fake_awatch(*args, **kwargs):
        yield changes

    monkeypatch.setattr(watcher_module, "awatch", fake_awatch)

    await watcher.run()

    assert event_handler.handle.await_count == 2
    event_handler.handle.assert_any_await(Change.added, file_path)
    event_handler.handle.assert_any_await(Change.modified, file_path)


@pytest.mark.asyncio
async def test_watcher_continues_after_individual_event_failure(
    watcher_setup,
    monkeypatch,
    tmp_path: Path,
):
    watcher, event_handler, _, _ = watcher_setup

    first_file = str(tmp_path / "resumes" / "first.pdf")
    second_file = str(tmp_path / "resumes" / "second.pdf")

    event_handler.handle.side_effect = [
        RuntimeError("Event processing failed"),
        None,
    ]

    async def fake_awatch(*args, **kwargs):
        yield {
            (Change.added, first_file),
            (Change.added, second_file),
        }

    monkeypatch.setattr(watcher_module, "awatch", fake_awatch)

    await watcher.run()

    assert event_handler.handle.await_count == 2


@pytest.mark.asyncio
async def test_watcher_propagates_cancellation(
    watcher_setup,
    monkeypatch,
):
    watcher, _, _, _ = watcher_setup

    async def fake_awatch(*args, **kwargs):
        raise asyncio.CancelledError
        yield  # Makes this an async generator.

    monkeypatch.setattr(watcher_module, "awatch", fake_awatch)

    with pytest.raises(asyncio.CancelledError):
        await watcher.run()


@pytest.mark.asyncio
async def test_watcher_propagates_unexpected_errors(
    watcher_setup,
    monkeypatch,
):
    watcher, _, _, _ = watcher_setup

    async def fake_awatch(*args, **kwargs):
        raise RuntimeError("Watcher failed")
        yield  # Makes this an async generator.

    monkeypatch.setattr(watcher_module, "awatch", fake_awatch)

    with pytest.raises(RuntimeError, match="Watcher failed"):
        await watcher.run()


def test_stop_sets_stop_event(watcher_setup):
    watcher, _, _, _ = watcher_setup

    assert not watcher.stop_event.is_set()

    watcher.stop()

    assert watcher.stop_event.is_set()