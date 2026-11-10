import asyncio
from pathlib import Path

import pytest

from agentic_ai_job_board.ingestion.file_validator import FileValidator


@pytest.fixture
def validator():
    return FileValidator()


@pytest.fixture
def stable_validator(monkeypatch):
    instance = FileValidator()

    async def mock_wait_until_stable(file_path: Path) -> bool:
        return True

    monkeypatch.setattr(
        instance,
        "wait_until_stable",
        mock_wait_until_stable,
    )

    return instance


@pytest.mark.parametrize("extension", [".pdf", ".docx", ".txt"])
def test_supported_file_extensions(
    stable_validator,
    tmp_path,
    extension,
):
    file_path = tmp_path / f"resume{extension}"
    file_path.write_text("Sample resume content")

    result = asyncio.run(stable_validator.validate(file_path))

    assert result.is_valid is True


def test_unsupported_file_extension(stable_validator, tmp_path):
    file_path = tmp_path / "resume.csv"
    file_path.write_text("Sample content")

    result = asyncio.run(stable_validator.validate(file_path))

    assert result.is_valid is False


def test_empty_file_is_rejected(stable_validator, tmp_path):
    file_path = tmp_path / "empty.txt"
    file_path.touch()

    result = asyncio.run(stable_validator.validate(file_path))

    assert result.is_valid is False


def test_missing_file_is_rejected(validator, tmp_path):
    file_path = tmp_path / "missing.txt"

    result = asyncio.run(validator.validate(file_path))

    assert result.is_valid is False


def test_unstable_file_is_rejected(validator, monkeypatch, tmp_path):
    file_path = tmp_path / "unstable.txt"
    file_path.write_text("Sample content")

    async def mock_wait_until_stable(file_path: Path) -> bool:
        return False

    monkeypatch.setattr(
        validator,
        "wait_until_stable",
        mock_wait_until_stable,
    )

    result = asyncio.run(validator.validate(file_path))

    assert result.is_valid is False

def test_oversized_file_is_rejected(
    stable_validator,
    monkeypatch,
    tmp_path,
):
    file_path = tmp_path / "large.txt"
    file_path.write_text("Sample content")

    monkeypatch.setattr(
        stable_validator,
        "max_file_size",
        0,
    )

    result = asyncio.run(stable_validator.validate(file_path))

    assert result.is_valid is False
    assert result.reason == "File exceeds maximum allowed size"
    assert result.retryable is False