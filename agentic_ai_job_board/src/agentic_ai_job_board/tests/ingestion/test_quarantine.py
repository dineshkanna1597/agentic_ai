from pathlib import Path

from agentic_ai_job_board.ingestion.events import FileSource
from agentic_ai_job_board.ingestion.quarantine import QuarantineService


def test_quarantine_moves_file_to_source_directory(
    tmp_path,
    monkeypatch,
):
    data_dir = tmp_path / "data"
    monkeypatch.setattr(
        "agentic_ai_job_board.ingestion.quarantine.settings.DATA_DIR",
        data_dir,
    )

    original_file = tmp_path / "resume.txt"
    original_file.write_text("Invalid resume content")

    service = QuarantineService()
    destination = service.quarantine(
        original_file,
        FileSource.RESUME,
        "Unsupported content",
    )

    assert destination is not None
    assert destination.exists()
    assert destination.parent == data_dir / "rejected" / "resume"
    assert destination.read_text() == "Invalid resume content"
    assert not original_file.exists()


def test_quarantine_uses_unique_filenames(
    tmp_path,
    monkeypatch,
):
    data_dir = tmp_path / "data"
    monkeypatch.setattr(
        "agentic_ai_job_board.ingestion.quarantine.settings.DATA_DIR",
        data_dir,
    )

    service = QuarantineService()
    destinations = []

    for _ in range(2):
        original_file = tmp_path / "resume.txt"
        original_file.write_text("Invalid resume content")

        destination = service.quarantine(
            original_file,
            FileSource.RESUME,
            "Rejected",
        )
        destinations.append(destination)

    assert all(path is not None for path in destinations)
    assert destinations[0] != destinations[1]
    assert all(path.exists() for path in destinations)


def test_quarantine_returns_none_for_missing_file(
    tmp_path,
    monkeypatch,
):
    data_dir = tmp_path / "data"
    monkeypatch.setattr(
        "agentic_ai_job_board.ingestion.quarantine.settings.DATA_DIR",
        data_dir,
    )

    service = QuarantineService()
    missing_file = tmp_path / "missing.txt"

    destination = service.quarantine(
        missing_file,
        FileSource.RESUME,
        "Missing file",
    )

    assert destination is None


def test_quarantine_returns_none_on_move_error(
    tmp_path,
    monkeypatch,
):
    data_dir = tmp_path / "data"
    monkeypatch.setattr(
        "agentic_ai_job_board.ingestion.quarantine.settings.DATA_DIR",
        data_dir,
    )

    original_file = tmp_path / "resume.txt"
    original_file.write_text("Invalid resume content")

    def raise_os_error(src, dst):
        raise OSError("Simulated filesystem error")

    monkeypatch.setattr(
        "agentic_ai_job_board.ingestion.quarantine.shutil.move",
        raise_os_error,
    )

    service = QuarantineService()
    destination = service.quarantine(
        original_file,
        FileSource.RESUME,
        "Rejected",
    )

    assert destination is None
    assert original_file.exists()