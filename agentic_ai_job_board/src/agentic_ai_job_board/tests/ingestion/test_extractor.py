
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from agentic_ai_job_board.ingestion.extractor import DocumentExtractor


@pytest.fixture
def extractor():
    return DocumentExtractor()


@pytest.fixture
def sample_documents():
    return [
        Document(
            page_content="Python Developer with Django experience",
            metadata={"page": 0},
        )
    ]


@pytest.mark.parametrize(
    ("extension", "loader_name"),
    [
        (".pdf", "PyMuPDFLoader"),
        (".docx", "Docx2txtLoader"),
        (".txt", "TextLoader"),
    ],
)
def test_extract_uses_correct_loader(
    extractor,
    tmp_path: Path,
    sample_documents,
    extension: str,
    loader_name: str,
):
    file_path = tmp_path / f"resume{extension}"
    file_path.touch()

    with patch(
        f"agentic_ai_job_board.ingestion.extractor.{loader_name}"
    ) as loader_mock:
        loader_mock.return_value.load.return_value = sample_documents

        documents = extractor.extract(file_path)

        if extension == ".txt":
            loader_mock.assert_called_once_with(
                str(file_path),
                encoding="utf-8",
            )
        else:
            loader_mock.assert_called_once_with(str(file_path))

    assert len(documents) == 1
    assert documents[0].page_content == (
        "Python Developer with Django experience"
    )
    assert documents[0].metadata["source"] == file_path.name
    assert documents[0].metadata["file_name"] == file_path.name
    assert documents[0].metadata["file_type"] == extension.lstrip(".")


def test_extract_preserves_existing_metadata(
    extractor,
    tmp_path: Path,
):
    file_path = tmp_path / "resume.pdf"
    file_path.touch()

    document = Document(
        page_content="Candidate profile",
        metadata={"page": 2, "author": "John"},
    )

    with patch(
        "agentic_ai_job_board.ingestion.extractor.PyMuPDFLoader"
    ) as loader_mock:
        loader_mock.return_value.load.return_value = [document]

        documents = extractor.extract(file_path)

    assert documents[0].metadata["page"] == 2
    assert documents[0].metadata["author"] == "John"
    assert documents[0].metadata["source"] == "resume.pdf"


def test_extract_raises_error_for_missing_file(
    extractor,
    tmp_path: Path,
):
    file_path = tmp_path / "missing.pdf"

    with pytest.raises(FileNotFoundError, match="Document not found"):
        extractor.extract(file_path)


def test_extract_raises_error_for_unsupported_extension(
    extractor,
    tmp_path: Path,
):
    file_path = tmp_path / "resume.csv"
    file_path.touch()

    with pytest.raises(
        ValueError,
        match="Unsupported document format",
    ):
        extractor.extract(file_path)


def test_extract_raises_error_when_loader_returns_no_documents(
    extractor,
    tmp_path: Path,
):
    file_path = tmp_path / "resume.pdf"
    file_path.touch()

    with patch(
        "agentic_ai_job_board.ingestion.extractor.PyMuPDFLoader"
    ) as loader_mock:
        loader_mock.return_value.load.return_value = []

        with pytest.raises(
            ValueError,
            match="No content extracted",
        ):
            extractor.extract(file_path)


def test_extract_raises_error_for_empty_extracted_content(
    extractor,
    tmp_path: Path,
):
    file_path = tmp_path / "resume.pdf"
    file_path.touch()

    empty_document = Document(page_content="   ")

    with patch(
        "agentic_ai_job_board.ingestion.extractor.PyMuPDFLoader"
    ) as loader_mock:
        loader_mock.return_value.load.return_value = [empty_document]

        with pytest.raises(
            ValueError,
            match="Document contains no extractable text",
        ):
            extractor.extract(file_path)


def test_extract_propagates_loader_exception(
    extractor,
    tmp_path: Path,
):
    file_path = tmp_path / "resume.pdf"
    file_path.touch()

    with patch(
        "agentic_ai_job_board.ingestion.extractor.PyMuPDFLoader"
    ) as loader_mock:
        loader_mock.return_value.load.side_effect = RuntimeError(
            "PDF extraction failed"
        )

        with pytest.raises(
            RuntimeError,
            match="PDF extraction failed",
        ):
            extractor.extract(file_path)
