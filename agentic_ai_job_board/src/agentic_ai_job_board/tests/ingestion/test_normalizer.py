import pytest
from langchain_core.documents import Document

from agentic_ai_job_board.ingestion.normalizer import DocumentNormalizer


@pytest.fixture
def normalizer():
    return DocumentNormalizer()


def test_normalize_cleans_whitespace(normalizer):
    documents = [
        Document(page_content="  Python   Developer  \r\n\r\n\r\n Django ")
    ]

    result = normalizer.normalize(documents)

    assert result[0].page_content == "Python Developer\n\nDjango"


def test_normalize_preserves_headings_and_bullets(normalizer):
    documents = [
        Document(
            page_content=(
                " EXPERIENCE \n"
                "• Python Developer\n"
                "• Built REST APIs\n"
                " SKILLS \n"
                "Python, Django, PostgreSQL"
            )
        )
    ]

    result = normalizer.normalize(documents)

    assert "EXPERIENCE" in result[0].page_content
    assert "• Python Developer" in result[0].page_content
    assert "• Built REST APIs" in result[0].page_content
    assert "SKILLS" in result[0].page_content
    assert "Python, Django, PostgreSQL" in result[0].page_content


def test_normalize_preserves_metadata(normalizer):
    documents = [
        Document(
            page_content="Python Developer",
            metadata={"source": "resume.pdf", "page": 0},
        )
    ]

    result = normalizer.normalize(documents)

    assert result[0].metadata == {
        "source": "resume.pdf",
        "page": 0,
    }
    assert result[0] is not documents[0]


def test_normalize_rejects_empty_document_list(normalizer):
    with pytest.raises(ValueError, match="No documents provided"):
        normalizer.normalize([])


def test_normalize_rejects_blank_content(normalizer):
    documents = [Document(page_content=" \n \t ")]

    with pytest.raises(
        ValueError,
        match="no content after normalization",
    ):
        normalizer.normalize(documents)


def test_normalize_does_not_modify_original_document(normalizer):
    original = Document(
        page_content="  Python   Developer  ",
        metadata={"source": "resume.txt"},
    )

    result = normalizer.normalize([original])

    assert original.page_content == "  Python   Developer  "
    assert result[0].page_content == "Python Developer"
