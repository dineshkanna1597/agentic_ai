import pytest
from pydantic import ValidationError

from agentic_ai_job_board.agents.document_quality.schemas import (
    DocumentQualityExtraction,
    DocumentQualityResult,
    QualityDimension,
    QualityDimensionName,
    QualityStatus,
)
from agentic_ai_job_board.agents.document_validation.schemas import (
    DocumentType,
)


@pytest.fixture
def quality_dimension():
    return QualityDimension(
        name=QualityDimensionName.SKILLS_CLARITY,
        score=85,
        feedback="Skills are clearly listed.",
        issues=[],
        suggestions=["Group skills by category."],
    )


def test_quality_dimension_creation(quality_dimension):
    assert quality_dimension.score == 85
    assert quality_dimension.name == QualityDimensionName.SKILLS_CLARITY


def test_quality_dimension_rejects_invalid_score():
    with pytest.raises(ValidationError):
        QualityDimension(
            name=QualityDimensionName.SKILLS_CLARITY,
            score=110,
            feedback="Invalid score.",
        )


def test_quality_dimension_rejects_extra_fields():
    with pytest.raises(ValidationError):
        QualityDimension(
            name=QualityDimensionName.SKILLS_CLARITY,
            score=85,
            feedback="Good.",
            unexpected_field="invalid",
        )


def test_document_quality_extraction_creation(quality_dimension):
    extraction = DocumentQualityExtraction(
        document_type=DocumentType.RESUME,
        summary="A well-structured resume.",
        dimensions=[quality_dimension],
    )

    assert extraction.document_type == DocumentType.RESUME
    assert len(extraction.dimensions) == 1
    assert extraction.strengths == []
    assert extraction.issues == []
    assert extraction.suggestions == []


def test_document_quality_result_creation(quality_dimension):
    result = DocumentQualityResult(
        document_type=DocumentType.JOB_POSTING,
        overall_score=85,
        status=QualityStatus.ACCEPTED,
        dimensions=[quality_dimension],
    )

    assert result.overall_score == 85
    assert result.status == QualityStatus.ACCEPTED
    assert result.document_type == DocumentType.JOB_POSTING


def test_document_quality_result_rejects_invalid_score():
    with pytest.raises(ValidationError):
        DocumentQualityResult(
            document_type=DocumentType.RESUME,
            overall_score=120,
            status=QualityStatus.ACCEPTED,
            dimensions=[],
        )