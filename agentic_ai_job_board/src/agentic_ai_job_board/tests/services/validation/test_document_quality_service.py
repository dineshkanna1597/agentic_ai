import pytest

from agentic_ai_job_board.agents.document_quality.schemas import (
    DocumentQualityExtraction,
    QualityDimension,
    QualityDimensionName,
    QualityStatus,
)
from agentic_ai_job_board.agents.document_validation.schemas import (
    DocumentType,
)
from agentic_ai_job_board.services.validation.document_quality_service import (
    DocumentQualityService,
)


@pytest.fixture
def quality_service():
    return DocumentQualityService()


@pytest.fixture
def resume_extraction():
    return DocumentQualityExtraction(
        document_type=DocumentType.RESUME,
        summary="A well-structured resume.",
        dimensions=[
            QualityDimension(
                name=QualityDimensionName.SKILLS_CLARITY,
                score=80,
                feedback="Skills are clearly presented.",
            ),
            QualityDimension(
                name=QualityDimensionName.PROJECT_CLARITY,
                score=90,
                feedback="Projects are well explained.",
            ),
        ],
        strengths=["Clear technical skills"],
        issues=[],
        suggestions=["Add measurable achievements"],
    )


def test_assess_returns_accepted_for_high_quality(
    quality_service,
    resume_extraction,
):
    result = quality_service.assess(
        resume_extraction,
        DocumentType.RESUME,
    )

    assert result.overall_score == 85
    assert result.status == QualityStatus.ACCEPTED


def test_assess_returns_needs_review_for_medium_quality(
    quality_service,
    resume_extraction,
):
    resume_extraction.dimensions[0].score = 50
    resume_extraction.dimensions[1].score = 60

    result = quality_service.assess(
        resume_extraction,
        DocumentType.RESUME,
    )

    assert result.overall_score == 55
    assert result.status == QualityStatus.NEEDS_REVIEW


def test_assess_returns_rejected_for_low_quality(
    quality_service,
    resume_extraction,
):
    resume_extraction.dimensions[0].score = 20
    resume_extraction.dimensions[1].score = 30

    result = quality_service.assess(
        resume_extraction,
        DocumentType.RESUME,
    )

    assert result.overall_score == 25
    assert result.status == QualityStatus.REJECTED


def test_assess_rejects_mismatched_document_type(
    quality_service,
    resume_extraction,
):
    with pytest.raises(ValueError, match="Expected"):
        quality_service.assess(
            resume_extraction,
            DocumentType.JOB_POSTING,
        )


def test_assess_rejects_empty_dimensions(quality_service):
    extraction = DocumentQualityExtraction(
        document_type=DocumentType.RESUME,
        summary="Resume assessment.",
        dimensions=[],
    )

    with pytest.raises(
        ValueError,
        match="At least one quality dimension",
    ):
        quality_service.assess(
            extraction,
            DocumentType.RESUME,
        )


def test_assess_preserves_feedback(
    quality_service,
    resume_extraction,
):
    result = quality_service.assess(
        resume_extraction,
        DocumentType.RESUME,
    )

    assert result.strengths == ["Clear technical skills"]
    assert result.issues == []
    assert result.suggestions == ["Add measurable achievements"]