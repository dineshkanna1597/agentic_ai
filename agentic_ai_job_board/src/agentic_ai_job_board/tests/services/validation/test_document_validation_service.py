import pytest

from agentic_ai_job_board.agents.document_validation.schemas import (
    DocumentClassification,
    DocumentType,
    ValidationStatus,
    ResumeValidationData,
    JobPostingValidationData,
)
from agentic_ai_job_board.services.validation.document_validation_service import (
    DocumentValidationService,
)
from agentic_ai_job_board.services.validation.resume_contact_extraction_service import (
    ResumeContactExtractionResult,
)

@pytest.fixture
def service():
    return DocumentValidationService()


def test_valid_resume_is_accepted(service):
    classification = DocumentClassification(
        document_type=DocumentType.RESUME,
        confidence=0.95,
        reasoning="Professional resume with candidate details.",
        detected_sections=["skills", "projects"],
    )

    data = ResumeValidationData(
        candidate_name="Alex Morgan",
        email="alex@example.com",
        skills=["Python", "Django"],
        projects=["Job matching application"],
    )

    result = service.validate_resume(classification, data)

    assert result.status == ValidationStatus.ACCEPTED
    assert result.errors == []


def test_fresh_graduate_resume_is_accepted(service):
    classification = DocumentClassification(
        document_type=DocumentType.RESUME,
        confidence=0.95,
        reasoning="Graduate resume with education and projects.",
    )

    data = ResumeValidationData(
        candidate_name="Alex Morgan",
        mobile="+1 415 555 2671",
        education=["Bachelor of Computer Science"],
        projects=["Student management system"],
    )

    result = service.validate_resume(classification, data)

    assert result.status == ValidationStatus.ACCEPTED


def test_resume_without_valid_contact_is_rejected(service):
    classification = DocumentClassification(
        document_type=DocumentType.RESUME,
        confidence=0.95,
        reasoning="Resume document.",
    )

    data = ResumeValidationData(
        candidate_name="Alex Morgan",
        email="invalid-email",
        skills=["Python"],
    )

    result = service.validate_resume(classification, data)

    assert result.status == ValidationStatus.REJECTED
    assert "At least one valid candidate email address or mobile number is required." in result.errors


def test_resume_without_meaningful_content_is_rejected(service):
    classification = DocumentClassification(
        document_type=DocumentType.RESUME,
        confidence=0.95,
        reasoning="Resume document.",
    )

    data = ResumeValidationData(
        candidate_name="Alex Morgan",
        email="alex@example.com",
    )

    result = service.validate_resume(classification, data)

    assert result.status == ValidationStatus.REJECTED


def test_valid_job_posting_is_accepted(service):
    classification = DocumentClassification(
        document_type=DocumentType.JOB_POSTING,
        confidence=0.95,
        reasoning="Software developer job posting.",
    )

    data = JobPostingValidationData(
        company_name="Example Technologies",
        job_title="Python Developer",
        responsibilities=["Build and maintain backend APIs."],
        required_skills=["Python", "Django"],
        location="Chennai, India",
    )

    result = service.validate_job_posting(classification, data)

    assert result.status == ValidationStatus.ACCEPTED


def test_remote_job_posting_without_location_is_accepted(service):
    classification = DocumentClassification(
        document_type=DocumentType.JOB_POSTING,
        confidence=0.95,
        reasoning="Remote job posting.",
    )

    data = JobPostingValidationData(
        company_name="Example Technologies",
        job_title="Python Developer",
        responsibilities=["Develop backend services."],
        required_skills=["Python"],
        is_remote=True,
    )

    result = service.validate_job_posting(classification, data)

    assert result.status == ValidationStatus.ACCEPTED


def test_job_posting_without_required_skills_is_rejected(service):
    classification = DocumentClassification(
        document_type=DocumentType.JOB_POSTING,
        confidence=0.95,
        reasoning="Job posting.",
    )

    data = JobPostingValidationData(
        company_name="Example Technologies",
        job_title="Python Developer",
        responsibilities=["Develop backend services."],
        location="Chennai, India",
    )

    result = service.validate_job_posting(classification, data)

    assert result.status == ValidationStatus.REJECTED


def test_low_confidence_classification_needs_review(service):
    classification = DocumentClassification(
        document_type=DocumentType.RESUME,
        confidence=0.55,
        reasoning="Document classification is uncertain.",
    )

    data = ResumeValidationData(
        candidate_name="Alex Morgan",
        email="alex@example.com",
        skills=["Python"],
    )

    result = service.validate_resume(classification, data)

    assert result.status == ValidationStatus.NEEDS_REVIEW
    assert result.warnings


def test_wrong_document_type_is_rejected(service):
    classification = DocumentClassification(
        document_type=DocumentType.JOB_POSTING,
        confidence=0.95,
        reasoning="This is a job posting, not a resume.",
    )

    data = ResumeValidationData(
        candidate_name="Alex Morgan",
        email="alex@example.com",
        skills=["Python"],
    )

    result = service.validate_resume(classification, data)

    assert result.status == ValidationStatus.REJECTED


def test_resume_with_locally_validated_email_is_accepted(service):
    classification = DocumentClassification(
        document_type=DocumentType.RESUME,
        confidence=0.95,
        reasoning="Resume with candidate details.",
    )

    data = ResumeValidationData(
        candidate_name="Alex Morgan",
        # LLM receives redacted text, so no email is extracted.
        email=None,
        skills=["Python", "Django"],
    )

    contact_result = ResumeContactExtractionResult(
        email="alex@example.com",
        mobile=None,
        is_email_valid=True,
        is_mobile_valid=False,
        has_valid_contact=True,
        is_ambiguous=False,
    )

    result = service.validate_resume(
        classification,
        data,
        contact_result=contact_result,
    )

    assert result.status == ValidationStatus.ACCEPTED


def test_resume_with_locally_validated_mobile_is_accepted(service):
    classification = DocumentClassification(
        document_type=DocumentType.RESUME,
        confidence=0.95,
        reasoning="Resume with candidate details.",
    )

    data = ResumeValidationData(
        candidate_name="Alex Morgan",
        skills=["Python"],
    )

    contact_result = ResumeContactExtractionResult(
        email=None,
        mobile="+919876543210",
        is_email_valid=False,
        is_mobile_valid=True,
        has_valid_contact=True,
        is_ambiguous=False,
    )

    result = service.validate_resume(
        classification,
        data,
        contact_result=contact_result,
    )

    assert result.status == ValidationStatus.ACCEPTED


def test_resume_without_locally_validated_contact_is_rejected(service):
    classification = DocumentClassification(
        document_type=DocumentType.RESUME,
        confidence=0.95,
        reasoning="Resume with candidate details.",
    )

    data = ResumeValidationData(
        candidate_name="Alex Morgan",
        skills=["Python"],
    )

    contact_result = ResumeContactExtractionResult(
        email=None,
        mobile=None,
        is_email_valid=False,
        is_mobile_valid=False,
        has_valid_contact=False,
        is_ambiguous=False,
    )

    result = service.validate_resume(
        classification,
        data,
        contact_result=contact_result,
    )

    assert result.status == ValidationStatus.REJECTED
    assert any(
        check.name == "contact_information" and not check.passed
        for check in result.checks
    )


def test_resume_with_ambiguous_contacts_needs_review(service):
    classification = DocumentClassification(
        document_type=DocumentType.RESUME,
        confidence=0.95,
        reasoning="Resume with candidate details.",
    )

    data = ResumeValidationData(
        candidate_name="Alex Morgan",
        skills=["Python"],
    )

    contact_result = ResumeContactExtractionResult(
        email=None,
        mobile=None,
        is_email_valid=False,
        is_mobile_valid=False,
        has_valid_contact=False,
        is_ambiguous=True,
    )

    result = service.validate_resume(
        classification,
        data,
        contact_result=contact_result,
    )

    # Ambiguous contacts fail the required contact check.
    assert result.status == ValidationStatus.NEEDS_REVIEW
    assert any(
        "Multiple candidate contact details" in warning
        for warning in result.warnings
    )
