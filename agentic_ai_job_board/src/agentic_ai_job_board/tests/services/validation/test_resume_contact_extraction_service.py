
import pytest

from agentic_ai_job_board.services.validation.contact_validation_service import (
    ContactValidationService,
)
from agentic_ai_job_board.services.validation.pii_detection_service import (
    PIIDetectionResult,
)
from agentic_ai_job_board.services.validation.resume_contact_extraction_service import (
    ResumeContactExtractionService,
)


@pytest.fixture
def service():
    return ResumeContactExtractionService(
        contact_validator=ContactValidationService(default_region="IN")
    )


def create_pii_result(
    text: str,
    emails: list[str] | None = None,
    phones: list[str] | None = None,
) -> PIIDetectionResult:
    return PIIDetectionResult(
        original_text=text,
        redacted_text=text,
        pii_detected=bool(emails or phones),
        pii_categories=[],
        email_addresses=emails or [],
        phone_numbers=phones or [],
    )


def test_extracts_candidate_email_and_mobile(service):
    text = (
        "Alex Morgan\n"
        "Email: alex@example.com\n"
        "Phone: +91 98765 43210\n"
        "Skills\n"
        "Python, Django"
    )

    result = service.extract(
        create_pii_result(
            text,
            emails=["alex@example.com"],
            phones=["+91 98765 43210"],
        )
    )

    assert result.email == "alex@example.com"
    assert result.mobile == "+919876543210"
    assert result.has_valid_contact is True
    assert result.is_ambiguous is False


def test_accepts_only_valid_email(service):
    text = "Alex Morgan\nEmail: alex@example.com\nSkills"

    result = service.extract(
        create_pii_result(text, emails=["alex@example.com"])
    )

    assert result.is_email_valid is True
    assert result.is_mobile_valid is False
    assert result.has_valid_contact is True


def test_accepts_only_valid_mobile(service):
    text = "Alex Morgan\nPhone: +91 98765 43210\nSkills"

    result = service.extract(
        create_pii_result(text, phones=["+91 98765 43210"])
    )

    assert result.is_mobile_valid is True
    assert result.is_email_valid is False
    assert result.has_valid_contact is True


def test_does_not_extract_reference_contacts(service):
    text = (
        "Alex Morgan\n"
        "Skills\n"
        "Python, Django\n"
        "References\n"
        "Email: reference@example.com\n"
        "Phone: +91 98765 43210"
    )

    result = service.extract(
        create_pii_result(
            text,
            emails=["reference@example.com"],
            phones=["+91 98765 43210"],
        )
    )

    assert result.email is None
    assert result.mobile is None
    assert result.has_valid_contact is False


def test_marks_multiple_header_emails_as_ambiguous(service):
    text = (
        "Alex Morgan\n"
        "Email: alex@example.com\n"
        "Alternate: alex.work@example.com\n"
        "Skills"
    )

    result = service.extract(
        create_pii_result(
            text,
            emails=["alex@example.com", "alex.work@example.com"],
        )
    )

    assert result.is_ambiguous is True
    assert result.email is None
    assert result.has_valid_contact is False


def test_returns_no_contact_when_none_exists(service):
    text = "Alex Morgan\nSkills\nPython, Django"

    result = service.extract(create_pii_result(text))

    assert result.email is None
    assert result.mobile is None
    assert result.has_valid_contact is False
    assert result.is_ambiguous is False
