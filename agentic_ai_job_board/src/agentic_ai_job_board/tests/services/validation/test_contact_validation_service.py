
import pytest

from agentic_ai_job_board.services.validation.contact_validation_service import (
    ContactValidationService,
)


@pytest.fixture
def service():
    return ContactValidationService(default_region="IN")


def test_valid_email(service):
    result = service.validate(email="candidate@example.com")

    assert result.is_email_valid is True
    assert result.has_valid_contact is True
    assert result.email == "candidate@example.com"


def test_valid_indian_mobile(service):
    result = service.validate(mobile="+91 98765 43210")

    assert result.is_mobile_valid is True
    assert result.has_valid_contact is True
    assert result.mobile == "+919876543210"


def test_email_and_mobile_are_both_valid(service):
    result = service.validate(
        email="candidate@example.com",
        mobile="+91 98765 43210",
    )

    assert result.is_email_valid is True
    assert result.is_mobile_valid is True
    assert result.has_valid_contact is True


def test_invalid_email_and_mobile(service):
    result = service.validate(
        email="invalid-email",
        mobile="123",
    )

    assert result.is_email_valid is False
    assert result.is_mobile_valid is False
    assert result.has_valid_contact is False


def test_missing_contact_information(service):
    result = service.validate()

    assert result.email is None
    assert result.mobile is None
    assert result.has_valid_contact is False


def test_invalid_email_with_valid_mobile(service):
    result = service.validate(
        email="invalid-email",
        mobile="+91 98765 43210",
    )

    assert result.is_email_valid is False
    assert result.is_mobile_valid is True
    assert result.has_valid_contact is True


def test_valid_email_with_invalid_mobile(service):
    result = service.validate(
        email="candidate@example.com",
        mobile="123",
    )

    assert result.is_email_valid is True
    assert result.is_mobile_valid is False
    assert result.has_valid_contact is True