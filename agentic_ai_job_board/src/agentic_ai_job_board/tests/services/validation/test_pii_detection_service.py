from agentic_ai_job_board.services.validation.pii_detection_service import (
    PIIDetectionService,
)


def test_detects_and_redacts_personal_information() -> None:
    service = PIIDetectionService()
    text = (
        "Alex Morgan\n"
        "Email: alex@example.com\n"
        "Phone: +1 415 555 2671\n"
        "Skills: Python, Django, PostgreSQL"
    )

    result = service.detect_and_redact(text)

    assert result.pii_detected is True

    assert "EMAIL_ADDRESS" in result.pii_categories
    assert "PHONE_NUMBER" in result.pii_categories
    assert "PERSON" not in result.pii_categories

    assert "alex@example.com" in result.email_addresses
    assert "+1 415 555 2671" in result.phone_numbers
    assert "alex@example.com" not in result.redacted_text
    assert "+1 415 555 2671" not in result.redacted_text

    assert "Alex Morgan" in result.redacted_text
    assert "Python" in result.redacted_text
    assert "Django" in result.redacted_text
    assert "PostgreSQL" in result.redacted_text


def test_preserves_technical_skills() -> None:
    service = PIIDetectionService()
    text = "Skills: Python, Django, PostgreSQL"

    result = service.detect_and_redact(text)

    assert result.redacted_text == text
    assert result.pii_detected is False


def test_empty_text_returns_without_pii() -> None:
    service = PIIDetectionService()

    result = service.detect_and_redact("   ")

    assert result.pii_detected is False
    assert result.pii_categories == []
    assert result.redacted_text == "   "


def test_original_text_is_preserved() -> None:
    service = PIIDetectionService()
    text = "Alex Morgan\nEmail: alex@example.com"

    result = service.detect_and_redact(text)

    assert result.original_text == text
    assert "alex@example.com" not in result.redacted_text