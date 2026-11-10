import re
from dataclasses import dataclass

import phonenumbers
from email_validator import EmailNotValidError, validate_email


@dataclass(frozen=True, slots=True)
class ContactValidationResult:
    """Result of email and phone validation."""

    email: str | None
    is_email_valid: bool

    mobile: str | None
    is_mobile_valid: bool

    has_valid_contact: bool


class ContactValidationService:
    """Validate email addresses and international phone numbers."""

    def __init__(self, default_region: str | None = None) -> None:
        self.default_region = default_region

    def validate(
        self,
        email: str | None = None,
        mobile: str | None = None,
    ) -> ContactValidationResult:
        normalized_email = self._validate_email(email)
        normalized_mobile = self._validate_mobile(mobile)

        is_email_valid = normalized_email is not None
        is_mobile_valid = normalized_mobile is not None

        return ContactValidationResult(
            email=normalized_email,
            is_email_valid=is_email_valid,
            mobile=normalized_mobile,
            is_mobile_valid=is_mobile_valid,
            has_valid_contact=is_email_valid or is_mobile_valid,
        )

    @staticmethod
    def _validate_email(email: str | None) -> str | None:
        if not email or not email.strip():
            return None

        try:
            result = validate_email(
                email.strip(),
                check_deliverability=False,
            )
            return result.normalized
        except EmailNotValidError:
            return None

    def _validate_mobile(self, mobile: str | None) -> str | None:
        if not mobile or not mobile.strip():
            return None

        # Remove common visual separators, but retain the leading +.
        cleaned = re.sub(r"[\s().-]", "", mobile.strip())

        try:
            parsed = phonenumbers.parse(
                cleaned,
                self.default_region,
            )

            if not phonenumbers.is_valid_number(parsed):
                return None

            return phonenumbers.format_number(
                parsed,
                phonenumbers.PhoneNumberFormat.E164,
            )
        except phonenumbers.NumberParseException:
            return None