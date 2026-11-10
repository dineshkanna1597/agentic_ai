
import re
from dataclasses import dataclass

from agentic_ai_job_board.services.validation.contact_validation_service import (
    ContactValidationService,
)
from agentic_ai_job_board.services.validation.pii_detection_service import (
    PIIDetectionResult,
)


@dataclass(frozen=True, slots=True)
class ResumeContactExtractionResult:
    email: str | None
    mobile: str | None
    is_email_valid: bool
    is_mobile_valid: bool
    has_valid_contact: bool
    is_ambiguous: bool


class ResumeContactExtractionService:
    """Extract and validate likely candidate contact details locally."""

    SECTION_HEADINGS = {
        "summary",
        "objective",
        "profile",
        "skills",
        "technical skills",
        "work experience",
        "professional experience",
        "experience",
        "education",
        "projects",
        "internships",
        "certifications",
        "references",
        "referees",
    }

    def __init__(
        self,
        contact_validator: ContactValidationService | None = None,
        header_line_limit: int = 12,
    ) -> None:
        self.contact_validator = (
            contact_validator or ContactValidationService()
        )
        self.header_line_limit = header_line_limit

    def extract(
        self,
        pii_result: PIIDetectionResult,
    ) -> ResumeContactExtractionResult:
        """Extract candidate contacts from the resume header."""

        header = self._get_header(pii_result.original_text)

        emails = self._find_candidates(
            header,
            pii_result.email_addresses,
        )
        mobiles = self._find_candidates(
            header,
            pii_result.phone_numbers,
        )

        is_ambiguous = len(emails) > 1 or len(mobiles) > 1

        # Do not guess when multiple candidate contacts are present.
        email = emails[0] if len(emails) == 1 else None
        mobile = mobiles[0] if len(mobiles) == 1 else None

        validation = self.contact_validator.validate(
            email=email,
            mobile=mobile,
        )

        return ResumeContactExtractionResult(
            email=validation.email,
            mobile=validation.mobile,
            is_email_valid=validation.is_email_valid,
            is_mobile_valid=validation.is_mobile_valid,
            has_valid_contact=(
                validation.has_valid_contact and not is_ambiguous
            ),
            is_ambiguous=is_ambiguous,
        )

    def _get_header(self, text: str) -> str:
        """Return the content before the first recognized resume section."""

        header_lines = []

        for line in text.splitlines()[: self.header_line_limit]:
            normalized = re.sub(
                r"[^a-z ]",
                "",
                line.strip().lower(),
            ).strip()

            if normalized in self.SECTION_HEADINGS:
                break

            header_lines.append(line)

        return "\n".join(header_lines)

    @staticmethod
    def _find_candidates(
        header: str,
        detected_values: list[str],
    ) -> list[str]:
        """Keep only detected values that appear in the resume header."""

        candidates = []

        for value in detected_values:
            if re.search(re.escape(value), header, re.IGNORECASE):
                if value not in candidates:
                    candidates.append(value)

        return candidates
