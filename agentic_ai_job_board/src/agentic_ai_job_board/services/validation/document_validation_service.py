
from agentic_ai_job_board.agents.document_validation.schemas import (
    DocumentClassification,
    DocumentType,
    DocumentValidationResult,
    JobPostingValidationData,
    ResumeValidationData,
    ValidationCheck,
    ValidationStatus,
)
from agentic_ai_job_board.services.validation.contact_validation_service import (
    ContactValidationService,
)
from agentic_ai_job_board.services.validation.resume_contact_extraction_service import (
    ResumeContactExtractionResult,
)

class DocumentValidationService:
    """Apply deterministic validation rules to classified documents."""

    MIN_CLASSIFICATION_CONFIDENCE = 0.70

    def __init__(
        self,
        contact_validator: ContactValidationService | None = None,
    ) -> None:
        self.contact_validator = (
            contact_validator or ContactValidationService()
        )

    def validate_resume(
        self,
        classification: DocumentClassification,
        data: ResumeValidationData,
        contact_result: ResumeContactExtractionResult | None = None,
    ) -> DocumentValidationResult:
        checks: list[ValidationCheck] = []

        self._check_classification(
            classification,
            DocumentType.RESUME,
            checks,
        )

        self._add_check(
            checks,
            name="candidate_name",
            passed=bool(data.candidate_name and data.candidate_name.strip()),
            required=True,
            message="Candidate name is required.",
        )

        if contact_result is not None:
            has_valid_contact = contact_result.has_valid_contact
        else:
            # Backward compatibility for existing callers and tests.
            contact = self.contact_validator.validate(
                email=data.email,
                mobile=data.mobile,
        )
            has_valid_contact = contact.has_valid_contact

        self._add_check(
            checks,
            name="contact_information",
            passed=has_valid_contact,
            required=True,
            message=(
                "At least one valid candidate email address or mobile number "
                "is required."
        ),
    )

        has_meaningful_content = any(
            [
                data.skills,
                data.work_experience,
                data.projects,
                data.education,
                data.internships,
                data.certifications,
            ]
        )

        self._add_check(
            checks,
            name="meaningful_content",
            passed=has_meaningful_content,
            required=True,
            message=(
                "At least one meaningful section is required: skills, "
                "experience, projects, education, internships, or certifications."
            ),
        )

        result = self._build_result(
            expected_type=DocumentType.RESUME,
            classification=classification,
            checks=checks,
        )

        if contact_result is not None and contact_result.is_ambiguous:
            result.status = ValidationStatus.NEEDS_REVIEW
            result.warnings.append(
                "Multiple candidate contact details were found in the resume header. "
                "Manual verification is required."
        )

        return result

    def validate_job_posting(
        self,
        classification: DocumentClassification,
        data: JobPostingValidationData,
    ) -> DocumentValidationResult:
        checks: list[ValidationCheck] = []

        self._check_classification(
            classification,
            DocumentType.JOB_POSTING,
            checks,
        )

        self._add_check(
            checks,
            name="company_name",
            passed=bool(data.company_name and data.company_name.strip()),
            required=True,
            message="Company name is required.",
        )

        self._add_check(
            checks,
            name="job_title",
            passed=bool(data.job_title and data.job_title.strip()),
            required=True,
            message="Job title is required.",
        )

        self._add_check(
            checks,
            name="responsibilities",
            passed=bool(data.responsibilities),
            required=True,
            message="Meaningful job responsibilities are required.",
        )

        has_skills = bool(
            data.required_skills or data.preferred_skills
        )

        self._add_check(
            checks,
            name="skills",
            passed=has_skills,
            required=True,
            message="At least one required or preferred skill is required.",
        )

        has_location = bool(
            data.location and data.location.strip()
        ) or data.is_remote

        self._add_check(
            checks,
            name="location",
            passed=has_location,
            required=True,
            message="A location or explicit remote designation is required.",
        )

        return self._build_result(
            expected_type=DocumentType.JOB_POSTING,
            classification=classification,
            checks=checks,
        )

    @staticmethod
    def _add_check(
        checks: list[ValidationCheck],
        name: str,
        passed: bool,
        required: bool,
        message: str,
    ) -> None:
        checks.append(
            ValidationCheck(
                name=name,
                passed=passed,
                required=required,
                message=message,
            )
        )

    def _check_classification(
        self,
        classification: DocumentClassification,
        expected_type: DocumentType,
        checks: list[ValidationCheck],
    ) -> None:
        self._add_check(
            checks,
            name="document_classification",
            passed=classification.document_type == expected_type,
            required=True,
            message=(
                f"Expected {expected_type.value}, "
                f"received {classification.document_type.value}."
            ),
        )

    def _build_result(
        self,
        expected_type: DocumentType,
        classification: DocumentClassification,
        checks: list[ValidationCheck],
    ) -> DocumentValidationResult:
        errors = [
            check.message
            for check in checks
            if check.required and not check.passed
        ]

        warnings: list[str] = []

        if classification.confidence < self.MIN_CLASSIFICATION_CONFIDENCE:
            warnings.append(
                "Document classification confidence is below 0.70."
            )

        if classification.document_type == DocumentType.UNKNOWN:
            warnings.append(
                "The document could not be confidently classified."
            )

        if warnings:
            status = ValidationStatus.NEEDS_REVIEW
        elif errors:
            status = ValidationStatus.REJECTED
        else:
            status = ValidationStatus.ACCEPTED

        return DocumentValidationResult(
            expected_document_type=expected_type,
            actual_document_type=classification.document_type,
            status=status,
            checks=checks,
            warnings=warnings,
            errors=errors,
        )
