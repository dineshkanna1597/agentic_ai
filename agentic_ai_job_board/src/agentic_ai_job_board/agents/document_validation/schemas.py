
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

def require_all_json_schema_fields(schema: dict) -> None:
    """Make every model property required in the generated JSON schema."""
    schema["required"] = list(schema.get("properties", {}).keys())

class DocumentType(str, Enum):
    """Supported document types."""

    RESUME = "resume"
    JOB_POSTING = "job_posting"
    UNKNOWN = "unknown"


class ValidationStatus(str, Enum):
    """Overall document validation status."""

    ACCEPTED = "accepted"
    REJECTED = "rejected"
    NEEDS_REVIEW = "needs_review"


class DocumentClassification(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra=require_all_json_schema_fields,
    )

    document_type: DocumentType
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(min_length=1)
    detected_sections: list[str] = Field(default_factory=list)

class ValidationCheck(BaseModel):
    """Result of an individual validation check."""

    model_config = ConfigDict(extra="forbid")

    name: str
    passed: bool
    required: bool

    message: str


class DocumentValidationResult(BaseModel):
    """Final validation result after all validation checks."""

    model_config = ConfigDict(extra="forbid")

    expected_document_type: DocumentType
    actual_document_type: DocumentType

    status: ValidationStatus

    checks: list[ValidationCheck] = Field(default_factory=list)

    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class ResumeValidationData(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra=require_all_json_schema_fields,
    )

    candidate_name: str | None = None
    email: str | None = None
    mobile: str | None = None
    skills: list[str] = Field(default_factory=list)
    work_experience: list[str] = Field(default_factory=list)
    projects: list[str] = Field(default_factory=list)
    education: list[str] = Field(default_factory=list)
    internships: list[str] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)

class JobPostingValidationData(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra=require_all_json_schema_fields,
    )

    company_name: str | None = None
    job_title: str | None = None
    responsibilities: list[str] = Field(default_factory=list)
    required_skills: list[str] = Field(default_factory=list)
    preferred_skills: list[str] = Field(default_factory=list)
    location: str | None = None
    is_remote: bool = False


class DocumentValidationExtraction(BaseModel):
    """Combined classification and extracted document data."""

    model_config = ConfigDict(extra="forbid")

    classification: DocumentClassification
    resume_data: ResumeValidationData | None
    job_posting_data: JobPostingValidationData | None