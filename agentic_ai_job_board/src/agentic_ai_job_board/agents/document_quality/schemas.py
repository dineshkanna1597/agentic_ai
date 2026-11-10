from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from agentic_ai_job_board.agents.document_validation.schemas import (
    DocumentType,
)


class QualityStatus(StrEnum):
    ACCEPTED = "accepted"
    NEEDS_REVIEW = "needs_review"
    REJECTED = "rejected"


class QualityDimensionName(StrEnum):
    # Resume dimensions
    SKILLS_CLARITY = "skills_clarity"
    EXPERIENCE_CLARITY = "experience_clarity"
    PROJECT_CLARITY = "project_clarity"
    EDUCATION_COMPLETENESS = "education_completeness"

    # Job posting dimensions
    RESPONSIBILITIES_CLARITY = "responsibilities_clarity"
    SKILLS_SPECIFICITY = "skills_specificity"
    REQUIREMENTS_CLARITY = "requirements_clarity"

    # Common dimensions
    CONTENT_COMPLETENESS = "content_completeness"
    READABILITY = "readability"
    ROLE_RELEVANCE = "role_relevance"


def require_all_json_schema_fields(schema: dict) -> None:
    """Make every model property required in the generated JSON schema."""
    schema["required"] = list(schema.get("properties", {}).keys())


class QualityDimension(BaseModel):
    """Quality evaluation of an individual document dimension."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra=require_all_json_schema_fields,
    )

    name: QualityDimensionName
    score: int = Field(ge=0, le=100)
    feedback: str
    issues: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)


class DocumentQualityExtraction(BaseModel):
    """Structured quality assessment extracted by the LLM."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra=require_all_json_schema_fields,
    )

    document_type: DocumentType
    summary: str
    dimensions: list[QualityDimension]
    strengths: list[str] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)


class DocumentQualityResult(BaseModel):
    """Final quality assessment produced by the quality service."""

    model_config = ConfigDict(extra="forbid")

    document_type: DocumentType
    overall_score: int = Field(ge=0, le=100)
    status: QualityStatus
    dimensions: list[QualityDimension]
    strengths: list[str] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)