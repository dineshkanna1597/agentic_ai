
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


def require_all_json_schema_fields(schema: dict) -> None:
    """Make every model property required in the generated JSON schema."""
    schema["required"] = list(schema.get("properties", {}).keys())


class ChunkingStrategy(StrEnum):
    SECTION_BASED = "section_based"
    RECURSIVE = "recursive"


class ChunkingPlan(BaseModel):
    """LLM-generated plan for splitting a document into chunks."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra=require_all_json_schema_fields,
    )

    strategy: ChunkingStrategy = Field(
        description=(
            "Chunking strategy selected based on the document structure. "
            "Use section_based when meaningful document sections are identifiable; "
            "otherwise use recursive."
        )
    )

    chunk_size: int = Field(
        ge=100,
        le=1000,
        description="Target chunk size measured in tokens.",
    )

    chunk_overlap: int = Field(
        ge=0,
        le=200,
        description="Overlap between adjacent chunks, measured in tokens.",
    )

    section_headers: list[str] = Field(
        default_factory=list,
        description=(
            "Identified section headings, such as Skills, Experience, "
            "Education, Responsibilities, and Qualifications."
        ),
    )

    reasoning: str = Field(
        description="Explanation for selecting this chunking strategy and configuration."
    )