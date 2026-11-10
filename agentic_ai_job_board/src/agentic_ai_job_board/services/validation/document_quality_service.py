from agentic_ai_job_board.agents.document_quality.schemas import (
    DocumentQualityExtraction,
    DocumentQualityResult,
    QualityStatus,
)
from agentic_ai_job_board.agents.document_validation.schemas import (
    DocumentType,
)


class DocumentQualityService:
    """Apply deterministic rules to document quality assessments."""

    ACCEPTED_THRESHOLD = 70
    NEEDS_REVIEW_THRESHOLD = 40

    def assess(
        self,
        extraction: DocumentQualityExtraction,
        expected_document_type: DocumentType,
    ) -> DocumentQualityResult:
        """Calculate overall quality and determine the final status."""

        if extraction.document_type != expected_document_type:
            raise ValueError(
                f"Expected {expected_document_type.value}, "
                f"received {extraction.document_type.value}."
            )

        if not extraction.dimensions:
            raise ValueError(
                "At least one quality dimension is required."
            )

        overall_score = round(
            sum(dimension.score for dimension in extraction.dimensions)
            / len(extraction.dimensions)
        )

        status = self._determine_status(overall_score)

        return DocumentQualityResult(
            document_type=extraction.document_type,
            overall_score=overall_score,
            status=status,
            dimensions=extraction.dimensions,
            strengths=extraction.strengths,
            issues=extraction.issues,
            suggestions=extraction.suggestions,
        )

    def _determine_status(self, score: int) -> QualityStatus:
        if score >= self.ACCEPTED_THRESHOLD:
            return QualityStatus.ACCEPTED

        if score >= self.NEEDS_REVIEW_THRESHOLD:
            return QualityStatus.NEEDS_REVIEW

        return QualityStatus.REJECTED