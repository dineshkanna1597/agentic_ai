
from dataclasses import dataclass

from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig


@dataclass(frozen=True, slots=True)
class PIIDetectionResult:
    """PII detection and anonymization result."""

    original_text: str
    redacted_text: str
    pii_detected: bool
    pii_categories: list[str]
    email_addresses: list[str]
    phone_numbers: list[str]


class PIIDetectionService:
    """Detect and redact relevant PII in documents."""

    ENTITIES = [
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    ]

    # Prevent common technical terms from being treated as person names.
    TECHNICAL_TERMS = {
        "python",
        "django",
        "postgresql",
        "javascript",
        "typescript",
        "fastapi",
        "react",
        "next.js",
        "sql",
        "langchain",
        "langgraph",
        "qdrant",
        "redis",
        "celery",
        "docker",
        "kubernetes",
        "aws",
        "azure",
        "git",
        "github",
        "pandas",
        "numpy",
        "pytorch",
        "tensorflow",
    }

    def __init__(
        self,
        analyzer: AnalyzerEngine | None = None,
        anonymizer: AnonymizerEngine | None = None,
    ) -> None:
        self.analyzer = analyzer or AnalyzerEngine()
        self.anonymizer = anonymizer or AnonymizerEngine()

    def detect_and_redact(
        self,
        text: str,
        language: str = "en",
    ) -> PIIDetectionResult:
        if not text.strip():
            return PIIDetectionResult(
                original_text=text,
                redacted_text=text,
                pii_detected=False,
                pii_categories=[],
                email_addresses=[],
                phone_numbers=[],
            )

        results = self.analyzer.analyze(
            text=text,
            language=language,
            entities=self.ENTITIES,
        )

        # Remove known technical terms misclassified as person names.
        filtered_results = [
            result
            for result in results
            if not (
                result.entity_type == "PERSON"
                and text[result.start:result.end].strip().casefold()
                in self.TECHNICAL_TERMS
            )
        ]

        email_addresses = [
            text[result.start:result.end]
            for result in filtered_results
            if result.entity_type == "EMAIL_ADDRESS"
        ]

        phone_numbers = [
            text[result.start:result.end]
            for result in filtered_results
            if result.entity_type == "PHONE_NUMBER"
        ]

        operators = {
            result.entity_type: OperatorConfig(
                "replace",
                {"new_value": f"<{result.entity_type}>"},
            )
            for result in filtered_results
        }

        anonymized = self.anonymizer.anonymize(
            text=text,
            analyzer_results=filtered_results,
            operators=operators,
        )

        categories = sorted(
            {result.entity_type for result in filtered_results}
        )

        return PIIDetectionResult(
            original_text=text,
            redacted_text=anonymized.text,
            pii_detected=bool(filtered_results),
            pii_categories=categories,
            email_addresses=email_addresses,
            phone_numbers=phone_numbers,
        )
