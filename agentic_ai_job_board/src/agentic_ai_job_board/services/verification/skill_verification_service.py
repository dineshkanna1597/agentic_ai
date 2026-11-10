
import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SkillEvidence:
    skill: str
    explicitly_present: bool
    evidence: list[str]


@dataclass(frozen=True)
class SkillVerificationResult:
    requested_skills: list[str]
    skill_evidence: list[SkillEvidence]
    is_full_match: bool
    missing_skills: list[str]


@dataclass(frozen=True)
class DocumentSkillVerification:
    document_id: str | None
    source: str | None
    document_type: str | None
    verification: SkillVerificationResult


class SkillVerificationService:
    """Verify requested skills using explicit document evidence."""

    @staticmethod
    def _contains_exact_skill(text: str, skill: str) -> bool:
        """Match a skill without accepting known compound alternatives."""

        normalized_text = re.sub(r"\s+", " ", text).casefold()
        normalized_skill = re.sub(r"\s+", " ", skill).strip().casefold()

        excluded_compounds = {
            "django": (r"\s+rest\s+framework",),
        }

        exclusions = excluded_compounds.get(normalized_skill, ())
        exclusion_pattern = "|".join(
            f"(?:{pattern})" for pattern in exclusions
        )

        if exclusion_pattern:
            pattern = (
                rf"(?<!\w){re.escape(normalized_skill)}"
                rf"(?!{exclusion_pattern})"
                rf"(?!\w)"
            )
        else:
            pattern = rf"(?<!\w){re.escape(normalized_skill)}(?!\w)"

        return re.search(pattern, normalized_text) is not None

    def verify(
        self,
        document_text: str,
        requested_skills: list[str],
    ) -> SkillVerificationResult:
        """Verify every requested skill independently."""

        skill_evidence = []

        for skill in requested_skills:
            present = self._contains_exact_skill(document_text, skill)

            skill_evidence.append(
                SkillEvidence(
                    skill=skill,
                    explicitly_present=present,
                    evidence=[skill] if present else [],
                )
            )

        missing_skills = [
            item.skill
            for item in skill_evidence
            if not item.explicitly_present
        ]

        return SkillVerificationResult(
            requested_skills=requested_skills,
            skill_evidence=skill_evidence,
            is_full_match=not missing_skills,
            missing_skills=missing_skills,
        )

    def verify_documents(
        self,
        retrieved_chunks: list[dict[str, Any]],
        requested_skills: list[str],
    ) -> list[DocumentSkillVerification]:
        """Aggregate chunks by document and verify each document."""

        documents: dict[str, dict[str, Any]] = {}

        for chunk in retrieved_chunks:
            metadata = chunk.get("metadata") or {}

            document_id = chunk.get("document_id") or metadata.get(
                "document_id"
            )
            source = (
                chunk.get("source")
                or metadata.get("file_name")
                or metadata.get("source")
            )

            if document_id:
                key = f"id:{document_id}"
            elif source:
                key = f"source:{source}"
            else:
                continue

            if key not in documents:
                documents[key] = {
                    "document_id": document_id,
                    "source": source,
                    "document_type": (
                        chunk.get("document_type")
                        or metadata.get("document_type")
                        or metadata.get("type")
                    ),
                    "contents": [],
                }

            content = chunk.get("content", "")

            if content and content not in documents[key]["contents"]:
                documents[key]["contents"].append(content)

        results = []

        for document in documents.values():
            document_text = "\n".join(document["contents"])

            verification = self.verify(
                document_text=document_text,
                requested_skills=requested_skills,
            )

            results.append(
                DocumentSkillVerification(
                    document_id=document["document_id"],
                    source=document["source"],
                    document_type=document["document_type"],
                    verification=verification,
                )
            )

        return results
