
import re


class SkillExtractionService:
    """Extract explicitly requested skills from user queries."""

    SKILL_TAXONOMY = [
        "Django REST Framework",
        "FastAPI",
        "PostgreSQL",
        "MySQL",
        "JavaScript",
        "TypeScript",
        "Next.js",
        "React",
        "Python",
        "Django",
        "Flask",
        "Celery",
        "Redis",
        "Qdrant",
        "LangChain",
        "LangGraph",
        "Docker",
        "Kubernetes",
        "REST API",
        "SQL",
    ]

    def extract(self, query: str) -> list[str]:
        """Extract skills without overlapping phrase matches."""

        matches = []

        # Match longer skill names before shorter ones.
        skills = sorted(
            self.SKILL_TAXONOMY,
            key=len,
            reverse=True,
        )

        occupied_spans = []

        for skill in skills:
            pattern = rf"(?<!\w){re.escape(skill)}(?!\w)"

            for match in re.finditer(
                pattern,
                query,
                flags=re.IGNORECASE,
            ):
                start, end = match.span()

                overlaps = any(
                    start < occupied_end and end > occupied_start
                    for occupied_start, occupied_end in occupied_spans
                )

                if overlaps:
                    continue

                occupied_spans.append((start, end))
                matches.append((start, skill))

        # Preserve the order in which skills appear in the query.
        matches.sort(key=lambda item: item[0])

        return [skill for _, skill in matches]
