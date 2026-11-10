from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path


class FileEventType(StrEnum):
    ADDED = "added"
    MODIFIED = "modified"
    DELETED = "deleted"


class FileSource(StrEnum):
    RESUME = "resume"
    POSTING = "posting"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class FileEvent:
    """Represents a filesystem event."""

    event_type: FileEventType
    source: FileSource
    file_path: Path
    timestamp: datetime

    @classmethod
    def create(
        cls,
        event_type: FileEventType,
        source: FileSource,
        file_path: Path,
    ) -> "FileEvent":
        return cls(
            event_type=event_type,
            source=source,
            file_path=file_path,
            timestamp=datetime.now(timezone.utc),
        )
