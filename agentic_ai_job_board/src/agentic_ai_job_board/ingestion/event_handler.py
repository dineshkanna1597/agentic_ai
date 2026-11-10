from pathlib import Path

from watchfiles import Change

from agentic_ai_job_board.core.config import settings
from agentic_ai_job_board.core.logging import get_logger
from agentic_ai_job_board.ingestion.events import (
    FileEvent,
    FileEventType,
    FileSource,
)
from agentic_ai_job_board.ingestion.file_validator import FileValidator
from agentic_ai_job_board.ingestion.quarantine import (
    QuarantineService,
)
from agentic_ai_job_board.ingestion.dispatcher import (
    IngestionDispatcher,
)

logger = get_logger(__name__)


class FileEventHandler:
    """Handle and validate filesystem events."""

    def __init__(self) -> None:
        self.resume_dir = settings.RESUME_DIR.resolve()
        self.posting_dir = settings.POSTING_DIR.resolve()

        self.validator = FileValidator()
        self.quarantine_service = QuarantineService()
        self.dispatcher = IngestionDispatcher()

    def identify_source(self, file_path: Path) -> FileSource:
        """Identify the document source."""

        resolved_path = file_path.resolve()

        if resolved_path.is_relative_to(self.resume_dir):
            return FileSource.RESUME

        if resolved_path.is_relative_to(self.posting_dir):
            return FileSource.POSTING

        return FileSource.UNKNOWN

    async def handle(
        self,
        change: Change,
        file_path: str,
    ) -> None:
        """Process an individual filesystem event."""

        path = Path(file_path)

        event_mapping = {
            Change.added: FileEventType.ADDED,
            Change.modified: FileEventType.MODIFIED,
            Change.deleted: FileEventType.DELETED,
        }

        event_type = event_mapping.get(change)

        if event_type is None:
            return

        source = self.identify_source(path)

        if source == FileSource.UNKNOWN:
            logger.warning(
                "Ignoring file outside monitored directories: %s",
                path,
            )
            return

        event = FileEvent.create(
            event_type=event_type,
            source=source,
            file_path=path,
        )

        logger.info(
            "Processing event | type=%s | source=%s | file=%s",
            event.event_type.value,
            event.source.value,
            event.file_path.name,
        )

        if event.event_type == FileEventType.DELETED:
            logger.info(
                "File deleted | source=%s | file=%s",
                event.source.value,
                event.file_path.name,
            )
            return

        result = await self.validator.validate(event.file_path)

        if not result.is_valid:
            logger.warning(
                "File rejected | source=%s | file=%s | reason=%s",
                event.source.value,
                event.file_path.name,
                result.reason,
            )

            if not result.retryable:
                self.quarantine_service.quarantine(
                    file_path=event.file_path,
                    source=event.source,
                    reason=result.reason,
                )

            return

        logger.info(
            "File validated | source=%s | file=%s | reason=%s",
            event.source.value,
            event.file_path.name,
            result.reason,
        )

        await self.dispatcher.dispatch(event)
