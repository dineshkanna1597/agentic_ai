
from agentic_ai_job_board.core.logging import get_logger
from agentic_ai_job_board.ingestion.events import (
    FileEvent,
    FileEventType,
    FileSource,
)
from agentic_ai_job_board.workers.ingestion_worker import (
    process_document,
)


logger = get_logger(__name__)


class IngestionDispatcher:
    """Submit validated file events to Celery."""

    async def dispatch(self, event: FileEvent) -> None:
        """Enqueue a document ingestion task."""

        if event.event_type == FileEventType.DELETED:
            logger.info(
                "Skipping ingestion for deleted file | file=%s",
                event.file_path.name,
            )
            return

        if event.source not in (FileSource.RESUME, FileSource.POSTING):
            logger.warning(
                "No ingestion task for source=%s | file=%s",
                event.source.value,
                event.file_path.name,
            )
            return

        try:
            result = process_document.apply_async(
                args=[
                    str(event.file_path),
                    event.source.value,
                    event.event_type.value,
                    event.timestamp.isoformat(),
                ],
            )

            logger.info(
                "Ingestion task queued | task_id=%s | source=%s | file=%s",
                result.id,
                event.source.value,
                event.file_path.name,
            )

        except Exception:
            logger.exception(
                "Failed to queue ingestion task | source=%s | file=%s",
                event.source.value,
                event.file_path.name,
            )
            raise