
import asyncio
from datetime import datetime
from pathlib import Path

from agentic_ai_job_board.core.celery_app import celery_app

from agentic_ai_job_board.core.config import settings
from agentic_ai_job_board.core.logging import configure_logging, get_logger
from agentic_ai_job_board.ingestion.events import (
    FileEvent,
    FileEventType,
    FileSource,
)
from agentic_ai_job_board.agents.data_ingestion.graph import (
    DataIngestionGraph,
)
from agentic_ai_job_board.agents.data_ingestion.context import (
    DataIngestionContext,
)

configure_logging()
logger = get_logger(__name__)


def build_event(
    file_path: str,
    source: str,
    event_type: str,
    timestamp: str,
) -> FileEvent:
    """Reconstruct a typed event from a JSON-safe task payload."""

    return FileEvent(
        event_type=FileEventType(event_type),
        source=FileSource(source),
        file_path=Path(file_path),
        timestamp=datetime.fromisoformat(timestamp),
    )

@celery_app.task(
    bind=True,
    name="ingestion.process_document",
    max_retries=settings.CELERY_TASK_MAX_RETRIES,
    default_retry_delay=settings.CELERY_TASK_RETRY_DELAY,
    acks_late=True,
)
def process_document(
    self,
    file_path: str,
    source: str,
    event_type: str,
    timestamp: str,
) -> dict[str, str]:
    """Execute document ingestion through the LangGraph workflow."""

    try:
        event = build_event(file_path, source, event_type, timestamp)
        context = DataIngestionContext(event=event)

        logger.info(
            "Document ingestion task started | task_id=%s | source=%s | file=%s",
            self.request.id,
            source,
            event.file_path.name,
        )

        result = asyncio.run(DataIngestionGraph().run(context))

        return {
            "status": result.get("processing_status", "completed"),
            "source": source,
            "file": event.file_path.name,
        }

    except Exception as exc:
        logger.exception(
            "Document ingestion task failed | task_id=%s | source=%s | file=%s",
            self.request.id,
            source,
            Path(file_path).name,
        )
        raise self.retry(exc=exc)