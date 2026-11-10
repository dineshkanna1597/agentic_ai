import shutil
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from agentic_ai_job_board.core.config import settings
from agentic_ai_job_board.core.logging import get_logger
from agentic_ai_job_board.ingestion.events import FileSource


logger = get_logger(__name__)


class QuarantineService:
    """Move rejected documents into source-specific folders."""

    def __init__(self) -> None:
        self.rejected_dir = settings.DATA_DIR / "rejected"

    def quarantine(
        self,
        file_path: Path,
        source: FileSource,
        reason: str,
    ) -> Path | None:
        """Move a rejected file while preserving the original."""

        try:
            if not file_path.is_file():
                logger.warning(
                    "Cannot quarantine missing file: %s",
                    file_path,
                )
                return None

            destination_dir = self.rejected_dir / source.value
            destination_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now(timezone.utc).strftime(
                "%Y%m%dT%H%M%S%fZ"
            )
            unique_id = uuid4().hex[:8]

            destination = destination_dir / (
                f"{file_path.stem}_{timestamp}_{unique_id}"
                f"{file_path.suffix}"
            )

            shutil.move(str(file_path), str(destination))

            logger.warning(
                "File quarantined | source=%s | original=%s "
                "| destination=%s | reason=%s",
                source.value,
                file_path.name,
                destination,
                reason,
            )

            return destination

        except OSError:
            logger.exception(
                "Failed to quarantine file: %s",
                file_path,
            )
            return None
