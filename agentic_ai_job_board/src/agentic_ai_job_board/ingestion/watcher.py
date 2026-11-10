
import asyncio
import threading

from watchfiles import awatch

from agentic_ai_job_board.core.config import settings
from agentic_ai_job_board.core.logging import get_logger
from agentic_ai_job_board.ingestion.event_handler import (
    FileEventHandler,
)


logger = get_logger(__name__)


class FileWatcher:
    """Asynchronously monitor resume and posting directories."""

    def __init__(self) -> None:
        self.watch_dirs = (
            settings.RESUME_DIR.resolve(),
            settings.POSTING_DIR.resolve(),
        )

        self.stop_event = threading.Event()
        self.event_handler = FileEventHandler()

        self._initialize_directories()

    def _initialize_directories(self) -> None:
        """Ensure monitored directories exist."""

        for directory in self.watch_dirs:
            directory.mkdir(parents=True, exist_ok=True)

            logger.info(
                "Monitoring directory initialized: %s",
                directory,
            )

    async def run(self) -> None:
        """Start asynchronous filesystem monitoring."""

        logger.info("File watcher starting...")

        try:
            async for changes in awatch(
                *self.watch_dirs,
                stop_event=self.stop_event,
                debounce=1000,
                step=100,
                rust_timeout=1000,
            ):
                for change, file_path in changes:
                    try:
                        await self.event_handler.handle(
                            change,
                            file_path,
                        )

                    except asyncio.CancelledError:
                        raise

                    except Exception:
                        logger.exception(
                            "Failed to process filesystem event: %s",
                            file_path,
                        )

        except asyncio.CancelledError:
            logger.info("File watcher task cancelled")
            raise

        except Exception:
            logger.exception("Unexpected error in file watcher")
            raise

        finally:
            logger.info("File watcher stopped")

    def stop(self) -> None:
        """Request graceful shutdown."""

        logger.info("File watcher shutdown requested")
        self.stop_event.set()