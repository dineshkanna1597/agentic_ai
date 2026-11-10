import asyncio

from agentic_ai_job_board.core.logging import (
    configure_logging,
    get_logger,
)
from agentic_ai_job_board.ingestion.watcher import FileWatcher


configure_logging()

logger = get_logger(__name__)


async def main() -> None:
    logger.info("Starting file monitoring service")

    watcher = FileWatcher()

    await watcher.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())

    except KeyboardInterrupt:
        logger.info("File monitoring service interrupted")

    except Exception:
        logger.exception("File monitoring service failed")
        raise
