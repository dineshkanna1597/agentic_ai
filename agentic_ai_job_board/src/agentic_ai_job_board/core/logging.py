
import logging
# import logging.config
from logging.handlers import RotatingFileHandler

from agentic_ai_job_board.core.config import settings


def configure_logging() -> None:
    """Configure application-wide logging."""

    settings.LOG_DIR.mkdir(parents=True, exist_ok=True)

    log_file = settings.LOG_DIR / settings.LOG_FILE

    formatter = logging.Formatter(
        fmt=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(name)s | "
            "%(funcName)s:%(lineno)d | "
            "%(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = RotatingFileHandler(
        filename=log_file,
        maxBytes=settings.LOG_MAX_BYTES,
        backupCount=settings.LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    root_logger = logging.getLogger()

    # Avoid duplicate handlers during repeated initialization.
    if root_logger.handlers:
        root_logger.handlers.clear()

    root_logger.setLevel(settings.LOG_LEVEL.upper())
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Return a named application logger."""

    return logging.getLogger(name)
