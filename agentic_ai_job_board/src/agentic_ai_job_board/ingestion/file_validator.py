import asyncio
import time
from dataclasses import dataclass
from pathlib import Path

from agentic_ai_job_board.core.config import settings
from agentic_ai_job_board.core.logging import get_logger


logger = get_logger(__name__)

@dataclass(frozen=True, slots=True)
class ValidationResult:
    is_valid: bool
    reason: str
    retryable: bool = False

class FileValidator:
    """Validate uploaded resume and posting documents."""

    def __init__(self) -> None:
        self.supported_extensions = {
            extension.lower()
            for extension in settings.SUPPORTED_FILE_EXTENSIONS
        }

        self.max_file_size = (
            settings.MAX_FILE_SIZE_MB * 1024 * 1024
        )

    async def wait_until_stable(self, file_path: Path) -> bool:
        """Wait until file size and modification time stop changing."""

        start_time = time.monotonic()
        previous_state = None
        stable_checks = 0

        while (
            time.monotonic() - start_time
            < settings.FILE_STABILITY_TIMEOUT_SECONDS
        ):
            try:
                stat = await asyncio.to_thread(file_path.stat)

                current_state = (
                    stat.st_size,
                    stat.st_mtime_ns,
                )

                if current_state == previous_state:
                    stable_checks += 1
                else:
                    stable_checks = 0

                if stable_checks >= settings.FILE_STABILITY_CHECKS:
                    return True

                previous_state = current_state

            except FileNotFoundError:
                return False

            await asyncio.sleep(
                settings.FILE_STABILITY_INTERVAL_SECONDS
            )

        return False

    async def validate(self, file_path: Path) -> ValidationResult:
        """Validate a document before ingestion."""

        try:
            if not await asyncio.to_thread(file_path.is_file):
                return ValidationResult(
                    is_valid=False,
                    reason="File does not exist or is not a regular file",
                    retryable=True
                )

            extension = file_path.suffix.lower()

            if extension not in self.supported_extensions:
                return ValidationResult(
                    is_valid=False,
                    reason=f"Unsupported file extension: {extension}",
                    retryable=False
                )

            is_stable = await self.wait_until_stable(file_path)

            if not is_stable:
                return ValidationResult(
                    is_valid=False,
                    reason="File did not become stable within the timeout",
                    retryable=True
                )

            stat = await asyncio.to_thread(file_path.stat)

            if stat.st_size == 0:
                return ValidationResult(
                    is_valid=False,
                    reason="File is empty",
                    retryable=False
                )

            if stat.st_size > self.max_file_size:
                return ValidationResult(
                    is_valid=False,
                    reason="File exceeds maximum allowed size",
                    retryable=False
                )

            return ValidationResult(
                is_valid=True,
                reason="File validation successful",
                retryable=False
            )

        except OSError:
            logger.exception(
                "File validation failed: %s",
                file_path,
            )

            return ValidationResult(
                is_valid=False,
                reason="Unable to access file",
                retryable=True
            )
