
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """Centralized application configuration."""

    # Application
    APP_NAME: str = "Agentic AI Job Board"
    APP_ENV: str = "development"
    DEBUG: bool = True

    # Directories
    DATA_DIR: Path = BASE_DIR / "src" / "agentic_ai_job_board" / "data"
    RESUME_DIR: Path = BASE_DIR / "src" / "agentic_ai_job_board" / "data" / "resumes"
    POSTING_DIR: Path = BASE_DIR / "src" / "agentic_ai_job_board" / "data" / "postings"
    LOG_DIR: Path = BASE_DIR / "src" / "agentic_ai_job_board" / "logs"

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "app.log"
    LOG_MAX_BYTES: int = 10 * 1024 * 1024  # 10 MB
    LOG_BACKUP_COUNT: int = 5

    # File validation
    MAX_FILE_SIZE_MB: int = 20
    SUPPORTED_FILE_EXTENSIONS: tuple[str, ...] = (
    ".pdf",
    ".docx",
    ".txt",
    )

    FILE_STABILITY_INTERVAL_SECONDS: float = 1.0    
    FILE_STABILITY_CHECKS: int = 3
    FILE_STABILITY_TIMEOUT_SECONDS: float = 30.0

    # Celery / Redis
    REDIS_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/1"
    CELERY_TASK_RETRY_DELAY: int = 5
    CELERY_TASK_MAX_RETRIES: int = 3

        # Qdrant
    QDRANT_STORAGE_PATH: Path = (
        BASE_DIR
        / "src"
        / "agentic_ai_job_board"
        / "rag"
        / "qdrant_local_storage"
    )
    QDRANT_COLLECTION_NAME: str = "career_portal"
    QDRANT_VECTOR_SIZE: int = 4096

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


settings = Settings()
