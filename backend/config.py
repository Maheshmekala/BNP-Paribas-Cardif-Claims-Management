"""
Configuration module for BNP Paribas Cardif Claims Management System.
Loads environment variables and provides typed configuration.
"""

import os
import logging
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load .env from project root
project_root = Path(__file__).resolve().parent.parent
load_dotenv(project_root / ".env")


def _get_env(key: str, default: str = "") -> str:
    """Get an environment variable with a fallback default."""
    return os.environ.get(key, default)


class Settings:
    """Application configuration loaded from environment variables."""

    # Project paths
    PROJECT_ROOT: Path = project_root
    BACKEND_DIR: Path = project_root / "backend"
    DATA_DIR: Path = project_root / "data"
    SAMPLES_DIR: Path = project_root / "data" / "samples"
    VECTOR_DB_PATH: Path = project_root / "data" / "vector_db"

    # Database
    DATABASE_URL: str = _get_env("DATABASE_URL", "sqlite:///./data/claims.db")

    # MCP Server
    MCP_SERVER_PORT: int = int(_get_env("MCP_SERVER_PORT", "8100"))

    # Backend API
    BACKEND_PORT: int = int(_get_env("BACKEND_PORT", "8000"))

    # Frontend
    FRONTEND_PORT: int = int(_get_env("FRONTEND_PORT", "8501"))

    # Logging
    LOG_LEVEL: str = _get_env("LOG_LEVEL", "INFO")

    # LLM API Keys (optional)
    OPENAI_API_KEY: Optional[str] = _get_env("OPENAI_API_KEY") or None
    ANTHROPIC_API_KEY: Optional[str] = _get_env("ANTHROPIC_API_KEY") or None

    # Tesseract
    TESSERACT_CMD: Optional[str] = _get_env("TESSERACT_CMD") or None

    # ChromaDB settings
    CHROMA_COLLECTION_POLICIES: str = "insurance_policies"
    CHROMA_COLLECTION_CLAIMS: str = "historical_claims"
    CHROMA_COLLECTION_FEW_SHOT: str = "few_shot_examples"

    # Embedding model
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"


settings = Settings()

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("claims_management")


def ensure_directories() -> None:
    """Create necessary directories if they don't exist."""
    for path in [settings.DATA_DIR, settings.SAMPLES_DIR, settings.VECTOR_DB_PATH]:
        path.mkdir(parents=True, exist_ok=True)
    logger.info("All directories verified/created.")
