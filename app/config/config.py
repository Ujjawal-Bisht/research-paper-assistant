import os
from pathlib import Path

from dotenv import load_dotenv


ENV_FILE = Path(__file__).resolve().parents[2] / ".env"

load_dotenv(
    dotenv_path=ENV_FILE,
    override=True,
)


# ============================================================
# Project Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SYSTEM_PROMPT_FILE = (
    Path(__file__).resolve().parent
    / "prompts"
    / "system_prompt.json"
)

QA_PROMPT_FILE = (
    Path(__file__).resolve().parent
    / "prompts"
    / "qa_prompt.json"
)

SUMMARY_PROMPT_FILE = (
    Path(__file__).resolve().parent
    / "prompts"
    / "summary_prompt.json"
)


# ============================================================
# Groq Configuration
# ============================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

GROQ_MODEL = "groq/compound"


# ============================================================
# Groq Retry Configuration
# ============================================================

# Maximum number of additional attempts after the initial
# Groq request fails because of a rate limit.
GROQ_MAX_RETRIES = 3


# Default delay used when Groq does not provide a retry
# duration in the rate-limit response.
GROQ_DEFAULT_RETRY_DELAY = 10


# Maximum delay allowed between retry attempts.
GROQ_MAX_RETRY_DELAY = 60


# ============================================================
# Embedding Configuration
# ============================================================

EMBEDDING_MODEL_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# RAG Configuration
# ============================================================

DEFAULT_CHUNK_SIZE = 500

DEFAULT_CHUNK_OVERLAP = 100

DEFAULT_TOP_K = 5


# ============================================================
# Vector Store Configuration
# ============================================================

VECTOR_STORE_PATH = "data/vector_store"

TEST_VECTOR_STORE_PATH = "data/test_sample_vector_store"


# ============================================================
# Summarization Configuration
# ============================================================

SUMMARY_BATCH_CHAR_LIMIT = 12000

SUMMARY_SYNTHESIS_CHAR_LIMIT = 10000

# ============================================================
# Application Configuration
# ============================================================

APP_TITLE = "Research Paper Assistant"


# ============================================================
# Logging Configuration
# ============================================================

LOG_DIR = PROJECT_ROOT / "logs"

LOG_FORMAT = (
    "%(asctime)s | "
    "%(levelname)-8s | "
    "%(name)s | "
    "%(message)s"
)

LOG_MAX_BYTES = 5 * 1024 * 1024

LOG_BACKUP_COUNT = 3


def validate_config() -> None:
    """
    Validate required application configuration.
    """

    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY is not configured. "
            "Add GROQ_API_KEY to the .env file."
        )