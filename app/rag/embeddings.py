from langchain_huggingface import HuggingFaceEmbeddings

from app.utils.logger import get_logger


logger = get_logger(__name__)


EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Create and return the local embedding model used by the RAG pipeline.

    The embedding model runs locally and does not require an external
    embedding API key.

    Returns
    -------
    HuggingFaceEmbeddings
        Configured Hugging Face embedding model.
    """

    logger.info(
        "Initializing embedding model: %s",
        EMBEDDING_MODEL_NAME,
    )

    try:
        embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
        )

        logger.info(
            "Embedding model initialized successfully: %s",
            EMBEDDING_MODEL_NAME,
        )

        return embedding_model

    except Exception:
        logger.exception(
            "Failed to initialize embedding model: %s",
            EMBEDDING_MODEL_NAME,
        )
        raise