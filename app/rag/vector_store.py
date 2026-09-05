from pathlib import Path
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.utils.logger import get_logger


logger = get_logger(__name__)


DEFAULT_VECTOR_STORE_PATH = "data/vector_store"


def create_vector_store(
    chunks: List[Document],
    embedding_model: Embeddings,
) -> FAISS:
    """
    Create a FAISS vector store from document chunks.

    Parameters
    ----------
    chunks : List[Document]
        Document chunks that will be embedded and indexed.

    embedding_model : Embeddings
        Embedding model used to convert text into vectors.

    Returns
    -------
    FAISS
        FAISS vector store containing the document chunks.

    Raises
    ------
    ValueError
        If no chunks are provided.
    """

    if not chunks:
        logger.warning(
            "No chunks provided for vector store creation."
        )
        raise ValueError(
            "Cannot create a vector store from empty chunks."
        )

    logger.info(
        "Creating FAISS vector store from %d chunks.",
        len(chunks),
    )

    try:
        vector_store = FAISS.from_documents(
            documents=chunks,
            embedding=embedding_model,
        )

        logger.info(
            "FAISS vector store created successfully. "
            "Indexed %d chunks.",
            len(chunks),
        )

        return vector_store

    except Exception:
        logger.exception(
            "Failed to create FAISS vector store."
        )
        raise


def save_vector_store(
    vector_store: FAISS,
    store_path: str = DEFAULT_VECTOR_STORE_PATH,
) -> None:
    """
    Save a FAISS vector store to disk.

    Parameters
    ----------
    vector_store : FAISS
        FAISS vector store to save.

    store_path : str, optional
        Directory where the vector store will be stored.
    """

    path = Path(store_path).resolve()

    logger.info(
        "Saving FAISS vector store to: %s",
        path,
    )

    try:
        path.mkdir(
            parents=True,
            exist_ok=True,
        )

        vector_store.save_local(str(path))

        logger.info(
            "FAISS vector store saved successfully to: %s",
            path,
        )

    except Exception:
        logger.exception(
            "Failed to save FAISS vector store to: %s",
            path,
        )
        raise


def load_vector_store(
    embedding_model: Embeddings,
    store_path: str = DEFAULT_VECTOR_STORE_PATH,
) -> FAISS:
    """
    Load a previously saved FAISS vector store.

    Parameters
    ----------
    embedding_model : Embeddings
        Embedding model associated with the vector store.

    store_path : str, optional
        Directory containing the saved FAISS vector store.

    Returns
    -------
    FAISS
        Loaded FAISS vector store.

    Raises
    ------
    FileNotFoundError
        If the vector store directory does not exist.
    """

    path = Path(store_path).resolve()

    if not path.exists():
        logger.error(
            "Vector store directory does not exist: %s",
            path,
        )
        raise FileNotFoundError(
            f"Vector store not found: {path}"
        )

    logger.info(
        "Loading FAISS vector store from: %s",
        path,
    )

    try:
        vector_store = FAISS.load_local(
            str(path),
            embedding_model,
            allow_dangerous_deserialization=True,
        )

        logger.info(
            "FAISS vector store loaded successfully."
        )

        return vector_store

    except Exception:
        logger.exception(
            "Failed to load FAISS vector store from: %s",
            path,
        )
        raise