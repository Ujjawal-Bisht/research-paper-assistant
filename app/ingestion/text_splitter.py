from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config.config import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
)
from app.utils.logger import get_logger


logger = get_logger(__name__)


def split_documents(
    documents: List[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Document]:
    
    """
    Split LangChain documents into smaller chunks for RAG.

    Parameters
    ----------
    documents : List[Document]
        Documents returned by the document loader.

    chunk_size : int, optional
        Maximum size of each chunk.

    chunk_overlap : int, optional
        Number of characters shared between consecutive chunks.

    Returns
    -------
    List[Document]
        List of chunked LangChain Document objects.
    """

    if not documents:
        logger.warning("No documents received for splitting.")
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative.")

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size."
        )

    logger.info(
        "Starting document splitting. Documents: %d, "
        "chunk_size: %d, chunk_overlap: %d",
        len(documents),
        chunk_size,
        chunk_overlap,
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks = splitter.split_documents(documents)

    logger.info(
        "Document splitting completed. Generated %d chunks.",
        len(chunks),
    )

    return chunks

