from typing import List

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_core.retrievers import BaseRetriever

from app.config.config import DEFAULT_TOP_K
from app.utils.logger import get_logger


logger = get_logger(__name__)


def get_retriever(
    vector_store: VectorStore,
    top_k: int = DEFAULT_TOP_K,
) -> BaseRetriever:
    """
    Create a retriever from an existing vector store.

    Parameters
    ----------
    vector_store : VectorStore
        Vector store containing the indexed document chunks.

    top_k : int, optional
        Number of relevant chunks to retrieve for each query.

    Returns
    -------
    BaseRetriever
        Configured LangChain retriever.

    Raises
    ------
    ValueError
        If top_k is less than or equal to zero.
    """

    if vector_store is None:
        raise ValueError(
            "vector_store cannot be None."
        )

    if top_k <= 0:
        raise ValueError(
            "top_k must be greater than zero."
        )

    logger.info(
        "Creating retriever with top_k=%d.",
        top_k,
    )

    try:
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": top_k,
            },
        )

        logger.info(
            "Retriever created successfully."
        )

        return retriever

    except Exception:
        logger.exception(
            "Failed to create retriever."
        )
        raise


def retrieve_documents(
    retriever: BaseRetriever,
    query: str,
) -> List[Document]:
    """
    Retrieve relevant document chunks for a user query.

    Parameters
    ----------
    retriever : BaseRetriever
        Configured LangChain retriever.

    query : str
        User's research question.

    Returns
    -------
    List[Document]
        Retrieved document chunks.

    Raises
    ------
    ValueError
        If the query is empty.
    """

    if retriever is None:
        raise ValueError(
            "retriever cannot be None."
        )

    if not query or not query.strip():
        raise ValueError(
            "Query cannot be empty."
        )

    query = query.strip()

    logger.info(
        "Retrieving documents for query: %s",
        query,
    )

    try:
        documents = retriever.invoke(query)

        logger.info(
            "Retrieved %d relevant chunks.",
            len(documents),
        )

        if not documents:
            logger.warning(
                "No relevant documents found for query: %s",
                query,
            )

        for index, document in enumerate(
            documents,
            start=1,
        ):
            logger.debug(
                "Retrieved chunk %d | source=%s | page=%s",
                index,
                document.metadata.get("source"),
                document.metadata.get("page"),
            )

        return documents

    except Exception:
        logger.exception(
            "Failed to retrieve documents for query: %s",
            query,
        )
        raise