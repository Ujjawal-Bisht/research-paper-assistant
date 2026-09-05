from langchain_core.documents import Document

from app.rag.embeddings import get_embedding_model
from app.rag.vector_store import create_vector_store
from app.rag.retriever import (
    get_retriever,
    retrieve_documents,
)


def test_retriever():
    """
    Test retriever creation and document retrieval.
    """

    documents = [
        Document(
            page_content=(
                "The research paper proposes a deep learning "
                "model for image classification."
            ),
            metadata={
                "source": "paper_a.pdf",
                "page": 1,
            },
        ),
        Document(
            page_content=(
                "The authors evaluate the proposed model using "
                "the CIFAR-10 dataset."
            ),
            metadata={
                "source": "paper_a.pdf",
                "page": 2,
            },
        ),
        Document(
            page_content=(
                "The experimental results show improved "
                "classification accuracy."
            ),
            metadata={
                "source": "paper_a.pdf",
                "page": 3,
            },
        ),
        Document(
            page_content=(
                "The paper discusses possible future work "
                "for improving the proposed architecture."
            ),
            metadata={
                "source": "paper_a.pdf",
                "page": 4,
            },
        ),
    ]

    embedding_model = get_embedding_model()

    vector_store = create_vector_store(
        documents,
        embedding_model,
    )

    retriever = get_retriever(
        vector_store,
        top_k=2,
    )

    results = retrieve_documents(
        retriever,
        "What dataset was used?"
    )

    assert results
    assert len(results) == 2

    assert results[0].metadata["source"] == "paper_a.pdf"

    print("\nRetrieved documents:")

    for index, document in enumerate(
        results,
        start=1,
    ):
        print(f"\n--- Result {index} ---")
        print(
            f"Source: {document.metadata.get('source')}"
        )
        print(
            f"Page: {document.metadata.get('page')}"
        )
        print(
            f"Content: {document.page_content}"
        )


def test_empty_query():
    """
    Verify that an empty query is rejected.
    """

    documents = [
        Document(
            page_content="Sample research content.",
            metadata={
                "source": "sample.pdf",
                "page": 1,
            },
        )
    ]

    embedding_model = get_embedding_model()

    vector_store = create_vector_store(
        documents,
        embedding_model,
    )

    retriever = get_retriever(
        vector_store,
        top_k=1,
    )

    try:
        retrieve_documents(
            retriever,
            "",
        )
        assert False, "Expected ValueError"

    except ValueError:
        pass


if __name__ == "__main__":
    test_retriever()
    test_empty_query()

    print("\nRetriever tests passed.")