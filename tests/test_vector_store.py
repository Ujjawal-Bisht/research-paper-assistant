from pathlib import Path

from langchain_core.documents import Document

from app.rag.embeddings import get_embedding_model
from app.rag.vector_store import (
    create_vector_store,
    save_vector_store,
    load_vector_store,
)


TEST_VECTOR_STORE_PATH = "data/test_vector_store"


def test_vector_store():
    """
    Test FAISS vector store creation, persistence, loading,
    and similarity search.
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
    ]

    embedding_model = get_embedding_model()

    # Create vector store
    vector_store = create_vector_store(
        documents,
        embedding_model,
    )

    assert vector_store is not None

    # Test similarity search
    results = vector_store.similarity_search(
        "What dataset was used?",
        k=1,
    )

    assert results
    assert len(results) == 1

    assert results[0].metadata["page"] == 2

    print("Similarity search result:")
    print(results[0].page_content)
    print("Metadata:", results[0].metadata)

    # Save vector store
    save_vector_store(
        vector_store,
        TEST_VECTOR_STORE_PATH,
    )

    store_path = Path(TEST_VECTOR_STORE_PATH)

    assert store_path.exists()
    assert (store_path / "index.faiss").exists()
    assert (store_path / "index.pkl").exists()

    # Load vector store
    loaded_vector_store = load_vector_store(
        embedding_model,
        TEST_VECTOR_STORE_PATH,
    )

    assert loaded_vector_store is not None

    # Test loaded vector store
    loaded_results = loaded_vector_store.similarity_search(
        "What dataset was used?",
        k=1,
    )

    assert loaded_results
    assert len(loaded_results) == 1

    assert loaded_results[0].metadata["page"] == 2

    print("\nLoaded vector store search result:")
    print(loaded_results[0].page_content)
    print("Metadata:", loaded_results[0].metadata)


if __name__ == "__main__":
    test_vector_store()

    print("\nVector store test passed.")