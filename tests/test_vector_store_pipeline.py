from app.ingestion.document_loader import load_files
from app.ingestion.text_splitter import split_documents
from app.rag.embeddings import get_embedding_model
from app.rag.vector_store import (
    create_vector_store,
    save_vector_store,
)


TEST_VECTOR_STORE_PATH = "data/test_sample_vector_store"


def test_vector_store_pipeline():
    """
    Test the complete pipeline:

    Documents → Chunks → Embeddings → FAISS
    """

    documents = load_files(
        "data/sample_papers"
    )

    assert documents

    chunks = split_documents(
        documents
    )

    assert chunks

    embedding_model = get_embedding_model()

    vector_store = create_vector_store(
        chunks,
        embedding_model,
    )

    assert vector_store is not None

    results = vector_store.similarity_search(
        "What is the main objective of the research?",
        k=3,
    )

    assert results
    assert len(results) == 3

    print(f"Documents loaded: {len(documents)}")
    print(f"Chunks generated: {len(chunks)}")
    print(f"Retrieved chunks: {len(results)}")

    for index, result in enumerate(results, start=1):
        print(f"\n--- Result {index} ---")
        print("Source:", result.metadata.get("source"))
        print("Page:", result.metadata.get("page"))
        print("Content:")
        print(result.page_content[:500])

    save_vector_store(
        vector_store,
        TEST_VECTOR_STORE_PATH,
    )

    print("\nReal document vector-store pipeline passed.")


if __name__ == "__main__":
    test_vector_store_pipeline()

    print("\nVector store pipeline test passed.")