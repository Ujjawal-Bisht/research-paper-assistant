from app.ingestion.document_loader import load_files
from app.ingestion.text_splitter import split_documents

from app.rag.embeddings import get_embedding_model
from app.rag.vector_store import create_vector_store
from app.rag.retriever import (
    get_retriever,
    retrieve_documents,
)


def test_retriever_pipeline():
    """
    Test the complete retrieval pipeline:

    Documents
        ↓
    Chunks
        ↓
    Embeddings
        ↓
    FAISS
        ↓
    Retriever
        ↓
    Relevant chunks
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

    retriever = get_retriever(
        vector_store,
        top_k=5,
    )

    query = (
        "What is the main objective of the research?"
    )

    results = retrieve_documents(
        retriever,
        query,
    )

    assert results
    assert len(results) == 5

    print("\n========================================")
    print("RETRIEVAL RESULTS")
    print("========================================")

    print(f"\nQuery: {query}")

    print(f"\nDocuments loaded: {len(documents)}")
    print(f"Chunks generated: {len(chunks)}")
    print(f"Chunks retrieved: {len(results)}")

    for index, document in enumerate(
        results,
        start=1,
    ):
        print(
            f"\n--- Retrieved Chunk {index} ---"
        )

        print(
            "Source:",
            document.metadata.get("source"),
        )

        print(
            "Page:",
            document.metadata.get("page"),
        )

        print(
            "Content:",
            document.page_content[:500],
        )


if __name__ == "__main__":
    test_retriever_pipeline()

    print("\nRetriever pipeline test passed.")