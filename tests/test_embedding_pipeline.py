from app.ingestion.document_loader import load_files
from app.ingestion.text_splitter import split_documents
from app.rag.embeddings import get_embedding_model


def test_embedding_pipeline():
    documents = load_files("data/sample_papers")

    assert documents

    chunks = split_documents(documents)

    assert chunks

    embedding_model = get_embedding_model()

    vector = embedding_model.embed_query(
        chunks[0].page_content
    )

    assert vector
    assert len(vector) > 0

    print(f"Documents loaded: {len(documents)}")
    print(f"Chunks generated: {len(chunks)}")
    print(f"Embedding dimensions: {len(vector)}")


if __name__ == "__main__":
    test_embedding_pipeline()
    print("Embedding pipeline test passed.")