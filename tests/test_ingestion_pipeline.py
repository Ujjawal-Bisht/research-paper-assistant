from app.ingestion.document_loader import load_files
from app.ingestion.text_splitter import split_documents


def test_ingestion_pipeline():
    documents = load_files("data/sample_papers")

    assert documents

    chunks = split_documents(documents)

    assert chunks

    print(f"Documents: {len(documents)}")
    print(f"Chunks: {len(chunks)}")

    print("\nFirst chunk:")
    print(chunks[0].page_content[:500])

    print("\nMetadata:")
    print(chunks[0].metadata)

    for i, chunk in enumerate(chunks[:5]):
        print(f"\n--- Chunk {i} ---")
        print(chunk.page_content)
        print("Metadata:", chunk.metadata)

if __name__ == "__main__":
    test_ingestion_pipeline()
    print("Ingestion pipeline test completed successfully.")

    