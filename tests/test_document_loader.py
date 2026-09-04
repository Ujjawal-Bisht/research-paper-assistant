from app.ingestion.document_loader import load_files

def test_doc_loader():
    documents = load_files("data/sample_papers")

    print(f"Documents loaded: {len(documents)}")
    print(documents[0].metadata)
    print(documents[0].page_content[:500])


if(__name__ == '__main__'):
    test_doc_loader()