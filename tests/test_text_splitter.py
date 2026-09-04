from langchain_core.documents import Document
from app.ingestion.text_splitter import split_documents


def test_split_documents():
    documents = [
        Document(
            page_content=(
                "This is a sample research paper. "
                "It contains enough text to test "
                "the document splitting functionality."
            ),
            metadata={
                "source": "sample.pdf",
                "page": 1,
            },
        )
    ]

    chunks = split_documents(
        documents,
        chunk_size=50,
        chunk_overlap=10,
    )

    assert len(chunks) > 1

    for chunk in chunks:
        assert chunk.page_content.strip()
        assert chunk.metadata["source"] == "sample.pdf"
        assert chunk.metadata["page"] == 1


if __name__ == "__main__":
    test_split_documents()
    print("All tests passed.")