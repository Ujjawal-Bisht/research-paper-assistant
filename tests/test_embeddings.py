from app.rag.embeddings import get_embedding_model


def test_embedding_model():
    embedding_model = get_embedding_model()

    assert embedding_model is not None

    vector = embedding_model.embed_query(
        "What methodology does this research paper use?"
    )

    assert vector is not None
    assert len(vector) > 0

    print(f"Embedding dimensions: {len(vector)}")


if __name__ == "__main__":
    test_embedding_model()
    print("Embedding test passed.")