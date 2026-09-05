from app.services.llm_service import (
    GROQ_MODEL,
    get_groq_client,
    generate_response,
)


def test_groq_client():
    """
    Test Groq client initialization.
    """

    client = get_groq_client()

    assert client is not None

    print(
        f"Groq client initialized successfully "
        f"using model: {GROQ_MODEL}"
    )


def test_groq_response():
    """
    Test a basic Groq model response.
    """

    client = get_groq_client()

    response = generate_response(
        client,
        (
            "Explain the purpose of a research paper "
            "in two sentences."
        ),
    )

    assert response is not None
    assert response.strip()

    print("\nGroq response:")
    print(response)


if __name__ == "__main__":
    test_groq_client()
    test_groq_response()

    print("\nGroq LLM tests passed.")