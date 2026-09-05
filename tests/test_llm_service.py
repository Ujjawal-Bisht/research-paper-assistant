import os

import pytest

from app.services.llm_service import (
    GROQ_MODEL,
    get_groq_client,
    generate_response,
)


LIVE_GROQ_TESTS_ENABLED = bool(
    os.getenv("RUN_LIVE_GROQ_TESTS")
)


pytestmark = pytest.mark.skipif(
    not LIVE_GROQ_TESTS_ENABLED,
    reason=(
        "Live Groq tests disabled. Set RUN_LIVE_GROQ_TESTS=1 to enable."
    ),
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