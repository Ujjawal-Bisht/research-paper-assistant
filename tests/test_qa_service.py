from langchain_core.documents import Document

from app.rag.embeddings import get_embedding_model
from app.rag.retriever import get_retriever
from app.rag.vector_store import load_vector_store
from app.services.qa_service import (
    answer_question,
    build_qa_prompt,
    extract_sources,
    format_context,
    load_qa_prompt,
)
from app.config.config import TEST_VECTOR_STORE_PATH


def test_load_qa_prompt():
    """
    Test that the QA prompt configuration can be loaded
    successfully from qa_prompt.json.
    """

    prompt_config = load_qa_prompt()

    assert isinstance(prompt_config, dict)
    assert "prompt_template" in prompt_config
    assert "instructions" in prompt_config
    assert "response_requirements" in prompt_config

    print("\nQA prompt loaded successfully.")


def test_format_context():
    """
    Test conversion of LangChain Documents into LLM context.
    """

    documents = [
        Document(
            page_content="This is research paper content.",
            metadata={
                "source": "sample.pdf",
                "page": 2,
            },
        ),
        Document(
            page_content="This is additional paper content.",
            metadata={
                "source": "sample.pdf",
                "page": 5,
            },
        ),
    ]

    context = format_context(documents)

    assert context
    assert "sample.pdf" in context
    assert "Page: 3" in context
    assert "Page: 6" in context
    assert "This is research paper content." in context
    assert "This is additional paper content." in context

    print("\nContext formatting passed.")


def test_build_qa_prompt():
    """
    Test construction of the final RAG prompt.
    """

    prompt_config = {
        "prompt_template": {
            "context": "{context}",
            "question": "{question}",
        },
        "instructions": [
            "Use only the provided context.",
            "Do not fabricate information.",
        ],
        "response_requirements": {
            "include_sources": True,
        },
    }

    context = (
        "Document: sample.pdf\n"
        "Page: 3\n"
        "Content:\n"
        "The proposed model improves classification accuracy."
    )

    question = (
        "What does the proposed model improve?"
    )

    prompt = build_qa_prompt(
        question=question,
        context=context,
        prompt_config=prompt_config,
    )

    assert prompt

    assert "Use only the provided context." in prompt
    assert "Do not fabricate information." in prompt
    assert "sample.pdf" in prompt
    assert "classification accuracy" in prompt
    assert question in prompt

    print("\nQA prompt construction passed.")


def test_extract_sources():
    """
    Test extraction of unique source and page information.
    """

    documents = [
        Document(
            page_content="Content 1",
            metadata={
                "source": "paper.pdf",
                "page": 1,
            },
        ),
        Document(
            page_content="Content 2",
            metadata={
                "source": "paper.pdf",
                "page": 1,
            },
        ),
        Document(
            page_content="Content 3",
            metadata={
                "source": "paper.pdf",
                "page": 4,
            },
        ),
    ]

    sources = extract_sources(documents)

    assert len(sources) == 2

    assert {
        "source": "paper.pdf",
        "page": 2,
    } in sources

    assert {
        "source": "paper.pdf",
        "page": 5,
    } in sources

    print("\nSource extraction passed.")


def test_answer_question():
    """
    End-to-end integration test:

        FAISS
        -> Retriever
        -> QA Service
        -> Groq
        -> Answer + Sources

    This test requires:
        1. A previously created FAISS vector store.
        2. GROQ_API_KEY in the .env file.
    """

    embedding_model = get_embedding_model()

    vector_store = load_vector_store(
        embedding_model=embedding_model,
        store_path=TEST_VECTOR_STORE_PATH,
    )

    retriever = get_retriever(
        vector_store=vector_store,
    )

    question = (
        "What is the main objective of the research paper?"
    )

    result = answer_question(
        retriever=retriever,
        question=question,
    )

    assert isinstance(result, dict)

    assert "answer" in result
    assert "sources" in result

    assert result["answer"]
    assert isinstance(result["sources"], list)

    print("\n========================================")
    print("QUESTION")
    print("========================================")
    print(question)

    print("\n========================================")
    print("ANSWER")
    print("========================================")
    print(result["answer"])

    print("\n========================================")
    print("SOURCES")
    print("========================================")

    for source in result["sources"]:
        print(
            f"- {source['source']} | "
            f"Page: {source['page']}"
        )


if __name__ == "__main__":
    test_load_qa_prompt()
    test_format_context()
    test_build_qa_prompt()
    test_extract_sources()
    test_answer_question()

    print("\nAll QA service tests passed.")