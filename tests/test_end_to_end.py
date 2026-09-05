from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.config.config import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_TOP_K,
)
from app.ingestion.document_loader import load_files
from app.ingestion.text_splitter import split_documents
from app.rag.embeddings import get_embedding_model
from app.rag.retriever import get_retriever
from app.rag.vector_store import (
    create_vector_store,
    load_vector_store,
    save_vector_store,
)
from app.services.qa_service import answer_question
from app.services.summarizer import summarize_documents


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "sample_papers"


# ---------------------------------------------------------------------------
# Mock LLM responses
# ---------------------------------------------------------------------------

MOCK_QA_RESPONSE = """
The research paper addresses the research problem described in the
retrieved paper context. The answer is based only on the supplied
research-paper content.
""".strip()


MOCK_INTERMEDIATE_SUMMARY = """
The paper section discusses the research problem, methodology,
experimental setup, findings, and relevant technical details.
""".strip()


MOCK_FINAL_SUMMARY = """
## Overview

The research paper investigates a specific research problem using the
methodology and experimental approach described by the authors.

## Research Problem

The paper focuses on the research problem presented in the supplied
research-paper content.

## Objective

The objective is to investigate the proposed research approach and
evaluate its effectiveness.

## Methodology

The authors use the methodology, datasets, experiments, and evaluation
procedure described in the paper.

## Key Findings

The paper reports findings based on the experiments described by the
authors.

## Main Contribution

The main contribution is the research approach and findings presented
by the authors.

## Limitations

The limitations are those supported by the supplied research-paper
content.

## Future Work

Future research directions are those supported by the supplied
research-paper content.
""".strip()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_sample_documents():
    """
    Load the research papers used by the end-to-end test.
    """
    assert DATA_DIR.exists(), (
        f"Sample paper directory does not exist: {DATA_DIR}"
    )

    documents = load_files(str(DATA_DIR))

    assert documents, (
        f"No research papers were loaded from: {DATA_DIR}"
    )

    return documents


def get_first_paper_documents(documents):
    """
    Select all pages belonging to the first research paper.

    The sample directory may contain multiple papers. The summarizer
    operates on one paper at a time, so this keeps the summarization
    portion of the test scoped to a single paper.
    """
    first_source = documents[0].metadata.get("source")

    assert first_source, (
        "The first document does not contain a source in its metadata."
    )

    paper_documents = [
        document
        for document in documents
        if document.metadata.get("source") == first_source
    ]

    assert paper_documents, (
        "Could not find documents belonging to the first paper."
    )

    return paper_documents


# ---------------------------------------------------------------------------
# Master end-to-end test
# ---------------------------------------------------------------------------

def test_complete_backend_pipeline(tmp_path, monkeypatch):
    """
    Complete backend pipeline test.

    Pipeline:

        PDF files
            ->
        Document Loader
            ->
        Text Splitter
            ->
        HuggingFace Embeddings
            ->
        FAISS Vector Store
            ->
        Save Vector Store
            ->
        Load Vector Store
            ->
        Retriever
            ->
        QA Service
            ->
        Summarization Service
            ->
        Final Answer/Summary + Sources

    The embedding model, FAISS vector store, retrieval, document loading,
    splitting, QA orchestration, and summarization orchestration are real.

    LLM responses are mocked so this test does not consume Groq API quota.
    """

    # -----------------------------------------------------------------------
    # STEP 1: File ingestion
    # -----------------------------------------------------------------------

    documents = get_sample_documents()

    print(
        f"\n[1/9] File ingestion successful: "
        f"{len(documents)} documents loaded."
    )

    assert len(documents) > 0

    for document in documents:
        assert document.page_content.strip()
        assert document.metadata.get("source")


    # -----------------------------------------------------------------------
    # STEP 2: Select one paper for paper-level operations
    # -----------------------------------------------------------------------

    paper_documents = get_first_paper_documents(documents)

    print(
        f"[2/9] Selected one paper: "
        f"{len(paper_documents)} page documents."
    )

    assert len(paper_documents) > 0


    # -----------------------------------------------------------------------
    # STEP 3: Text splitting
    # -----------------------------------------------------------------------

    chunks = split_documents(
        paper_documents,
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
    )

    print(
        f"[3/9] Text splitting successful: "
        f"{len(chunks)} chunks created."
    )

    assert chunks
    assert len(chunks) > len(paper_documents)

    for chunk in chunks:
        assert chunk.page_content.strip()
        assert chunk.metadata.get("source")


    # -----------------------------------------------------------------------
    # STEP 4: Embedding generation
    # -----------------------------------------------------------------------

    embedding_model = get_embedding_model()

    print(
        "[4/9] Embedding model initialized successfully."
    )

    assert embedding_model is not None

    # Generate embeddings for a real chunk to verify that the embedding
    # model is actually operational.
    test_embedding = embedding_model.embed_query(
        chunks[0].page_content
    )

    assert test_embedding
    assert len(test_embedding) > 0

    print(
        f"[4/9] Embedding generation successful: "
        f"{len(test_embedding)} dimensions."
    )


    # -----------------------------------------------------------------------
    # STEP 5: FAISS vector-store creation
    # -----------------------------------------------------------------------

    vector_store = create_vector_store(
        chunks=chunks,
        embedding_model=embedding_model,
    )

    print(
        "[5/9] FAISS vector store created successfully."
    )

    assert vector_store is not None


    # -----------------------------------------------------------------------
    # STEP 6: Vector-store persistence
    # -----------------------------------------------------------------------

    vector_store_path = tmp_path / "e2e_vector_store"

    save_vector_store(
        vector_store=vector_store,
        store_path=str(vector_store_path),
    )

    assert vector_store_path.exists()

    loaded_vector_store = load_vector_store(
        embedding_model=embedding_model,
        store_path=str(vector_store_path),
    )

    assert loaded_vector_store is not None

    print(
        "[6/9] FAISS vector store save/load successful."
    )


    # -----------------------------------------------------------------------
    # STEP 7: Retriever
    # -----------------------------------------------------------------------

    retriever = get_retriever(
        vector_store=loaded_vector_store,
        top_k=DEFAULT_TOP_K,
    )

    assert retriever is not None

    test_question = (
        "What is the main research problem addressed by the paper?"
    )

    retrieved_documents = retriever.invoke(test_question)

    print(
        f"[7/9] Retrieval successful: "
        f"{len(retrieved_documents)} documents retrieved."
    )

    assert retrieved_documents
    assert len(retrieved_documents) <= DEFAULT_TOP_K

    for document in retrieved_documents:
        assert document.page_content.strip()
        assert document.metadata.get("source")


    # -----------------------------------------------------------------------
    # STEP 8: QA service
    # -----------------------------------------------------------------------

    mock_client = MagicMock()

    monkeypatch.setattr(
        "app.services.qa_service.get_groq_client",
        lambda: mock_client,
    )

    monkeypatch.setattr(
        "app.services.qa_service.generate_response",
        lambda user_prompt, client: MOCK_QA_RESPONSE,
    )

    qa_result = answer_question(
        retriever=retriever,
        question=test_question,
    )

    print(
        "[8/9] QA pipeline completed successfully."
    )

    assert qa_result is not None
    assert isinstance(qa_result, dict)

    assert "answer" in qa_result
    assert "sources" in qa_result

    assert qa_result["answer"]
    assert isinstance(qa_result["sources"], list)
    assert qa_result["sources"]

    for source in qa_result["sources"]:
        assert "source" in source
        assert "page" in source


    # -----------------------------------------------------------------------
    # STEP 9: Summarization service
    # -----------------------------------------------------------------------

    monkeypatch.setattr(
        "app.services.summarizer.get_groq_client",
        lambda: mock_client,
    )

    monkeypatch.setattr(
        "app.services.summarizer.generate_response",
        lambda user_prompt, client: (
            MOCK_FINAL_SUMMARY
            if "intermediate summary" not in user_prompt.lower()
            else MOCK_INTERMEDIATE_SUMMARY
        ),
    )

    summary_result = summarize_documents(
        paper_documents
    )

    print(
        "[9/9] Summarization pipeline completed successfully."
    )

    assert summary_result is not None
    assert isinstance(summary_result, dict)

    assert "summary" in summary_result
    assert "sources" in summary_result

    assert summary_result["summary"]
    assert isinstance(summary_result["sources"], list)
    assert summary_result["sources"]

    for source in summary_result["sources"]:
        assert "source" in source
        assert "page" in source


    # -----------------------------------------------------------------------
    # Final validation
    # -----------------------------------------------------------------------

    print("\n" + "=" * 60)
    print("COMPLETE BACKEND PIPELINE PASSED")
    print("=" * 60)

    print(f"Documents loaded : {len(documents)}")
    print(f"Paper pages      : {len(paper_documents)}")
    print(f"Chunks created   : {len(chunks)}")
    print(f"Embedding size   : {len(test_embedding)}")
    print(f"Retrieved chunks : {len(retrieved_documents)}")
    print(f"QA sources       : {len(qa_result['sources'])}")
    print(f"Summary sources  : {len(summary_result['sources'])}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Optional live Groq test
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not bool(
        __import__("os").getenv("RUN_LIVE_GROQ_E2E")
    ),
    reason=(
        "Live Groq E2E test disabled. "
        "Set RUN_LIVE_GROQ_E2E=1 to enable it."
    ),
)
def test_live_groq_pipeline():
    """
    Optional real-Groq integration test.

    This test intentionally uses the real Groq API and therefore consumes
    Groq quota.

    Enable with:

        PowerShell:
        $env:RUN_LIVE_GROQ_E2E="1"
        pytest -s tests/test_end_to_end.py::test_live_groq_pipeline

    Do NOT run this test repeatedly.
    """

    documents = get_sample_documents()

    paper_documents = get_first_paper_documents(
        documents
    )

    assert paper_documents

    # -----------------------------------------------------------------------
    # Real summarization through Groq
    # -----------------------------------------------------------------------

    summary_result = summarize_documents(
        paper_documents
    )

    assert summary_result
    assert isinstance(summary_result, dict)

    assert summary_result.get("summary")
    assert summary_result.get("sources")

    # -----------------------------------------------------------------------
    # Real RAG pipeline
    # -----------------------------------------------------------------------

    chunks = split_documents(
        paper_documents,
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
    )

    assert chunks

    embedding_model = get_embedding_model()

    vector_store = create_vector_store(
        chunks=chunks,
        embedding_model=embedding_model,
    )

    retriever = get_retriever(
        vector_store=vector_store,
        top_k=DEFAULT_TOP_K,
    )

    question = (
        "What is the main research problem addressed by the paper?"
    )

    qa_result = answer_question(
        retriever=retriever,
        question=question,
    )

    assert qa_result
    assert qa_result.get("answer")
    assert qa_result.get("sources")

    print("\n" + "=" * 60)
    print("LIVE GROQ END-TO-END PIPELINE PASSED")
    print("=" * 60)
    print("\nQA ANSWER:\n")
    print(qa_result["answer"])
    print("\nSUMMARY:\n")
    print(summary_result["summary"])
    print("\n" + "=" * 60)