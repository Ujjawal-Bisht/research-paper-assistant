from langchain_core.documents import Document

from app.config.config import (
    SUMMARY_BATCH_CHAR_LIMIT,
    SUMMARY_SYNTHESIS_CHAR_LIMIT,
)
from app.ingestion.document_loader import load_files
from app.services.summarizer import (
    build_final_summary_prompt,
    build_intermediate_summary_prompt,
    create_summary_batches,
    extract_sources,
    format_document_context,
    load_summary_prompt,
    summarize_documents,
    synthesize_summaries,
)


DATA_DIR = "data/sample_papers"


# ============================================================
# Prompt Loading Tests
# ============================================================

def test_load_summary_prompt():
    prompt_config = load_summary_prompt()

    assert isinstance(
        prompt_config,
        dict,
    )

    assert "intermediate_summary" in prompt_config

    assert "final_summary" in prompt_config

    assert "instructions" in (
        prompt_config["intermediate_summary"]
    )

    assert "instructions" in (
        prompt_config["final_summary"]
    )


# ============================================================
# Document Formatting Tests
# ============================================================

def test_format_document_context():
    documents = [
        Document(
            page_content=(
                "This is the first page of the "
                "research paper."
            ),
            metadata={
                "source": "sample.pdf",
                "page": 0,
            },
        ),
        Document(
            page_content=(
                "This is the second page of the "
                "research paper."
            ),
            metadata={
                "source": "sample.pdf",
                "page": 1,
            },
        ),
    ]

    context = format_document_context(
        documents
    )

    assert context

    assert "sample.pdf" in context

    assert "Page: 1" in context

    assert "Page: 2" in context

    assert (
        "This is the first page"
        in context
    )

    assert (
        "This is the second page"
        in context
    )


def test_format_document_context_empty():
    context = format_document_context(
        []
    )

    assert context == ""


# ============================================================
# Summary Batch Tests
# ============================================================

def test_create_summary_batches():
    documents = [
        Document(
            page_content=(
                "A " * 100
            ),
            metadata={
                "source": "sample.pdf",
                "page": 0,
            },
        ),
        Document(
            page_content=(
                "B " * 100
            ),
            metadata={
                "source": "sample.pdf",
                "page": 1,
            },
        ),
        Document(
            page_content=(
                "C " * 100
            ),
            metadata={
                "source": "sample.pdf",
                "page": 2,
            },
        ),
    ]

    batches = create_summary_batches(
        documents,
        max_characters=250,
    )

    assert batches

    assert len(batches) > 1

    for batch in batches:
        assert batch.strip()

        assert len(batch) <= 250


def test_create_summary_batches_empty():
    try:
        create_summary_batches(
            [],
            max_characters=100,
        )

        assert False, (
            "Expected ValueError was not raised."
        )

    except ValueError as exc:
        assert (
            "Documents cannot be empty"
            in str(exc)
        )

def test_create_summary_batches_invalid_limit():
    documents = [
        Document(
            page_content="Sample content.",
            metadata={
                "source": "sample.pdf",
                "page": 0,
            },
        )
    ]

    try:
        create_summary_batches(
            documents,
            max_characters=0,
        )

        assert False, (
            "Expected ValueError was not raised."
        )

    except ValueError as exc:
        assert (
            "max_characters"
            in str(exc)
        )


def test_summary_batch_limit_configuration():
    assert SUMMARY_BATCH_CHAR_LIMIT > 0

    assert (
        SUMMARY_SYNTHESIS_CHAR_LIMIT
        > 1
    )


# ============================================================
# Intermediate Prompt Tests
# ============================================================

def test_build_intermediate_summary_prompt():
    prompt_config = load_summary_prompt()

    context = (
        "The paper proposes a machine-learning "
        "method for classification."
    )

    prompt = build_intermediate_summary_prompt(
        context=context,
        prompt_config=prompt_config,
    )

    assert prompt

    assert (
        "RESEARCH PAPER SECTION"
        in prompt
    )

    assert (
        "machine-learning method"
        in prompt
    )

    assert (
        "INSTRUCTIONS"
        in prompt
    )


def test_build_intermediate_summary_prompt_empty_context():
    prompt_config = load_summary_prompt()

    try:
        build_intermediate_summary_prompt(
            context="",
            prompt_config=prompt_config,
        )

        assert False, (
            "Expected ValueError was not raised."
        )

    except ValueError as exc:
        assert (
            "Context cannot be empty"
            in str(exc)
        )


# ============================================================
# Final Synthesis Prompt Tests
# ============================================================

def test_build_final_summary_prompt():
    prompt_config = load_summary_prompt()

    summaries = [
        (
            "The paper introduces a new "
            "classification method."
        ),
        (
            "The experiments demonstrate "
            "improved classification performance."
        ),
    ]

    prompt = build_final_summary_prompt(
        summaries=summaries,
        prompt_config=prompt_config,
    )

    assert prompt

    assert (
        "INTERMEDIATE SUMMARIES"
        in prompt
    )

    assert (
        "classification method"
        in prompt
    )

    assert (
        "improved classification"
        in prompt
    )

    assert (
        "INSTRUCTIONS"
        in prompt
    )


def test_build_final_summary_prompt_empty():
    prompt_config = load_summary_prompt()

    try:
        build_final_summary_prompt(
            summaries=[],
            prompt_config=prompt_config,
        )

        assert False, (
            "Expected ValueError was not raised."
        )

    except ValueError as exc:
        assert (
            "Intermediate summaries"
            in str(exc)
        )


# ============================================================
# Synthesis Tests Without Groq
# ============================================================

def test_synthesize_single_summary():
    prompt_config = load_summary_prompt()

    summaries = [
        (
            "This is already a complete "
            "summary of the research paper."
        )
    ]

    class DummyClient:
        pass

    result = synthesize_summaries(
        summaries=summaries,
        prompt_config=prompt_config,
        client=DummyClient(),
    )

    assert result == summaries[0]


def test_synthesize_forces_progress_when_groups_do_not_shrink(monkeypatch):
    prompt_config = load_summary_prompt()

    summaries = [
        "A " * 3500,
        "B " * 3500,
    ]

    calls = []

    def fake_generate_response(client, user_prompt):
        calls.append(user_prompt)
        return "Final synthesized summary."

    monkeypatch.setattr(
        "app.services.summarizer.generate_response",
        fake_generate_response,
    )

    result = synthesize_summaries(
        summaries=summaries,
        prompt_config=prompt_config,
        client=object(),
        max_characters=5000,
    )

    assert result == "Final synthesized summary."
    assert len(calls) == 1


# ============================================================
# Source Extraction Tests
# ============================================================

def test_extract_sources():
    documents = [
        Document(
            page_content="First page.",
            metadata={
                "source": "paper.pdf",
                "page": 0,
            },
        ),
        Document(
            page_content="Second page.",
            metadata={
                "source": "paper.pdf",
                "page": 1,
            },
        ),
        Document(
            page_content="Duplicate first page.",
            metadata={
                "source": "paper.pdf",
                "page": 0,
            },
        ),
    ]

    sources = extract_sources(
        documents
    )

    assert len(sources) == 2

    assert {
        "source": "paper.pdf",
        "page": 1,
    } in sources

    assert {
        "source": "paper.pdf",
        "page": 2,
    } in sources


def test_extract_sources_empty():
    sources = extract_sources(
        []
    )

    assert sources == []


# ============================================================
# Single-Paper Integration Test
# ============================================================

def test_summarize_documents(monkeypatch):
    """
    Integration test for the complete single-paper
    hierarchical summarization pipeline.

    This test deliberately selects ONE PDF from the sample
    paper directory. The application itself is designed to
    summarize one paper at a time.
    """

    documents = load_files(
        DATA_DIR
    )

    assert documents

    # --------------------------------------------------------
    # The sample directory may contain multiple papers.
    #
    # Select documents belonging to exactly one paper.
    # --------------------------------------------------------

    first_source = documents[0].metadata.get(
        "source"
    )

    single_paper_documents = [
        document
        for document in documents
        if document.metadata.get(
            "source"
        ) == first_source
    ]

    assert single_paper_documents

    monkeypatch.setattr(
        "app.services.summarizer.get_groq_client",
        lambda: object(),
    )

    monkeypatch.setattr(
        "app.services.summarizer.generate_response",
        lambda client, user_prompt: (
            "Generated summary for the supplied research-paper content."
        ),
    )

    result = summarize_documents(
        single_paper_documents
    )

    assert isinstance(
        result,
        dict,
    )

    assert "summary" in result

    assert "sources" in result

    assert result["summary"]

    assert isinstance(
        result["summary"],
        str,
    )

    assert result["sources"]

    assert isinstance(
        result["sources"],
        list,
    )

    print(
        "\n"
        "==================================================\n"
        "FINAL RESEARCH PAPER SUMMARY\n"
        "==================================================\n"
    )

    print(
        result["summary"]
    )

    print(
        "\n"
        "==================================================\n"
        "SOURCES\n"
        "==================================================\n"
    )

    for source in result["sources"]:
        print(source)


# ============================================================
# Manual Test Entry Point
# ============================================================

if __name__ == "__main__":
    test_load_summary_prompt()

    test_format_document_context()

    test_create_summary_batches()

    test_build_intermediate_summary_prompt()

    test_build_final_summary_prompt()

    test_synthesize_single_summary()

    test_extract_sources()

    print(
        "All summarizer unit tests passed."
    )