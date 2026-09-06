from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

from app.services.pipeline_service import ResearchPaperPipeline


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Research Paper Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Session-state initialization
# ---------------------------------------------------------------------------

def initialize_session_state() -> None:
    """Initialize all Streamlit session-state values used by the application."""
    defaults = {
        "pipeline": None,
        "uploaded_file_names": [],
        "documents": [],
        "summary_result": None,
        "qa_result": None,
        "pipeline_built": False,
        "upload_directory": None,
        "pipeline_config": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def reset_application() -> None:
    """Reset the current application session."""
    st.session_state.pipeline = None
    st.session_state.uploaded_file_names = []
    st.session_state.documents = []
    st.session_state.summary_result = None
    st.session_state.qa_result = None
    st.session_state.pipeline_built = False
    st.session_state.upload_directory = None
    st.session_state.pipeline_config = None


def save_uploaded_files(uploaded_files: List[Any]) -> str:
    """
    Save uploaded documents to a temporary directory.

    Returns:
        Path to the temporary upload directory.
    """
    upload_directory = tempfile.mkdtemp(
        prefix="research_paper_assistant_"
    )

    upload_path = Path(upload_directory)

    for uploaded_file in uploaded_files:
        file_path = upload_path / uploaded_file.name

        with file_path.open("wb") as destination:
            destination.write(uploaded_file.getbuffer())

    return str(upload_path)


def build_pipeline(
    uploaded_files: List[Any],
    chunk_size: int,
    chunk_overlap: int,
    top_k: int,
) -> None:
    """
    Build and store the research-paper pipeline in Streamlit session state.
    """
    upload_directory = save_uploaded_files(uploaded_files)

    pipeline = ResearchPaperPipeline(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=top_k,
    )

    documents = pipeline.build_pipeline(
        data_dir=upload_directory,
        save_vector_store_to_disk=False,
    )

    st.session_state.pipeline = pipeline
    st.session_state.documents = documents
    st.session_state.upload_directory = upload_directory
    st.session_state.uploaded_file_names = [
        uploaded_file.name for uploaded_file in uploaded_files
    ]
    st.session_state.pipeline_built = True
    st.session_state.pipeline_config = {
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "top_k": top_k,
    }
    st.session_state.summary_result = None
    st.session_state.qa_result = None


def render_sources(sources: List[Dict[str, Any]]) -> None:
    """Render source information returned by the QA/summarization pipeline."""
    if not sources:
        st.info("No source metadata was returned.")
        return

    st.markdown("#### Sources")

    for index, source in enumerate(sources, start=1):
        source_name = source.get("source", "Unknown source")
        page = source.get("page", "Unknown page")

        with st.expander(f"Source {index}"):
            st.write(f"**Source:** {source_name}")
            st.write(f"**Page:** {page}")

            content = source.get("content")

            if content:
                st.write("**Retrieved content:**")
                st.write(content)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar() -> List[Any]:
    """
    Render the sidebar controls.

    Returns:
        List of uploaded files.
    """
    with st.sidebar:
        st.header("Configuration")

        uploaded_files = st.file_uploader(
            "Upload research papers",
            type=["pdf", "docx"],
            accept_multiple_files=True,
            help="Upload one or more PDF or DOCX research papers.",
        )

        st.divider()

        st.subheader("RAG Parameters")

        chunk_size = st.number_input(
            "Chunk size",
            min_value=100,
            max_value=5000,
            value=500,
            step=100,
            help="Number of characters in each document chunk.",
        )

        chunk_overlap = st.number_input(
            "Chunk overlap",
            min_value=0,
            max_value=1000,
            value=100,
            step=50,
            help="Number of overlapping characters between chunks.",
        )

        top_k = st.number_input(
            "Top-k",
            min_value=1,
            max_value=20,
            value=5,
            step=1,
            help="Number of chunks retrieved for each query.",
        )

        st.divider()

        if st.button(
            "Reset Application",
            use_container_width=True,
            type="secondary",
        ):
            reset_application()
            st.rerun()

    # Store the current configuration for the build button.
    st.session_state.current_chunk_size = int(chunk_size)
    st.session_state.current_chunk_overlap = int(chunk_overlap)
    st.session_state.current_top_k = int(top_k)

    return uploaded_files


# ---------------------------------------------------------------------------
# Pipeline status
# ---------------------------------------------------------------------------

def render_pipeline_status() -> None:
    """Render the current pipeline status."""
    st.subheader("Pipeline Status")

    pipeline = st.session_state.pipeline

    if pipeline is None or not st.session_state.pipeline_built:
        st.info("Pipeline has not been built yet.")
        return

    try:
        status = pipeline.get_status()
    except Exception:
        status = {}

    if not isinstance(status, dict):
        status = {}

    documents_count = status.get(
        "documents_count",
        len(st.session_state.documents),
    )

    chunks_count = status.get(
        "chunks_count",
        "-",
    )

    top_k = status.get(
        "top_k",
        st.session_state.pipeline_config.get("top_k", "-")
        if st.session_state.pipeline_config
        else "-",
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Documents",
            documents_count,
        )

    with col2:
        st.metric(
            "Chunks",
            chunks_count,
        )

    with col3:
        st.metric(
            "Top-k",
            top_k,
        )


# ---------------------------------------------------------------------------
# Summary section
# ---------------------------------------------------------------------------

def render_summary_section() -> None:
    """Render the research-paper summarization workflow."""
    st.subheader("Research Paper Summary")

    if not st.session_state.pipeline_built:
        st.info("Build the pipeline before generating a summary.")
        return

    if st.button(
        "Generate Summary",
        use_container_width=True,
        type="primary",
    ):
        pipeline = st.session_state.pipeline

        if pipeline is None:
            st.error("Pipeline is not available.")
            return

        with st.spinner(
            "Generating research-paper summary. "
            "This may take some time for longer papers..."
        ):
            try:
                result = pipeline.summarize()

                st.session_state.summary_result = result

            except Exception as exc:
                st.error(
                    "Failed to generate the summary. "
                    "Check the application logs for details."
                )
                st.exception(exc)
                return

    summary_result = st.session_state.summary_result

    if not summary_result:
        return

    st.success("Summary generated successfully.")

    if isinstance(summary_result, dict):
        summary_text = summary_result.get("summary")

        if summary_text:
            st.markdown("### Summary")
            st.markdown(summary_text)

        sources = summary_result.get("sources", [])

        if sources:
            render_sources(sources)

        # Display other useful result fields without breaking the UI.
        for key, value in summary_result.items():
            if key in {"summary", "sources"}:
                continue

            if value is not None:
                st.markdown(f"### {key.replace('_', ' ').title()}")
                st.write(value)

    else:
        st.markdown("### Summary")
        st.write(summary_result)


# ---------------------------------------------------------------------------
# Question-answering section
# ---------------------------------------------------------------------------

def render_qa_section() -> None:
    """Render the document question-answering workflow."""
    st.subheader("Ask Questions")

    if not st.session_state.pipeline_built:
        st.info("Build the pipeline before asking questions.")
        return

    question = st.text_area(
        "Ask a question about the uploaded paper",
        placeholder=(
            "Example: What methodology does the paper use?"
        ),
        height=120,
    )

    if st.button(
        "Ask Question",
        use_container_width=True,
        type="primary",
    ):
        if not question.strip():
            st.warning("Please enter a question.")
            return

        pipeline = st.session_state.pipeline

        if pipeline is None:
            st.error("Pipeline is not available.")
            return

        with st.spinner("Retrieving relevant sections and generating answer..."):
            try:
                result = pipeline.ask_question(
                    question.strip()
                )

                st.session_state.qa_result = result

            except Exception as exc:
                st.error(
                    "Failed to answer the question. "
                    "Check the application logs for details."
                )
                st.exception(exc)
                return

    qa_result = st.session_state.qa_result

    if not qa_result:
        return

    st.success("Answer generated successfully.")

    if isinstance(qa_result, dict):
        answer = qa_result.get("answer")

        if answer:
            st.markdown("### Answer")
            st.markdown(answer)

        sources = qa_result.get("sources", [])

        if sources:
            render_sources(sources)

        for key, value in qa_result.items():
            if key in {"answer", "sources"}:
                continue

            if value is not None:
                st.markdown(f"### {key.replace('_', ' ').title()}")
                st.write(value)

    else:
        st.markdown("### Answer")
        st.write(qa_result)


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the Streamlit application."""
    initialize_session_state()

    st.title("📚 Research Paper Assistant")

    st.markdown(
        """
        Upload research papers, build a retrieval pipeline, ask questions
        grounded in the uploaded documents, and generate structured summaries.
        """
    )

    st.divider()

    uploaded_files = render_sidebar()

    # -----------------------------------------------------------------------
    # Upload and build section
    # -----------------------------------------------------------------------

    st.subheader("Documents")

    if uploaded_files:
        st.write(
            f"**{len(uploaded_files)} file(s) selected:**"
        )

        for uploaded_file in uploaded_files:
            file_size_mb = uploaded_file.size / (1024 * 1024)

            st.write(
                f"- `{uploaded_file.name}` "
                f"({file_size_mb:.2f} MB)"
            )

        st.divider()

        chunk_size = st.session_state.current_chunk_size
        chunk_overlap = st.session_state.current_chunk_overlap
        top_k = st.session_state.current_top_k

        if st.button(
            "Build RAG Pipeline",
            use_container_width=True,
            type="primary",
        ):
            try:
                with st.spinner(
                    "Loading documents, creating embeddings, "
                    "building FAISS index, and initializing retriever..."
                ):
                    build_pipeline(
                        uploaded_files=uploaded_files,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        top_k=top_k,
                    )

                st.success(
                    "RAG pipeline built successfully."
                )

            except Exception as exc:
                st.error(
                    "Failed to build the RAG pipeline."
                )
                st.exception(exc)

    elif st.session_state.pipeline_built:
        st.success(
            "Pipeline is already built. "
            "You can continue using the current documents."
        )

    else:
        st.info(
            "Upload one or more PDF/DOCX research papers from the sidebar."
        )

    # -----------------------------------------------------------------------
    # Status
    # -----------------------------------------------------------------------

    st.divider()

    render_pipeline_status()

    # -----------------------------------------------------------------------
    # Main functionality
    # -----------------------------------------------------------------------

    if st.session_state.pipeline_built:
        st.divider()

        summary_column, qa_column = st.columns(
            2,
            gap="large",
        )

        with summary_column:
            render_summary_section()

        with qa_column:
            render_qa_section()


# ---------------------------------------------------------------------------
# Local execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()