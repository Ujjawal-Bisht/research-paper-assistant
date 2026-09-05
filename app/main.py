import tempfile
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st
from langchain_core.documents import Document

from app.services.pipeline_service import ResearchPaperPipeline
from app.utils.logger import get_logger


logger = get_logger(__name__)


APP_TITLE = "Research Paper Assistant"
APP_DESCRIPTION = (
    "Upload research papers, generate summaries, and ask questions "
    "using retrieval-augmented generation."
)

SUPPORTED_FILE_TYPES = ["pdf", "docx"]


def initialize_session_state() -> None:
    """
    Initialize Streamlit session-state variables.

    Session state allows the pipeline and generated results to survive
    Streamlit reruns caused by user interactions.
    """
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = None

    if "uploaded_file_names" not in st.session_state:
        st.session_state.uploaded_file_names = []

    if "documents" not in st.session_state:
        st.session_state.documents = []

    if "summary_result" not in st.session_state:
        st.session_state.summary_result = None

    if "qa_result" not in st.session_state:
        st.session_state.qa_result = None

    if "pipeline_built" not in st.session_state:
        st.session_state.pipeline_built = False

    if "upload_directory" not in st.session_state:
        st.session_state.upload_directory = None


def configure_page() -> None:
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def render_header() -> None:
    """Render the application header."""
    st.title(APP_TITLE)

    st.markdown(
        """
        Upload one or more research papers and use the assistant to:

        - Generate a structured research-paper summary.
        - Ask questions about the uploaded papers.
        - Retrieve relevant passages from the papers.
        - View the sources used to generate responses.
        """
    )


def create_upload_directory() -> Path:
    """
    Create a temporary directory for the current Streamlit session.

    Returns:
        Path to the temporary upload directory.
    """
    if st.session_state.upload_directory is None:
        temporary_directory = tempfile.mkdtemp(
            prefix="research_paper_assistant_"
        )

        st.session_state.upload_directory = Path(
            temporary_directory
        )

        logger.info(
            "Created temporary upload directory: %s",
            temporary_directory,
        )

    return st.session_state.upload_directory


def save_uploaded_files(uploaded_files: List[Any]) -> Path:
    """
    Save Streamlit UploadedFile objects to the session upload directory.

    Args:
        uploaded_files: Files uploaded through Streamlit.

    Returns:
        Directory containing the saved files.
    """
    upload_directory = create_upload_directory()

    for uploaded_file in uploaded_files:
        file_path = upload_directory / uploaded_file.name

        with file_path.open("wb") as output_file:
            output_file.write(uploaded_file.getbuffer())

        logger.info(
            "Saved uploaded file: %s",
            file_path,
        )

    return upload_directory


def render_sidebar() -> List[Any]:
    """
    Render the sidebar and return uploaded files.

    Returns:
        List of uploaded Streamlit files.
    """
    with st.sidebar:
        st.header("Research Papers")

        uploaded_files = st.file_uploader(
            "Upload research papers",
            type=SUPPORTED_FILE_TYPES,
            accept_multiple_files=True,
            help=(
                "Upload one or more PDF or DOCX research papers."
            ),
        )

        if uploaded_files:
            st.success(
                f"{len(uploaded_files)} file(s) selected."
            )

            for uploaded_file in uploaded_files:
                st.write(f"📄 {uploaded_file.name}")

        st.divider()

        st.subheader("Pipeline Configuration")

        chunk_size = st.number_input(
            "Chunk size",
            min_value=100,
            max_value=5000,
            value=500,
            step=100,
            help=(
                "Maximum number of characters in each text chunk."
            ),
        )

        chunk_overlap = st.number_input(
            "Chunk overlap",
            min_value=0,
            max_value=1000,
            value=100,
            step=50,
            help=(
                "Number of overlapping characters between chunks."
            ),
        )

        top_k = st.number_input(
            "Top-k retrieval",
            min_value=1,
            max_value=20,
            value=5,
            step=1,
            help=(
                "Number of relevant chunks retrieved for a question."
            ),
        )

        st.divider()

        if st.button(
            "Reset Application",
            use_container_width=True,
        ):
            reset_application()

        st.caption(
            "The current configuration is applied when the pipeline "
            "is built."
        )

    st.session_state.pipeline_config = {
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "top_k": top_k,
    }

    return uploaded_files


def reset_application() -> None:
    """
    Reset the Streamlit application state.

    This clears the in-memory pipeline and generated results.
    """
    logger.info("Resetting Streamlit application state.")

    keys_to_remove = [
        "pipeline",
        "uploaded_file_names",
        "documents",
        "summary_result",
        "qa_result",
        "pipeline_built",
        "upload_directory",
        "pipeline_config",
    ]

    for key in keys_to_remove:
        st.session_state.pop(key, None)

    st.rerun()


def build_pipeline(uploaded_files: List[Any]) -> None:
    """
    Save uploaded files and build the research-paper pipeline.

    Args:
        uploaded_files: Files uploaded through Streamlit.
    """
    if not uploaded_files:
        st.warning(
            "Please upload at least one research paper first."
        )
        return

    configuration = st.session_state.pipeline_config

    if configuration["chunk_overlap"] >= configuration["chunk_size"]:
        st.error(
            "Chunk overlap must be smaller than chunk size."
        )
        return

    current_file_names = [
        uploaded_file.name
        for uploaded_file in uploaded_files
    ]

    with st.spinner(
        "Processing research papers and building the RAG pipeline..."
    ):
        try:
            upload_directory = save_uploaded_files(
                uploaded_files
            )

            pipeline = ResearchPaperPipeline(
                chunk_size=configuration["chunk_size"],
                chunk_overlap=configuration["chunk_overlap"],
                top_k=configuration["top_k"],
            )

            pipeline_info = pipeline.build_pipeline(
                data_dir=str(upload_directory),
                save_vector_store_to_disk=False,
            )

            st.session_state.pipeline = pipeline
            st.session_state.documents = pipeline_info["documents"]
            st.session_state.uploaded_file_names = current_file_names
            st.session_state.pipeline_built = True

            st.session_state.summary_result = None
            st.session_state.qa_result = None

            logger.info(
                "Streamlit pipeline built successfully | "
                "files=%d | documents=%d | chunks=%d",
                len(current_file_names),
                pipeline_info["document_count"],
                pipeline_info["chunk_count"],
            )

            st.success(
                "Research-paper pipeline built successfully."
            )

        except Exception as error:
            logger.exception(
                "Failed to build Streamlit pipeline."
            )

            st.error(
                "Unable to process the research papers. "
                "Please check the uploaded files and try again."
            )


def render_pipeline_status() -> None:
    """Display the current pipeline status."""
    pipeline = st.session_state.pipeline

    if pipeline is None:
        return

    status = pipeline.get_status()

    st.subheader("Pipeline Status")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Pages Loaded",
            status["document_count"],
        )

    with col2:
        st.metric(
            "Chunks",
            status["chunk_count"],
        )

    with col3:
        st.metric(
            "Top-k",
            status["top_k"],
        )

    with col4:
        embedding_status = (
            "Ready"
            if status["embedding_model_initialized"]
            else "Not Ready"
        )

        st.metric(
            "Embeddings",
            embedding_status,
        )


def render_summary_section() -> None:
    """Render the research-paper summarization section."""
    pipeline = st.session_state.pipeline

    if pipeline is None:
        return

    st.subheader("Research Paper Summary")

    if st.button(
        "Generate Summary",
        type="primary",
        use_container_width=True,
    ):
        with st.spinner(
            "Generating research-paper summary..."
        ):
            try:
                result = pipeline.summarize()

                st.session_state.summary_result = result

                logger.info(
                    "Summary generated successfully."
                )

            except Exception as error:
                logger.exception(
                    "Failed to generate summary."
                )

                st.error(
                    "Unable to generate the summary right now. "
                    "Please try again later."
                )

    summary_result = st.session_state.summary_result

    if not summary_result:
        st.info(
            "Click 'Generate Summary' to create a structured "
            "summary of the uploaded research paper(s)."
        )
        return

    summary = summary_result.get("summary")

    if summary:
        st.markdown(summary)

    sources = summary_result.get("sources", [])

    if sources:
        render_sources(
            sources=sources,
            title="Summary Sources",
        )


def render_question_section() -> None:
    """Render the question-answering section."""
    pipeline = st.session_state.pipeline

    if pipeline is None:
        return

    st.subheader("Ask Questions")

    question = st.text_area(
        "Enter your question",
        placeholder=(
            "Example: What is the main research problem "
            "addressed by the paper?"
        ),
        height=100,
    )

    if st.button(
        "Ask Question",
        type="primary",
        use_container_width=True,
    ):
        if not question or not question.strip():
            st.warning(
                "Please enter a question."
            )
            return

        with st.spinner(
            "Searching the research paper and generating an answer..."
        ):
            try:
                result = pipeline.ask_question(
                    question=question,
                )

                st.session_state.qa_result = result

                logger.info(
                    "Question answered successfully."
                )

            except Exception as error:
                logger.exception(
                    "Failed to answer question."
                )

                st.error(
                    "Unable to answer the question right now. "
                    "Please try again later."
                )

    qa_result = st.session_state.qa_result

    if not qa_result:
        return

    answer = qa_result.get("answer")

    if answer:
        st.markdown("### Answer")
        st.markdown(answer)

    sources = qa_result.get("sources", [])

    if sources:
        render_sources(
            sources=sources,
            title="Retrieved Sources",
        )


def render_sources(
    sources: List[Dict[str, Any]],
    title: str,
) -> None:
    """
    Render source information.

    Args:
        sources: Source dictionaries returned by QA or summarization.
        title: Section title for the sources.
    """
    if not sources:
        return

    with st.expander(title):
        for index, source in enumerate(
            sources,
            start=1,
        ):
            source_name = source.get(
                "source",
                "Unknown source",
            )

            page = source.get(
                "page",
                "Unknown",
            )

            st.markdown(
                f"**{index}. {Path(source_name).name}** "
                f"— Page {page}"
            )


def render_empty_state() -> None:
    """Render instructions when no pipeline has been built."""
    if st.session_state.pipeline is not None:
        return

    st.info(
        """
        **Getting started**

        1. Upload one or more research papers using the sidebar.
        2. Configure chunk size, overlap, and top-k retrieval.
        3. Build the pipeline.
        4. Generate a summary or ask questions about the papers.
        """
    )


def _run_application() -> None:
    """Run the Streamlit application body."""
    configure_page()
    initialize_session_state()

    render_header()

    uploaded_files = render_sidebar()

    build_button_col, status_col = st.columns(
        [2, 1]
    )

    with build_button_col:
        if st.button(
            "Build Research Paper Pipeline",
            type="primary",
            use_container_width=True,
        ):
            build_pipeline(uploaded_files)

    with status_col:
        if st.session_state.pipeline_built:
            st.success("Pipeline Ready")
        else:
            st.warning("Pipeline Not Built")

    render_empty_state()

    if st.session_state.pipeline_built:
        st.divider()

        render_pipeline_status()

        st.divider()

        summary_col, qa_col = st.columns(
            2
        )

        with summary_col:
            render_summary_section()

        with qa_col:
            render_question_section()


def main() -> None:
    """Run the Streamlit application behind a safe error boundary."""
    try:
        _run_application()
    except Exception:
        logger.exception(
            "Unhandled Streamlit application error."
        )

        st.error(
            "The application encountered an unexpected problem. "
            "Please refresh the page and try again."
        )


if __name__ == "__main__":
    main()