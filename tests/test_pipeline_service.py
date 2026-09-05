from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from app.services.pipeline_service import ResearchPaperPipeline


def create_sample_documents():
    """Create sample page-level documents for testing."""
    return [
        Document(
            page_content="This is the first page of the research paper.",
            metadata={
                "source": "paper_1.pdf",
                "page": 0,
            },
        ),
        Document(
            page_content="This is the second page of the research paper.",
            metadata={
                "source": "paper_1.pdf",
                "page": 1,
            },
        ),
    ]


def create_sample_chunks():
    """Create sample chunks for testing."""
    return [
        Document(
            page_content="This is the first research paper chunk.",
            metadata={
                "source": "paper_1.pdf",
                "page": 0,
            },
        ),
        Document(
            page_content="This is the second research paper chunk.",
            metadata={
                "source": "paper_1.pdf",
                "page": 1,
            },
        ),
    ]


@pytest.fixture
def pipeline(tmp_path):
    """Create a pipeline instance using a temporary vector-store path."""
    return ResearchPaperPipeline(
        chunk_size=500,
        chunk_overlap=100,
        top_k=5,
        vector_store_path=str(tmp_path / "vector_store"),
    )


def test_pipeline_initialization(pipeline):
    """Test that the pipeline initializes with the expected defaults."""
    assert pipeline.chunk_size == 500
    assert pipeline.chunk_overlap == 100
    assert pipeline.top_k == 5

    assert pipeline.embedding_model is None
    assert pipeline.vector_store is None
    assert pipeline.retriever is None

    assert pipeline.documents == []
    assert pipeline.chunks == []


def test_pipeline_rejects_invalid_chunk_size(tmp_path):
    """Test that an invalid chunk size raises ValueError."""
    with pytest.raises(ValueError, match="chunk_size must be greater than 0"):
        ResearchPaperPipeline(
            chunk_size=0,
            vector_store_path=str(tmp_path / "vector_store"),
        )


def test_pipeline_rejects_negative_chunk_overlap(tmp_path):
    """Test that a negative chunk overlap raises ValueError."""
    with pytest.raises(
        ValueError,
        match="chunk_overlap cannot be negative",
    ):
        ResearchPaperPipeline(
            chunk_size=500,
            chunk_overlap=-1,
            vector_store_path=str(tmp_path / "vector_store"),
        )


def test_pipeline_rejects_invalid_chunk_overlap(tmp_path):
    """Test that chunk overlap must be smaller than chunk size."""
    with pytest.raises(
        ValueError,
        match="chunk_overlap must be smaller than chunk_size",
    ):
        ResearchPaperPipeline(
            chunk_size=500,
            chunk_overlap=500,
            vector_store_path=str(tmp_path / "vector_store"),
        )


def test_pipeline_rejects_invalid_top_k(tmp_path):
    """Test that an invalid top_k raises ValueError."""
    with pytest.raises(
        ValueError,
        match="top_k must be greater than 0",
    ):
        ResearchPaperPipeline(
            top_k=0,
            vector_store_path=str(tmp_path / "vector_store"),
        )


def test_pipeline_rejects_empty_vector_store_path(tmp_path):
    """Test that an empty vector-store path raises ValueError."""
    with pytest.raises(
        ValueError,
        match="vector_store_path must not be empty",
    ):
        ResearchPaperPipeline(
            vector_store_path="",
        )


def test_load_documents(pipeline, monkeypatch, tmp_path):
    """Test document loading and state update."""
    sample_documents = create_sample_documents()

    monkeypatch.setattr(
        "app.services.pipeline_service.load_files",
        lambda data_dir: sample_documents,
    )

    data_dir = tmp_path / "papers"
    data_dir.mkdir()

    documents = pipeline.load_documents(str(data_dir))

    assert documents == sample_documents
    assert pipeline.documents == sample_documents
    assert len(pipeline.documents) == 2


def test_load_documents_rejects_missing_directory(
    pipeline,
    tmp_path,
):
    """Test that a missing document directory raises FileNotFoundError."""
    missing_directory = tmp_path / "does_not_exist"

    with pytest.raises(
        FileNotFoundError,
        match="Document directory does not exist",
    ):
        pipeline.load_documents(str(missing_directory))


def test_load_documents_rejects_file_path(
    pipeline,
    tmp_path,
):
    """Test that a file path cannot be used as the document directory."""
    file_path = tmp_path / "paper.pdf"
    file_path.write_text("test")

    with pytest.raises(
        NotADirectoryError,
        match="Expected a directory",
    ):
        pipeline.load_documents(str(file_path))


def test_load_documents_rejects_empty_path(pipeline):
    """Test that an empty document directory path raises ValueError."""
    with pytest.raises(
        ValueError,
        match="data_dir must not be empty",
    ):
        pipeline.load_documents("")


def test_load_documents_handles_empty_result(
    pipeline,
    monkeypatch,
    tmp_path,
):
    """Test behavior when the loader returns no documents."""
    monkeypatch.setattr(
        "app.services.pipeline_service.load_files",
        lambda data_dir: [],
    )

    data_dir = tmp_path / "papers"
    data_dir.mkdir()

    documents = pipeline.load_documents(str(data_dir))

    assert documents == []
    assert pipeline.documents == []


def test_split_documents(pipeline, monkeypatch):
    """Test document splitting and state update."""
    sample_documents = create_sample_documents()
    sample_chunks = create_sample_chunks()

    pipeline.documents = sample_documents

    captured_arguments = {}

    def mock_split_documents(
        documents,
        chunk_size,
        chunk_overlap,
    ):
        captured_arguments["documents"] = documents
        captured_arguments["chunk_size"] = chunk_size
        captured_arguments["chunk_overlap"] = chunk_overlap

        return sample_chunks

    monkeypatch.setattr(
        "app.services.pipeline_service.split_documents",
        mock_split_documents,
    )

    chunks = pipeline.split_documents()

    assert chunks == sample_chunks
    assert pipeline.chunks == sample_chunks

    assert captured_arguments["documents"] == sample_documents
    assert captured_arguments["chunk_size"] == 500
    assert captured_arguments["chunk_overlap"] == 100


def test_split_documents_accepts_explicit_documents(
    pipeline,
    monkeypatch,
):
    """Test splitting explicitly supplied documents."""
    sample_documents = create_sample_documents()
    sample_chunks = create_sample_chunks()

    def mock_split_documents(
        documents,
        chunk_size,
        chunk_overlap,
    ):
        assert documents == sample_documents
        assert chunk_size == 500
        assert chunk_overlap == 100

        return sample_chunks

    monkeypatch.setattr(
        "app.services.pipeline_service.split_documents",
        mock_split_documents,
    )

    chunks = pipeline.split_documents(
        documents=sample_documents,
    )

    assert chunks == sample_chunks
    assert pipeline.chunks == sample_chunks


def test_split_documents_requires_documents(pipeline):
    """Test that splitting without documents raises ValueError."""
    with pytest.raises(
        ValueError,
        match="No documents available for splitting",
    ):
        pipeline.split_documents()


def test_split_documents_rejects_empty_chunks(
    pipeline,
    monkeypatch,
):
    """Test that an empty splitting result raises ValueError."""
    pipeline.documents = create_sample_documents()

    monkeypatch.setattr(
        "app.services.pipeline_service.split_documents",
        lambda documents, chunk_size, chunk_overlap: [],
    )

    with pytest.raises(
        ValueError,
        match="Document splitting produced no chunks",
    ):
        pipeline.split_documents()


def test_initialize_embeddings(pipeline, monkeypatch):
    """Test embedding-model initialization."""
    mock_embedding_model = MagicMock()

    monkeypatch.setattr(
        "app.services.pipeline_service.get_embedding_model",
        lambda: mock_embedding_model,
    )

    embedding_model = pipeline.initialize_embeddings()

    assert embedding_model is mock_embedding_model
    assert pipeline.embedding_model is mock_embedding_model


def test_initialize_embeddings_reuses_existing_model(
    pipeline,
    monkeypatch,
):
    """Test that an existing embedding model is reused."""
    existing_embedding_model = MagicMock()
    pipeline.embedding_model = existing_embedding_model

    mock_get_embedding_model = MagicMock()

    monkeypatch.setattr(
        "app.services.pipeline_service.get_embedding_model",
        mock_get_embedding_model,
    )

    embedding_model = pipeline.initialize_embeddings()

    assert embedding_model is existing_embedding_model
    mock_get_embedding_model.assert_not_called()


def test_build_vector_store(pipeline, monkeypatch):
    """Test vector-store creation and persistence."""
    sample_chunks = create_sample_chunks()
    mock_embedding_model = MagicMock()
    mock_vector_store = MagicMock()

    pipeline.chunks = sample_chunks
    pipeline.embedding_model = mock_embedding_model

    create_vector_store_mock = MagicMock(
        return_value=mock_vector_store
    )
    save_vector_store_mock = MagicMock()

    monkeypatch.setattr(
        "app.services.pipeline_service.create_vector_store",
        create_vector_store_mock,
    )
    monkeypatch.setattr(
        "app.services.pipeline_service.save_vector_store",
        save_vector_store_mock,
    )

    vector_store = pipeline.build_vector_store()

    assert vector_store is mock_vector_store
    assert pipeline.vector_store is mock_vector_store

    create_vector_store_mock.assert_called_once_with(
        chunks=sample_chunks,
        embedding_model=mock_embedding_model,
    )

    save_vector_store_mock.assert_called_once_with(
        vector_store=mock_vector_store,
        store_path=pipeline.vector_store_path,
    )


def test_build_vector_store_without_saving(
    pipeline,
    monkeypatch,
):
    """Test vector-store creation without persistence."""
    sample_chunks = create_sample_chunks()
    mock_embedding_model = MagicMock()
    mock_vector_store = MagicMock()

    pipeline.chunks = sample_chunks
    pipeline.embedding_model = mock_embedding_model

    create_vector_store_mock = MagicMock(
        return_value=mock_vector_store
    )
    save_vector_store_mock = MagicMock()

    monkeypatch.setattr(
        "app.services.pipeline_service.create_vector_store",
        create_vector_store_mock,
    )
    monkeypatch.setattr(
        "app.services.pipeline_service.save_vector_store",
        save_vector_store_mock,
    )

    vector_store = pipeline.build_vector_store(
        save=False,
    )

    assert vector_store is mock_vector_store
    assert pipeline.vector_store is mock_vector_store

    create_vector_store_mock.assert_called_once()
    save_vector_store_mock.assert_not_called()


def test_build_vector_store_initializes_embeddings_when_needed(
    pipeline,
    monkeypatch,
):
    """Test automatic embedding initialization."""
    sample_chunks = create_sample_chunks()
    mock_embedding_model = MagicMock()
    mock_vector_store = MagicMock()

    pipeline.chunks = sample_chunks

    monkeypatch.setattr(
        "app.services.pipeline_service.get_embedding_model",
        lambda: mock_embedding_model,
    )

    monkeypatch.setattr(
        "app.services.pipeline_service.create_vector_store",
        lambda chunks, embedding_model: mock_vector_store,
    )

    monkeypatch.setattr(
        "app.services.pipeline_service.save_vector_store",
        lambda vector_store, store_path: None,
    )

    vector_store = pipeline.build_vector_store()

    assert vector_store is mock_vector_store
    assert pipeline.embedding_model is mock_embedding_model


def test_build_vector_store_requires_chunks(pipeline):
    """Test that vector-store creation requires chunks."""
    with pytest.raises(
        ValueError,
        match="No chunks available for vector store creation",
    ):
        pipeline.build_vector_store()


def test_load_vector_store(pipeline, monkeypatch):
    """Test loading an existing vector store."""
    mock_embedding_model = MagicMock()
    mock_vector_store = MagicMock()

    pipeline.embedding_model = mock_embedding_model

    load_vector_store_mock = MagicMock(
        return_value=mock_vector_store
    )

    monkeypatch.setattr(
        "app.services.pipeline_service.load_vector_store",
        load_vector_store_mock,
    )

    vector_store = pipeline.load_vector_store()

    assert vector_store is mock_vector_store
    assert pipeline.vector_store is mock_vector_store

    load_vector_store_mock.assert_called_once_with(
        embedding_model=mock_embedding_model,
        store_path=pipeline.vector_store_path,
    )


def test_load_vector_store_initializes_embeddings_when_needed(
    pipeline,
    monkeypatch,
):
    """Test automatic embedding initialization before loading."""
    mock_embedding_model = MagicMock()
    mock_vector_store = MagicMock()

    monkeypatch.setattr(
        "app.services.pipeline_service.get_embedding_model",
        lambda: mock_embedding_model,
    )

    monkeypatch.setattr(
        "app.services.pipeline_service.load_vector_store",
        lambda embedding_model, store_path: mock_vector_store,
    )

    vector_store = pipeline.load_vector_store()

    assert vector_store is mock_vector_store
    assert pipeline.embedding_model is mock_embedding_model
    assert pipeline.vector_store is mock_vector_store


def test_initialize_retriever(pipeline, monkeypatch):
    """Test retriever initialization."""
    mock_vector_store = MagicMock()
    mock_retriever = MagicMock()

    pipeline.vector_store = mock_vector_store

    get_retriever_mock = MagicMock(
        return_value=mock_retriever
    )

    monkeypatch.setattr(
        "app.services.pipeline_service.get_retriever",
        get_retriever_mock,
    )

    retriever = pipeline.initialize_retriever()

    assert retriever is mock_retriever
    assert pipeline.retriever is mock_retriever

    get_retriever_mock.assert_called_once_with(
        vector_store=mock_vector_store,
        top_k=5,
    )


def test_initialize_retriever_accepts_explicit_vector_store(
    pipeline,
    monkeypatch,
):
    """Test retriever initialization with an explicit vector store."""
    mock_vector_store = MagicMock()
    mock_retriever = MagicMock()

    monkeypatch.setattr(
        "app.services.pipeline_service.get_retriever",
        lambda vector_store, top_k: mock_retriever,
    )

    retriever = pipeline.initialize_retriever(
        vector_store=mock_vector_store,
    )

    assert retriever is mock_retriever
    assert pipeline.retriever is mock_retriever


def test_initialize_retriever_requires_vector_store(pipeline):
    """Test that retriever creation requires a vector store."""
    with pytest.raises(
        ValueError,
        match="No vector store available for retriever creation",
    ):
        pipeline.initialize_retriever()


def test_build_pipeline(pipeline, monkeypatch):
    """Test complete pipeline orchestration."""
    sample_documents = create_sample_documents()
    sample_chunks = create_sample_chunks()

    mock_embedding_model = MagicMock()
    mock_vector_store = MagicMock()
    mock_retriever = MagicMock()

    monkeypatch.setattr(
        pipeline,
        "load_documents",
        MagicMock(return_value=sample_documents),
    )
    monkeypatch.setattr(
        pipeline,
        "split_documents",
        MagicMock(return_value=sample_chunks),
    )
    monkeypatch.setattr(
        pipeline,
        "initialize_embeddings",
        MagicMock(return_value=mock_embedding_model),
    )
    monkeypatch.setattr(
        pipeline,
        "build_vector_store",
        MagicMock(return_value=mock_vector_store),
    )
    monkeypatch.setattr(
        pipeline,
        "initialize_retriever",
        MagicMock(return_value=mock_retriever),
    )

    result = pipeline.build_pipeline(
        data_dir="data/sample_papers",
        save_vector_store_to_disk=True,
    )

    assert result["documents"] == sample_documents
    assert result["chunks"] == sample_chunks
    assert result["embedding_model"] is mock_embedding_model
    assert result["vector_store"] is mock_vector_store
    assert result["retriever"] is mock_retriever
    assert result["document_count"] == 2
    assert result["chunk_count"] == 2
    assert result["top_k"] == 5

    pipeline.load_documents.assert_called_once_with(
        "data/sample_papers",
    )
    pipeline.split_documents.assert_called_once_with(
        sample_documents,
    )
    pipeline.initialize_embeddings.assert_called_once()
    pipeline.build_vector_store.assert_called_once_with(
        chunks=sample_chunks,
        save=True,
    )
    pipeline.initialize_retriever.assert_called_once_with(
        vector_store=mock_vector_store,
    )


def test_load_pipeline(pipeline, monkeypatch):
    """Test loading the complete persisted pipeline."""
    mock_embedding_model = MagicMock()
    mock_vector_store = MagicMock()
    mock_retriever = MagicMock()

    monkeypatch.setattr(
        pipeline,
        "initialize_embeddings",
        MagicMock(return_value=mock_embedding_model),
    )
    monkeypatch.setattr(
        pipeline,
        "load_vector_store",
        MagicMock(return_value=mock_vector_store),
    )
    monkeypatch.setattr(
        pipeline,
        "initialize_retriever",
        MagicMock(return_value=mock_retriever),
    )

    result = pipeline.load_pipeline()

    assert result["embedding_model"] is mock_embedding_model
    assert result["vector_store"] is mock_vector_store
    assert result["retriever"] is mock_retriever
    assert result["top_k"] == 5

    pipeline.initialize_embeddings.assert_called_once()
    pipeline.load_vector_store.assert_called_once()
    pipeline.initialize_retriever.assert_called_once_with(
        vector_store=mock_vector_store,
    )


def test_ask_question(pipeline, monkeypatch):
    """Test question answering through the pipeline."""
    mock_retriever = MagicMock()
    expected_result = {
        "answer": "The paper investigates the proposed method.",
        "sources": [
            {
                "source": "paper_1.pdf",
                "page": 1,
            }
        ],
    }

    pipeline.retriever = mock_retriever

    answer_question_mock = MagicMock(
        return_value=expected_result
    )

    monkeypatch.setattr(
        "app.services.pipeline_service.answer_question",
        answer_question_mock,
    )

    result = pipeline.ask_question(
        "What is the main contribution?"
    )

    assert result == expected_result

    answer_question_mock.assert_called_once_with(
        retriever=mock_retriever,
        question="What is the main contribution?",
    )


def test_ask_question_strips_whitespace(
    pipeline,
    monkeypatch,
):
    """Test that question whitespace is removed before processing."""
    mock_retriever = MagicMock()
    expected_result = {
        "answer": "Test answer.",
        "sources": [],
    }

    pipeline.retriever = mock_retriever

    answer_question_mock = MagicMock(
        return_value=expected_result
    )

    monkeypatch.setattr(
        "app.services.pipeline_service.answer_question",
        answer_question_mock,
    )

    result = pipeline.ask_question(
        "   What is the methodology?   "
    )

    assert result == expected_result

    answer_question_mock.assert_called_once_with(
        retriever=mock_retriever,
        question="What is the methodology?",
    )


def test_ask_question_rejects_empty_question(pipeline):
    """Test that an empty question raises ValueError."""
    pipeline.retriever = MagicMock()

    with pytest.raises(
        ValueError,
        match="question must not be empty",
    ):
        pipeline.ask_question("")


def test_ask_question_requires_retriever(pipeline):
    """Test that QA requires an initialized retriever."""
    with pytest.raises(
        ValueError,
        match="Retriever is not initialized",
    ):
        pipeline.ask_question(
            "What is the main contribution?"
        )


def test_summarize(pipeline, monkeypatch):
    """Test summarization through the pipeline."""
    sample_documents = create_sample_documents()

    expected_result = {
        "summary": "This is the research paper summary.",
        "sources": [
            {
                "source": "paper_1.pdf",
                "page": 1,
            }
        ],
    }

    pipeline.documents = sample_documents

    summarize_documents_mock = MagicMock(
        return_value=expected_result
    )

    monkeypatch.setattr(
        "app.services.pipeline_service.summarize_documents",
        summarize_documents_mock,
    )

    result = pipeline.summarize()

    assert result == expected_result

    summarize_documents_mock.assert_called_once_with(
        documents=sample_documents,
    )


def test_summarize_accepts_explicit_documents(
    pipeline,
    monkeypatch,
):
    """Test summarization with explicitly supplied documents."""
    sample_documents = create_sample_documents()

    expected_result = {
        "summary": "Test summary.",
        "sources": [],
    }

    summarize_documents_mock = MagicMock(
        return_value=expected_result
    )

    monkeypatch.setattr(
        "app.services.pipeline_service.summarize_documents",
        summarize_documents_mock,
    )

    result = pipeline.summarize(
        documents=sample_documents,
    )

    assert result == expected_result

    summarize_documents_mock.assert_called_once_with(
        documents=sample_documents,
    )


def test_summarize_requires_documents(pipeline):
    """Test that summarization requires documents."""
    with pytest.raises(
        ValueError,
        match="No documents available for summarization",
    ):
        pipeline.summarize()


def test_get_status_initial_state(pipeline):
    """Test pipeline status before initialization."""
    status = pipeline.get_status()

    assert status == {
        "documents_loaded": False,
        "document_count": 0,
        "chunks_created": False,
        "chunk_count": 0,
        "embedding_model_initialized": False,
        "vector_store_initialized": False,
        "retriever_initialized": False,
        "chunk_size": 500,
        "chunk_overlap": 100,
        "top_k": 5,
        "vector_store_path": pipeline.vector_store_path,
    }


def test_get_status_after_initialization(pipeline):
    """Test pipeline status after components are initialized."""
    pipeline.documents = create_sample_documents()
    pipeline.chunks = create_sample_chunks()
    pipeline.embedding_model = MagicMock()
    pipeline.vector_store = MagicMock()
    pipeline.retriever = MagicMock()

    status = pipeline.get_status()

    assert status["documents_loaded"] is True
    assert status["document_count"] == 2

    assert status["chunks_created"] is True
    assert status["chunk_count"] == 2

    assert status["embedding_model_initialized"] is True
    assert status["vector_store_initialized"] is True
    assert status["retriever_initialized"] is True

    assert status["chunk_size"] == 500
    assert status["chunk_overlap"] == 100
    assert status["top_k"] == 5


def test_reset(pipeline):
    """Test that reset clears in-memory pipeline state."""
    pipeline.documents = create_sample_documents()
    pipeline.chunks = create_sample_chunks()
    pipeline.embedding_model = MagicMock()
    pipeline.vector_store = MagicMock()
    pipeline.retriever = MagicMock()

    pipeline.reset()

    assert pipeline.documents == []
    assert pipeline.chunks == []

    assert pipeline.embedding_model is None
    assert pipeline.vector_store is None
    assert pipeline.retriever is None


def test_reset_does_not_change_configuration(pipeline):
    """Test that reset preserves pipeline configuration."""
    original_config = {
        "chunk_size": pipeline.chunk_size,
        "chunk_overlap": pipeline.chunk_overlap,
        "top_k": pipeline.top_k,
        "vector_store_path": pipeline.vector_store_path,
    }

    pipeline.documents = create_sample_documents()
    pipeline.chunks = create_sample_chunks()
    pipeline.embedding_model = MagicMock()
    pipeline.vector_store = MagicMock()
    pipeline.retriever = MagicMock()

    pipeline.reset()

    assert pipeline.chunk_size == original_config["chunk_size"]
    assert pipeline.chunk_overlap == original_config["chunk_overlap"]
    assert pipeline.top_k == original_config["top_k"]
    assert pipeline.vector_store_path == original_config["vector_store_path"]