from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore

from app.config.config import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_TOP_K,
    VECTOR_STORE_PATH,
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
from app.utils.logger import get_logger


logger = get_logger(__name__)


class ResearchPaperPipeline:
    """
    Application-level orchestration service for the Research Paper Assistant.

    This class coordinates document ingestion, text splitting, embeddings,
    FAISS vector storage, retrieval, question answering, and summarization.

    Lower-level implementation details remain inside their respective modules.
    Streamlit should interact with this service rather than directly calling
    ingestion, RAG, QA, or summarization components.
    """

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        top_k: int = DEFAULT_TOP_K,
        vector_store_path: str = VECTOR_STORE_PATH,
    ) -> None:
        """
        Initialize the research paper pipeline.

        Args:
            chunk_size: Maximum size of each text chunk.
            chunk_overlap: Number of overlapping characters between chunks.
            top_k: Number of documents returned by the retriever.
            vector_store_path: Path used to save/load the FAISS index.
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0.")

        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative.")

        if chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size."
            )

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0.")

        if not vector_store_path or not vector_store_path.strip():
            raise ValueError(
                "vector_store_path must not be empty."
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.vector_store_path = vector_store_path

        self.embedding_model = None
        self.vector_store: Optional[VectorStore] = None
        self.retriever: Optional[BaseRetriever] = None

        self.documents: List[Document] = []
        self.chunks: List[Document] = []

        logger.info(
            "ResearchPaperPipeline initialized | "
            "chunk_size=%d | chunk_overlap=%d | top_k=%d | "
            "vector_store_path=%s",
            self.chunk_size,
            self.chunk_overlap,
            self.top_k,
            self.vector_store_path,
        )

    def load_documents(self, data_dir: str) -> List[Document]:
        """
        Load research papers from a directory.

        Supported file types are determined by the existing document loader.

        Args:
            data_dir: Directory containing research papers.

        Returns:
            List of loaded page-level Documents.
        """
        if not data_dir or not data_dir.strip():
            raise ValueError("data_dir must not be empty.")

        data_path = Path(data_dir).resolve()

        if not data_path.exists():
            raise FileNotFoundError(
                f"Document directory does not exist: {data_path}"
            )

        if not data_path.is_dir():
            raise NotADirectoryError(
                f"Expected a directory but received: {data_path}"
            )

        logger.info(
            "Loading research papers from: %s",
            data_path,
        )

        try:
            documents = load_files(str(data_path))

            if not documents:
                logger.warning(
                    "No documents were loaded from: %s",
                    data_path,
                )
                self.documents = []
                return []

            self.documents = documents

            logger.info(
                "Document loading completed: %d page documents loaded.",
                len(documents),
            )

            return documents

        except Exception:
            logger.exception(
                "Failed to load documents from: %s",
                data_path,
            )
            raise

    def split_documents(
        self,
        documents: Optional[List[Document]] = None,
    ) -> List[Document]:
        """
        Split documents into chunks.

        Args:
            documents: Documents to split. If omitted, the documents
                previously loaded by this pipeline are used.

        Returns:
            List of document chunks.
        """
        if documents is None:
            documents = self.documents

        if not documents:
            raise ValueError(
                "No documents available for splitting."
            )

        logger.info(
            "Splitting %d documents | chunk_size=%d | chunk_overlap=%d",
            len(documents),
            self.chunk_size,
            self.chunk_overlap,
        )

        try:
            chunks = split_documents(
                documents=documents,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )

            if not chunks:
                raise ValueError(
                    "Document splitting produced no chunks."
                )

            self.chunks = chunks

            logger.info(
                "Document splitting completed: %d chunks created.",
                len(chunks),
            )

            return chunks

        except Exception:
            logger.exception("Failed to split documents.")
            raise

    def initialize_embeddings(self) -> Any:
        """
        Initialize the configured embedding model.

        Returns:
            Initialized embedding model.
        """
        if self.embedding_model is not None:
            logger.debug("Using existing embedding model.")
            return self.embedding_model

        logger.info("Initializing embedding model.")

        try:
            self.embedding_model = get_embedding_model()

            logger.info("Embedding model initialized successfully.")

            return self.embedding_model

        except Exception:
            logger.exception(
                "Failed to initialize embedding model."
            )
            raise

    def build_vector_store(
        self,
        chunks: Optional[List[Document]] = None,
        save: bool = True,
    ) -> VectorStore:
        """
        Create a FAISS vector store from document chunks.

        Args:
            chunks: Chunks to index. If omitted, previously created chunks
                are used.
            save: Whether to persist the vector store to disk.

        Returns:
            Created FAISS vector store.
        """
        if chunks is None:
            chunks = self.chunks

        if not chunks:
            raise ValueError(
                "No chunks available for vector store creation."
            )

        if self.embedding_model is None:
            self.initialize_embeddings()

        logger.info(
            "Creating vector store from %d chunks.",
            len(chunks),
        )

        try:
            vector_store = create_vector_store(
                chunks=chunks,
                embedding_model=self.embedding_model,
            )

            self.vector_store = vector_store

            if save:
                save_vector_store(
                    vector_store=vector_store,
                    store_path=self.vector_store_path,
                )

                logger.info(
                    "Vector store saved to: %s",
                    Path(self.vector_store_path).resolve(),
                )

            return vector_store

        except Exception:
            logger.exception(
                "Failed to create vector store."
            )
            raise

    def load_vector_store(self) -> VectorStore:
        """
        Load an existing FAISS vector store from disk.

        Returns:
            Loaded FAISS vector store.
        """
        if self.embedding_model is None:
            self.initialize_embeddings()

        logger.info(
            "Loading vector store from: %s",
            Path(self.vector_store_path).resolve(),
        )

        try:
            vector_store = load_vector_store(
                embedding_model=self.embedding_model,
                store_path=self.vector_store_path,
            )

            self.vector_store = vector_store

            logger.info(
                "Vector store loaded successfully."
            )

            return vector_store

        except Exception:
            logger.exception(
                "Failed to load vector store."
            )
            raise

    def initialize_retriever(
        self,
        vector_store: Optional[VectorStore] = None,
    ) -> BaseRetriever:
        """
        Create a retriever from the current vector store.

        Args:
            vector_store: Vector store to use. If omitted, the vector store
                previously created or loaded by this pipeline is used.

        Returns:
            Configured retriever.
        """
        if vector_store is None:
            vector_store = self.vector_store

        if vector_store is None:
            raise ValueError(
                "No vector store available for retriever creation."
            )

        logger.info(
            "Initializing retriever with top_k=%d.",
            self.top_k,
        )

        try:
            retriever = get_retriever(
                vector_store=vector_store,
                top_k=self.top_k,
            )

            self.retriever = retriever

            logger.info(
                "Retriever initialized successfully."
            )

            return retriever

        except Exception:
            logger.exception(
                "Failed to initialize retriever."
            )
            raise

    def build_pipeline(
        self,
        data_dir: str,
        save_vector_store_to_disk: bool = True,
    ) -> Dict[str, Any]:
        """
        Build the complete RAG pipeline from a directory of research papers.

        Pipeline:

            files
              ↓
            document loading
              ↓
            text splitting
              ↓
            embeddings
              ↓
            FAISS
              ↓
            retriever

        Args:
            data_dir: Directory containing research papers.
            save_vector_store_to_disk: Whether to persist the FAISS index.

        Returns:
            Dictionary containing pipeline components and metadata.
        """
        logger.info(
            "Starting complete research paper pipeline build."
        )

        documents = self.load_documents(data_dir)
        chunks = self.split_documents(documents)

        embedding_model = self.initialize_embeddings()

        vector_store = self.build_vector_store(
            chunks=chunks,
            save=save_vector_store_to_disk,
        )

        retriever = self.initialize_retriever(
            vector_store=vector_store,
        )

        pipeline_info = {
            "documents": documents,
            "chunks": chunks,
            "embedding_model": embedding_model,
            "vector_store": vector_store,
            "retriever": retriever,
            "document_count": len(documents),
            "chunk_count": len(chunks),
            "top_k": self.top_k,
        }

        logger.info(
            "Complete pipeline build finished | "
            "documents=%d | chunks=%d | top_k=%d",
            len(documents),
            len(chunks),
            self.top_k,
        )

        return pipeline_info

    def load_pipeline(self) -> Dict[str, Any]:
        """
        Load the persisted vector store and initialize its retriever.

        Returns:
            Dictionary containing the loaded vector store, retriever,
            and embedding model.
        """
        logger.info(
            "Loading existing research paper pipeline."
        )

        embedding_model = self.initialize_embeddings()

        vector_store = self.load_vector_store()

        retriever = self.initialize_retriever(
            vector_store=vector_store,
        )

        pipeline_info = {
            "embedding_model": embedding_model,
            "vector_store": vector_store,
            "retriever": retriever,
            "top_k": self.top_k,
        }

        logger.info(
            "Existing pipeline loaded successfully."
        )

        return pipeline_info

    def ask_question(
        self,
        question: str,
    ) -> Dict[str, Any]:
        """
        Answer a question using the current retriever.

        Args:
            question: User's research-paper question.

        Returns:
            Dictionary containing the answer and source information.
        """
        if not question or not question.strip():
            raise ValueError(
                "question must not be empty."
            )

        if self.retriever is None:
            raise ValueError(
                "Retriever is not initialized. "
                "Build or load the pipeline before asking questions."
            )

        logger.info(
            "Processing research question: %s",
            question.strip(),
        )

        try:
            result = answer_question(
                retriever=self.retriever,
                question=question.strip(),
            )

            logger.info(
                "Question answering completed successfully."
            )

            return result

        except Exception:
            logger.exception(
                "Failed to answer research question."
            )
            raise

    def summarize(
        self,
        documents: Optional[List[Document]] = None,
    ) -> Dict[str, Any]:
        """
        Generate a summary for research-paper documents.

        Args:
            documents: Documents to summarize. If omitted, documents
                previously loaded by this pipeline are used.

        Returns:
            Dictionary containing the summary and source information.
        """
        if documents is None:
            documents = self.documents

        if not documents:
            raise ValueError(
                "No documents available for summarization."
            )

        logger.info(
            "Starting research-paper summarization for %d documents.",
            len(documents),
        )

        try:
            result = summarize_documents(
                documents=documents,
            )

            logger.info(
                "Research-paper summarization completed successfully."
            )

            return result

        except Exception:
            logger.exception(
                "Failed to summarize research-paper documents."
            )
            raise

    def get_status(self) -> Dict[str, Any]:
        """
        Return the current state of the pipeline.

        Returns:
            Dictionary describing initialized pipeline components.
        """
        return {
            "documents_loaded": bool(self.documents),
            "document_count": len(self.documents),
            "chunks_created": bool(self.chunks),
            "chunk_count": len(self.chunks),
            "embedding_model_initialized": (
                self.embedding_model is not None
            ),
            "vector_store_initialized": (
                self.vector_store is not None
            ),
            "retriever_initialized": (
                self.retriever is not None
            ),
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "top_k": self.top_k,
            "vector_store_path": self.vector_store_path,
        }

    def reset(self) -> None:
        """
        Reset the in-memory pipeline state.

        This does not delete the persisted FAISS vector store.
        """
        logger.info("Resetting in-memory pipeline state.")

        self.embedding_model = None
        self.vector_store = None
        self.retriever = None
        self.documents = []
        self.chunks = []

        logger.info("In-memory pipeline state reset successfully.")