from pathlib import Path
from typing import List
from langchain_community.document_loaders import (
    PyPDFLoader, Docx2txtLoader
)
from langchain_core.documents import Document

from app.utils.logger import get_logger

logger = get_logger(__name__)

def load_files(data_dir: str) -> List[Document]:
    """
    Load all documents from the specified directory and convert them into LangChain document structure.
    Supported files include PDF, DOCX.
    """
    data_path = Path(data_dir).resolve()

    if not data_path.exists():
        logger.error("Data directory does not exist: %s", data_path)
        raise FileNotFoundError(f"Data directory not found: {data_path}")

    if not data_path.is_dir():
        logger.error("Provided path is not a directory: %s", data_path)
        raise NotADirectoryError(f"Not a directory: {data_path}")

    logger.debug(f"Data Path: {data_path}")

    documents = []

    # PDF files
    pdf_files = list(data_path.glob("**/*.pdf"))

    logger.debug(f"Found {len(pdf_files)} PDF files: {[str(file) for file in pdf_files]}")

    for file in pdf_files:
        logger.debug(f"Loading pdf: {file}")
        try:
            loader = PyPDFLoader(str(file))
            loaded = loader.load()
            logger.debug(f"Loaded {len(loaded)} pages from {file}")
            documents.extend(loaded)
        except Exception as e:
            logger.exception(f"Failed to load pdf {file}: {e}")

    # DOCX files
    docx_files = list(data_path.glob("**/*.docx"))
    logger.debug(f"Found {len(docx_files)} DOCX files: {[str(file) for file in docx_files]}")
    for file in docx_files:
        try:
            loader = Docx2txtLoader(str(file))
            loaded = loader.load()
            logger.debug(f"Loaded {len(loaded)} docs from {file}")
            documents.extend(loaded)
        except Exception as e:
            logger.exception(f"Failed to load docx {file}: {e}")

    logger.info("Document loading completed. Total documents loaded: %d", len(documents))
    return documents