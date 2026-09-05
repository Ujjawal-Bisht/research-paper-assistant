import json
from typing import Any, Dict, List

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from app.config.config import QA_PROMPT_FILE
from app.rag.retriever import retrieve_documents
from app.services.llm_service import (
    generate_response,
    get_groq_client,
)
from app.utils.logger import get_logger


logger = get_logger(__name__)


def load_qa_prompt() -> Dict[str, Any]:
    """
    Load the question-answering prompt configuration
    from qa_prompt.json.
    """

    logger.info(
        "Loading QA prompt from: %s",
        QA_PROMPT_FILE,
    )

    if not QA_PROMPT_FILE.exists():
        logger.error(
            "QA prompt file not found: %s",
            QA_PROMPT_FILE,
        )

        raise FileNotFoundError(
            f"QA prompt file not found: {QA_PROMPT_FILE}"
        )

    try:
        with open(
            QA_PROMPT_FILE,
            "r",
            encoding="utf-8",
        ) as file:
            prompt_config = json.load(file)

    except json.JSONDecodeError as exc:
        logger.exception(
            "Invalid JSON in QA prompt file."
        )

        raise ValueError(
            "qa_prompt.json contains invalid JSON."
        ) from exc

    except Exception:
        logger.exception(
            "Failed to load QA prompt configuration."
        )
        raise

    logger.info(
        "QA prompt loaded successfully."
    )

    return prompt_config


def format_context(documents: List[Document]) -> str:
    """
    Convert retrieved documents into a formatted context string
    for the LLM.
    """

    if not documents:
        logger.warning(
            "No documents available to build context."
        )

        return ""

    context_parts = []

    for index, document in enumerate(
        documents,
        start=1,
    ):
        source = document.metadata.get(
            "source",
            "Unknown source",
        )

        page = document.metadata.get(
            "page",
            "Unknown page",
        )

        # PyPDFLoader commonly uses zero-based page numbers.
        # Convert them to human-readable page numbers.
        if isinstance(page, int):
            page = page + 1

        context_parts.append(
            f"[Source {index}]\n"
            f"Document: {source}\n"
            f"Page: {page}\n"
            f"Content:\n"
            f"{document.page_content.strip()}"
        )

    context = "\n\n".join(context_parts)

    logger.info(
        "Formatted %d documents into context.",
        len(documents),
    )

    logger.debug(
        "Context length: %d characters.",
        len(context),
    )

    return context


def build_qa_prompt(
    question: str,
    context: str,
    prompt_config: Dict[str, Any],
) -> str:
    """
    Build the final user prompt using the QA prompt
    configuration and retrieved context.
    """

    if not question or not question.strip():
        raise ValueError(
            "Question cannot be empty."
        )

    if not context or not context.strip():
        raise ValueError(
            "Context cannot be empty."
        )

    instructions = prompt_config.get(
        "instructions",
        [],
    )

    response_requirements = prompt_config.get(
        "response_requirements",
        {},
    )

    prompt_template = prompt_config.get(
        "prompt_template",
        {},
    )

    template_context = prompt_template.get(
        "context",
        "{context}",
    )

    template_question = prompt_template.get(
        "question",
        "{question}",
    )

    instruction_text = ""

    if instructions:
        instruction_text = (
            "INSTRUCTIONS:\n"
            + "\n".join(
                f"- {instruction}"
                for instruction in instructions
            )
        )

    requirements_text = ""

    if response_requirements:
        requirements_text = (
            "\n\nRESPONSE REQUIREMENTS:\n"
            + "\n".join(
                f"- {key}: {value}"
                for key, value in response_requirements.items()
            )
        )

    formatted_context = template_context.replace(
        "{context}",
        context,
    )

    formatted_question = template_question.replace(
        "{question}",
        question.strip(),
    )

    final_prompt = (
        f"{instruction_text}"
        f"{requirements_text}"
        f"\n\n"
        f"RESEARCH PAPER CONTEXT:\n"
        f"{formatted_context}"
        f"\n\n"
        f"USER QUESTION:\n"
        f"{formatted_question}"
    )

    logger.info(
        "QA prompt constructed successfully."
    )

    logger.debug(
        "Final QA prompt length: %d characters.",
        len(final_prompt),
    )

    return final_prompt


def extract_sources(
    documents: List[Document],
) -> List[Dict[str, Any]]:
    """
    Extract unique source information from retrieved documents.
    """

    sources = []

    seen_sources = set()

    for document in documents:
        source = document.metadata.get(
            "source",
            "Unknown source",
        )

        page = document.metadata.get(
            "page",
            None,
        )

        if isinstance(page, int):
            page = page + 1

        source_key = (
            str(source),
            page,
        )

        if source_key in seen_sources:
            continue

        seen_sources.add(source_key)

        sources.append(
            {
                "source": source,
                "page": page,
            }
        )

    logger.info(
        "Extracted %d unique sources.",
        len(sources),
    )

    return sources


def answer_question(
    retriever: BaseRetriever,
    question: str,
) -> Dict[str, Any]:
    """
    Execute the complete RAG question-answering pipeline.

    Pipeline:
        Question
        -> Retriever
        -> Relevant documents
        -> Context
        -> QA prompt
        -> Groq LLM
        -> Answer + sources
    """

    if retriever is None:
        raise ValueError(
            "Retriever cannot be None."
        )

    if not question or not question.strip():
        raise ValueError(
            "Question cannot be empty."
        )

    question = question.strip()

    logger.info(
        "Processing research question: %s",
        question,
    )

    try:
        # --------------------------------------------------------
        # Step 1: Retrieve relevant documents
        # --------------------------------------------------------

        documents = retrieve_documents(
            retriever,
            question,
        )

        if not documents:
            logger.warning(
                "No relevant documents found for question."
            )

            return {
                "answer": (
                    "I could not find relevant information "
                    "in the uploaded research papers."
                ),
                "sources": [],
            }

        # --------------------------------------------------------
        # Step 2: Build context
        # --------------------------------------------------------

        context = format_context(
            documents
        )

        # --------------------------------------------------------
        # Step 3: Load QA prompt
        # --------------------------------------------------------

        prompt_config = load_qa_prompt()

        # --------------------------------------------------------
        # Step 4: Build final prompt
        # --------------------------------------------------------

        qa_prompt = build_qa_prompt(
            question=question,
            context=context,
            prompt_config=prompt_config,
        )

        # --------------------------------------------------------
        # Step 5: Initialize Groq client
        # --------------------------------------------------------

        client = get_groq_client()

        # --------------------------------------------------------
        # Step 6: Generate answer
        # --------------------------------------------------------

        answer = generate_response(
            client=client,
            user_prompt=qa_prompt,
        )

        # --------------------------------------------------------
        # Step 7: Extract sources
        # --------------------------------------------------------

        sources = extract_sources(
            documents
        )

        logger.info(
            "Question answered successfully."
        )

        return {
            "answer": answer,
            "sources": sources,
        }

    except Exception:
        logger.exception(
            "Failed to answer research question."
        )
        raise