import json
from typing import Any, Dict, List

from groq import Groq
from langchain_core.documents import Document

from app.config.config import (
    SUMMARY_PROMPT_FILE,
    SUMMARY_BATCH_CHAR_LIMIT,
    SUMMARY_SYNTHESIS_CHAR_LIMIT,
)

from app.services.llm_service import (
    generate_response,
    get_groq_client,
)

from app.utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================
# Prompt Loading
# ============================================================

def load_summary_prompt() -> Dict[str, Any]:
    """
    Load the summarization prompt configuration from
    summary_prompt.json.
    """

    logger.info(
        "Loading summary prompt from: %s",
        SUMMARY_PROMPT_FILE,
    )

    if not SUMMARY_PROMPT_FILE.exists():
        logger.error(
            "Summary prompt file not found: %s",
            SUMMARY_PROMPT_FILE,
        )

        raise FileNotFoundError(
            f"Summary prompt file not found: "
            f"{SUMMARY_PROMPT_FILE}"
        )

    try:
        with open(
            SUMMARY_PROMPT_FILE,
            "r",
            encoding="utf-8",
        ) as file:
            prompt_config = json.load(file)

    except json.JSONDecodeError as exc:
        logger.exception(
            "Invalid JSON in summary prompt file."
        )

        raise ValueError(
            "summary_prompt.json contains invalid JSON."
        ) from exc

    except Exception:
        logger.exception(
            "Failed to load summary prompt configuration."
        )
        raise

    if not prompt_config:
        raise ValueError(
            "summary_prompt.json is empty."
        )

    logger.info(
        "Summary prompt loaded successfully."
    )

    return prompt_config


# ============================================================
# Document Formatting
# ============================================================

def format_document(
    document: Document,
    section_number: int,
) -> str:
    """
    Format a single document for summarization.
    """

    source = document.metadata.get(
        "source",
        "Unknown source",
    )

    page = document.metadata.get(
        "page",
        "Unknown page",
    )

    if isinstance(page, int):
        page = page + 1

    content = document.page_content.strip()

    if not content:
        return ""

    return (
        f"[Section {section_number}]\n"
        f"Document: {source}\n"
        f"Page: {page}\n"
        f"Content:\n"
        f"{content}"
    )


def format_document_context(
    documents: List[Document],
) -> str:
    """
    Format multiple documents into a single text context.

    This function is retained as a general formatting utility.
    Batch construction is handled separately so that large
    papers are never sent to the LLM in one request.
    """

    if not documents:
        logger.warning(
            "No documents available for formatting."
        )

        return ""

    context_parts = []

    for index, document in enumerate(
        documents,
        start=1,
    ):
        formatted_document = format_document(
            document=document,
            section_number=index,
        )

        if formatted_document:
            context_parts.append(
                formatted_document
            )

    context = "\n\n".join(
        context_parts
    )

    logger.info(
        "Formatted %d documents for summarization.",
        len(context_parts),
    )

    logger.debug(
        "Document context length: %d characters.",
        len(context),
    )

    return context


# ============================================================
# Summary Batch Construction
# ============================================================

def create_summary_batches(
    documents: List[Document],
    max_characters: int = SUMMARY_BATCH_CHAR_LIMIT,
) -> List[str]:
    """
    Divide one research paper into manageable summarization
    batches.

    Documents are added sequentially until adding another
    document would exceed the configured character limit.

    A single document larger than the limit is split into
    smaller text sections.
    """

    if not documents:
        raise ValueError(
            "Documents cannot be empty."
        )

    if max_characters <= 0:
        raise ValueError(
            "max_characters must be greater than 0."
        )

    batches: List[str] = []
    current_batch: List[str] = []
    current_length = 0

    for index, document in enumerate(
        documents,
        start=1,
    ):
        formatted_document = format_document(
            document=document,
            section_number=index,
        )

        if not formatted_document:
            continue

        document_length = len(
            formatted_document
        )

        # ----------------------------------------------------
        # Normal case:
        # The document can fit inside the configured batch.
        # ----------------------------------------------------

        if document_length <= max_characters:

            if (
                current_batch
                and current_length
                + document_length
                + 2
                > max_characters
            ):
                batches.append(
                    "\n\n".join(current_batch)
                )

                current_batch = []
                current_length = 0

            current_batch.append(
                formatted_document
            )

            current_length += (
                document_length + 2
            )

            continue

        # ----------------------------------------------------
        # Large-document case:
        # Split a single document into smaller sections.
        # ----------------------------------------------------

        if current_batch:
            batches.append(
                "\n\n".join(current_batch)
            )

            current_batch = []
            current_length = 0

        content = document.page_content.strip()

        source = document.metadata.get(
            "source",
            "Unknown source",
        )

        page = document.metadata.get(
            "page",
            "Unknown page",
        )

        if isinstance(page, int):
            page = page + 1

        content_prefix = (
            f"Document: {source}\n"
            f"Page: {page}\n"
            f"Content:\n"
        )

        available_characters = (
            max_characters
            - len(content_prefix)
            - 100
        )

        if available_characters <= 0:
            raise ValueError(
                "SUMMARY_BATCH_CHAR_LIMIT is too small "
                "to construct a valid summary batch."
            )

        start = 0

        while start < len(content):
            end = (
                start
                + available_characters
            )

            section_content = content[
                start:end
            ]

            batches.append(
                f"[Section {index}]\n"
                f"{content_prefix}"
                f"{section_content}"
            )

            start = end

    if current_batch:
        batches.append(
            "\n\n".join(current_batch)
        )

    if not batches:
        logger.warning(
            "No usable summary batches were created."
        )

        return []

    logger.info(
        "Created %d summarization batches.",
        len(batches),
    )

    for index, batch in enumerate(
        batches,
        start=1,
    ):
        logger.debug(
            "Summary batch %d length: %d characters.",
            index,
            len(batch),
        )

    return batches


# ============================================================
# Intermediate Summary Prompt
# ============================================================

def build_intermediate_summary_prompt(
    context: str,
    prompt_config: Dict[str, Any],
) -> str:
    """
    Build the prompt used to summarize one section of the
    research paper.
    """

    if not context or not context.strip():
        raise ValueError(
            "Context cannot be empty."
        )

    intermediate_config = prompt_config.get(
        "intermediate_summary",
        {},
    )

    instruction = intermediate_config.get(
        "instruction",
        "",
    )

    instructions = intermediate_config.get(
        "instructions",
        [],
    )

    response_requirements = (
        intermediate_config.get(
            "response_requirements",
            {},
        )
    )

    instruction_text = ""

    if instruction:
        instruction_text = (
            f"TASK:\n"
            f"{instruction}"
        )

    instructions_text = ""

    if instructions:
        instructions_text = (
            "\n\nINSTRUCTIONS:\n"
            + "\n".join(
                f"- {item}"
                for item in instructions
            )
        )

    requirements_text = ""

    if response_requirements:
        requirements_text = (
            "\n\nRESPONSE REQUIREMENTS:\n"
            + "\n".join(
                f"- {key}: {value}"
                for key, value
                in response_requirements.items()
            )
        )

    final_prompt = (
        f"{instruction_text}"
        f"{instructions_text}"
        f"{requirements_text}"
        f"\n\n"
        f"RESEARCH PAPER SECTION:\n"
        f"{context}"
    )

    logger.debug(
        "Intermediate summary prompt length: %d characters.",
        len(final_prompt),
    )

    return final_prompt


# ============================================================
# Final Summary Prompt
# ============================================================

def build_final_summary_prompt(
    summaries: List[str],
    prompt_config: Dict[str, Any],
) -> str:
    """
    Build the prompt used to synthesize intermediate
    summaries into one final research-paper summary.
    """

    if not summaries:
        raise ValueError(
            "Intermediate summaries cannot be empty."
        )

    final_config = prompt_config.get(
        "final_summary",
        {},
    )

    instruction = final_config.get(
        "instruction",
        "",
    )

    instructions = final_config.get(
        "instructions",
        [],
    )

    response_requirements = (
        final_config.get(
            "response_requirements",
            {},
        )
    )

    summary_parts = []

    for index, summary in enumerate(
        summaries,
        start=1,
    ):
        if not summary or not summary.strip():
            continue

        summary_parts.append(
            f"[Intermediate Summary {index}]\n"
            f"{summary.strip()}"
        )

    if not summary_parts:
        raise ValueError(
            "No usable intermediate summaries were provided."
        )

    instruction_text = ""

    if instruction:
        instruction_text = (
            f"TASK:\n"
            f"{instruction}"
        )

    instructions_text = ""

    if instructions:
        instructions_text = (
            "\n\nINSTRUCTIONS:\n"
            + "\n".join(
                f"- {item}"
                for item in instructions
            )
        )

    requirements_text = ""

    if response_requirements:
        requirements_text = (
            "\n\nRESPONSE REQUIREMENTS:\n"
            + "\n".join(
                f"- {key}: {value}"
                for key, value
                in response_requirements.items()
            )
        )

    summaries_context = "\n\n".join(
        summary_parts
    )

    final_prompt = (
        f"{instruction_text}"
        f"{instructions_text}"
        f"{requirements_text}"
        f"\n\n"
        f"INTERMEDIATE SUMMARIES FROM ONE "
        f"RESEARCH PAPER:\n"
        f"{summaries_context}"
    )

    logger.debug(
        "Final synthesis prompt length: %d characters.",
        len(final_prompt),
    )

    return final_prompt


# ============================================================
# LLM Summarization
# ============================================================

def summarize_batch(
    batch: str,
    prompt_config: Dict[str, Any],
    client: Any,
) -> str:
    """
    Generate an intermediate summary for one batch.
    """

    if not batch or not batch.strip():
        raise ValueError(
            "Summary batch cannot be empty."
        )

    logger.info(
        "Generating intermediate summary for batch "
        "(%d characters).",
        len(batch),
    )

    prompt = build_intermediate_summary_prompt(
        context=batch,
        prompt_config=prompt_config,
    )

    summary = generate_response(
        client=client,
        user_prompt=prompt,
    )

    if not summary:
        raise ValueError(
            "Groq returned an empty intermediate summary."
        )

    logger.info(
        "Intermediate summary generated successfully."
    )

    logger.debug(
        "Intermediate summary length: %d characters.",
        len(summary),
    )

    return summary


def synthesize_summaries(
    summaries: List[str],
    prompt_config: Dict[str, Any],
    client: Groq,
    max_characters: int = SUMMARY_SYNTHESIS_CHAR_LIMIT,
) -> str:
    """
    Hierarchically synthesize intermediate summaries into one final summary.

    Summaries are grouped by character size rather than by a fixed number
    of summaries. This prevents excessively large requests to the LLM.

    Args:
        summaries: List of intermediate summaries.
        prompt_config: Loaded summary prompt configuration.
        client: Initialized Groq client.
        max_characters: Maximum combined summary size allowed in one
            synthesis group.

    Returns:
        A single synthesized summary.

    Raises:
        ValueError: If summaries are empty or max_characters is invalid.
    """

    if not summaries:
        raise ValueError("No summaries provided for synthesis.")

    if max_characters <= 0:
        raise ValueError("max_characters must be greater than 0.")

    logger.info(
        "Starting hierarchical synthesis of %d summaries.",
        len(summaries),
    )

    current_summaries = summaries

    synthesis_round = 1

    while len(current_summaries) > 1:
        logger.info(
            "Starting synthesis round %d with %d summaries.",
            synthesis_round,
            len(current_summaries),
        )

        groups = []
        current_group = []
        current_size = 0

        for summary in current_summaries:
            summary_size = len(summary)

            # If this summary alone is larger than the limit,
            # process it separately rather than combining it with
            # another summary.
            if summary_size > max_characters:
                if current_group:
                    groups.append(current_group)
                    current_group = []
                    current_size = 0

                groups.append([summary])

                logger.warning(
                    "Summary exceeds synthesis character limit: "
                    "%d > %d characters. Processing it separately.",
                    summary_size,
                    max_characters,
                )

                continue

            # If adding this summary would exceed the limit,
            # close the current group first.
            if (
                current_group
                and current_size + summary_size > max_characters
            ):
                groups.append(current_group)

                logger.debug(
                    "Closed synthesis group at %d characters.",
                    current_size,
                )

                current_group = []
                current_size = 0

            current_group.append(summary)
            current_size += summary_size

        # Add the final group.
        if current_group:
            groups.append(current_group)

        logger.info(
            "Created %d size-aware synthesis groups.",
            len(groups),
        )

        next_summaries = []

        for group_index, group in enumerate(groups, start=1):
            group_size = sum(len(summary) for summary in group)

            logger.info(
                "Processing synthesis group %d/%d "
                "(%d summaries, %d characters).",
                group_index,
                len(groups),
                len(group),
                group_size,
            )

            prompt = build_final_summary_prompt(
                summaries=group,
                prompt_config=prompt_config,
            )

            logger.debug(
                "Final synthesis prompt length: %d characters.",
                len(prompt),
            )

            summary = generate_response(
                user_prompt=prompt,
                client=client,
            )

            next_summaries.append(summary)

            logger.info(
                "Synthesis group %d completed. "
                "Generated summary length: %d characters.",
                group_index,
                len(summary),
            )

        current_summaries = next_summaries

        logger.info(
            "Synthesis round %d completed. "
            "Generated %d summaries.",
            synthesis_round,
            len(current_summaries),
        )

        synthesis_round += 1

    logger.info(
        "Hierarchical synthesis completed. Final summary length: %d characters.",
        len(current_summaries[0]),
    )

    return current_summaries[0]


# ============================================================
# Source Extraction
# ============================================================

def extract_sources(
    documents: List[Document],
) -> List[Dict[str, Any]]:
    """
    Extract unique document/page references from the paper.
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

        seen_sources.add(
            source_key
        )

        sources.append(
            {
                "source": source,
                "page": page,
            }
        )

    logger.info(
        "Extracted %d unique summary sources.",
        len(sources),
    )

    return sources


# ============================================================
# Main Summarization Function
# ============================================================

def summarize_documents(
    documents: List[Document],
) -> Dict[str, Any]:
    """
    Summarize one research paper.

    The input should contain Documents belonging to a single
    research paper.

    The paper is processed in multiple manageable batches
    rather than sending the entire paper to Groq in one request.
    """

    if not documents:
        raise ValueError(
            "Documents cannot be empty."
        )

    logger.info(
        "Starting research-paper summarization "
        "for %d documents.",
        len(documents),
    )

    try:
        # ----------------------------------------------------
        # Load prompt configuration once.
        # ----------------------------------------------------

        prompt_config = load_summary_prompt()

        # ----------------------------------------------------
        # Create manageable batches.
        # ----------------------------------------------------

        batches = create_summary_batches(
            documents=documents,
            max_characters=SUMMARY_BATCH_CHAR_LIMIT,
        )

        if not batches:
            logger.warning(
                "No usable content found in documents."
            )

            return {
                "summary": (
                    "No usable research-paper content "
                    "was found for summarization."
                ),
                "sources": [],
            }

        logger.info(
            "Paper will be summarized using %d batches.",
            len(batches),
        )

        # ----------------------------------------------------
        # Initialize Groq client once.
        # ----------------------------------------------------

        client = get_groq_client()

        # ----------------------------------------------------
        # Generate intermediate summaries.
        #
        # They are processed sequentially instead of sending
        # many requests simultaneously.
        # ----------------------------------------------------

        intermediate_summaries = []

        for index, batch in enumerate(
            batches,
            start=1,
        ):
            logger.info(
                "Processing summary batch %d/%d.",
                index,
                len(batches),
            )

            summary = summarize_batch(
                batch=batch,
                prompt_config=prompt_config,
                client=client,
            )

            intermediate_summaries.append(
                summary
            )

        logger.info(
            "Generated %d intermediate summaries.",
            len(intermediate_summaries),
        )

        # ----------------------------------------------------
        # Final synthesis.
        # ----------------------------------------------------

        logger.info(
            "Starting final summary synthesis."
        )

        final_summary = synthesize_summaries(
            summaries=intermediate_summaries,
            prompt_config=prompt_config,
            client=client,
        )

        # ----------------------------------------------------
        # Extract paper sources.
        # ----------------------------------------------------

        sources = extract_sources(
            documents
        )

        logger.info(
            "Research-paper summary generated successfully."
        )

        return {
            "summary": final_summary,
            "sources": sources,
        }

    except Exception:
        logger.exception(
            "Failed to summarize research paper."
        )
        raise