import json
import time

from typing import Any

from groq import Groq
from groq import RateLimitError

from app.config.config import (
    GROQ_API_KEY,
    GROQ_DEFAULT_RETRY_DELAY,
    GROQ_MAX_RETRIES,
    GROQ_MAX_RETRY_DELAY,
    GROQ_MODEL,
    SYSTEM_PROMPT_FILE,
)
from app.utils.logger import get_logger


logger = get_logger(__name__)


# ============================================================
# System Prompt
# ============================================================

def load_system_prompt() -> str:
    """
    Load the system prompt configuration from
    system_prompt.json and convert it into a prompt string.
    """

    logger.info(
        "Loading system prompt from: %s",
        SYSTEM_PROMPT_FILE,
    )

    if not SYSTEM_PROMPT_FILE.exists():
        logger.error(
            "System prompt file not found: %s",
            SYSTEM_PROMPT_FILE,
        )

        raise FileNotFoundError(
            f"System prompt file not found: "
            f"{SYSTEM_PROMPT_FILE}"
        )

    try:
        with open(
            SYSTEM_PROMPT_FILE,
            "r",
            encoding="utf-8",
        ) as file:
            prompt_config = json.load(file)

    except json.JSONDecodeError as exc:
        logger.exception(
            "Invalid JSON in system prompt file."
        )

        raise ValueError(
            "system_prompt.json contains invalid JSON."
        ) from exc

    except Exception:
        logger.exception(
            "Failed to read system prompt configuration."
        )

        raise

    if not prompt_config:
        raise ValueError(
            "system_prompt.json is empty."
        )

    prompt_parts = []

    role = prompt_config.get(
        "role"
    )

    persona = prompt_config.get(
        "persona"
    )

    tone = prompt_config.get(
        "tone",
        [],
    )

    objectives = prompt_config.get(
        "objectives",
        [],
    )

    behavior = prompt_config.get(
        "behavior",
        [],
    )

    rules = prompt_config.get(
        "rules",
        [],
    )

    source_policy = prompt_config.get(
        "source_policy",
        {},
    )

    response_style = prompt_config.get(
        "response_style",
        {},
    )

    if role:
        prompt_parts.append(
            f"ROLE:\n{role}"
        )

    if persona:
        prompt_parts.append(
            f"PERSONA:\n{persona}"
        )

    if tone:
        prompt_parts.append(
            "TONE:\n"
            + "\n".join(
                f"- {item}"
                for item in tone
            )
        )

    if objectives:
        prompt_parts.append(
            "OBJECTIVES:\n"
            + "\n".join(
                f"- {item}"
                for item in objectives
            )
        )

    if behavior:
        prompt_parts.append(
            "BEHAVIOR:\n"
            + "\n".join(
                f"- {item}"
                for item in behavior
            )
        )

    if rules:
        prompt_parts.append(
            "RULES:\n"
            + "\n".join(
                f"- {item}"
                for item in rules
            )
        )

    if source_policy:
        prompt_parts.append(
            "SOURCE POLICY:\n"
            + "\n".join(
                f"- {key}: {value}"
                for key, value
                in source_policy.items()
            )
        )

    if response_style:
        prompt_parts.append(
            "RESPONSE STYLE:\n"
            + "\n".join(
                f"- {key}: {value}"
                for key, value
                in response_style.items()
            )
        )

    system_prompt = "\n\n".join(
        prompt_parts
    )

    if not system_prompt.strip():
        raise ValueError(
            "No valid system prompt content found "
            "in system_prompt.json."
        )

    logger.info(
        "System prompt loaded successfully."
    )

    logger.debug(
        "System prompt length: %d characters.",
        len(system_prompt),
    )

    return system_prompt


# ============================================================
# Groq Client
# ============================================================

def get_groq_client() -> Groq:
    """
    Create and return a configured Groq API client.
    """

    if not GROQ_API_KEY:
        logger.error(
            "GROQ_API_KEY environment variable "
            "is not configured."
        )

        raise ValueError(
            "GROQ_API_KEY is not configured. "
            "Add your Groq API key to the .env file."
        )

    logger.info(
        "Initializing Groq client with model: %s",
        GROQ_MODEL,
    )

    try:
        client = Groq(
            api_key=GROQ_API_KEY
        )

        logger.info(
            "Groq client initialized successfully."
        )

        return client

    except Exception:
        logger.exception(
            "Failed to initialize Groq client."
        )

        raise


# ============================================================
# Retry Handling
# ============================================================

def get_retry_delay(
    error: RateLimitError,
    retry_attempt: int,
) -> float:
    """
    Determine how long to wait before retrying a
    rate-limited Groq request.

    The function first attempts to use retry information
    supplied by Groq. If that information is unavailable,
    exponential backoff is used.
    """

    # --------------------------------------------------------
    # Try to read retry-after information from the response.
    # --------------------------------------------------------

    response = getattr(
        error,
        "response",
        None,
    )

    if response is not None:
        headers = getattr(
            response,
            "headers",
            None,
        )

        if headers:
            retry_after = headers.get(
                "retry-after"
            )

            if retry_after is not None:
                try:
                    delay = float(
                        retry_after
                    )

                    return min(
                        delay,
                        GROQ_MAX_RETRY_DELAY,
                    )

                except (
                    TypeError,
                    ValueError,
                ):
                    logger.debug(
                        "Could not parse retry-after "
                        "header: %s",
                        retry_after,
                    )

    # --------------------------------------------------------
    # Fall back to exponential backoff.
    # --------------------------------------------------------

    delay = (
        GROQ_DEFAULT_RETRY_DELAY
        * (2 ** (retry_attempt - 1))
    )

    return min(
        delay,
        GROQ_MAX_RETRY_DELAY,
    )


# ============================================================
# Response Generation
# ============================================================

def generate_response(
    client: Groq,
    user_prompt: str,
    system_prompt: str | None = None,
) -> str:
    """
    Generate a response using the Groq Compound model.

    Rate-limit errors are automatically retried using the
    retry configuration defined in config.py.

    If no system prompt is explicitly provided, the prompt
    is loaded from system_prompt.json.
    """

    if client is None:
        raise ValueError(
            "Groq client cannot be None."
        )

    if not user_prompt or not user_prompt.strip():
        raise ValueError(
            "User prompt cannot be empty."
        )

    user_prompt = user_prompt.strip()

    if system_prompt is None:
        system_prompt = load_system_prompt()

    logger.info(
        "Sending request to Groq model: %s",
        GROQ_MODEL,
    )

    logger.debug(
        "User prompt length: %d characters.",
        len(user_prompt),
    )

    # --------------------------------------------------------
    # Initial request + configured retry attempts.
    # --------------------------------------------------------

    total_attempts = (
        GROQ_MAX_RETRIES + 1
    )

    for attempt in range(
        1,
        total_attempts + 1,
    ):
        try:
            logger.debug(
                "Groq request attempt %d/%d.",
                attempt,
                total_attempts,
            )

            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                compound_custom={
                    "tools": {
                        "enabled_tools": []
                    }
                },
            )

            content = (
                response
                .choices[0]
                .message
                .content
            )

            if not content:
                logger.warning(
                    "Groq returned an empty response."
                )

                return ""

            logger.info(
                "Groq response generated successfully."
            )

            logger.debug(
                "Response length: %d characters.",
                len(content),
            )

            return content.strip()

        except RateLimitError as exc:

            # ------------------------------------------------
            # If this was the final allowed attempt, propagate
            # the original Groq error.
            # ------------------------------------------------

            if attempt >= total_attempts:
                logger.error(
                    "Groq rate limit persisted after "
                    "%d attempts.",
                    total_attempts,
                )

                raise

            retry_delay = get_retry_delay(
                error=exc,
                retry_attempt=attempt,
            )

            logger.warning(
                "Groq rate limit reached on attempt "
                "%d/%d. Retrying in %.2f seconds.",
                attempt,
                total_attempts,
                retry_delay,
            )

            time.sleep(
                retry_delay
            )

        except Exception:
            logger.exception(
                "Groq API request failed."
            )

            raise

    # --------------------------------------------------------
    # This point should never be reached.
    # --------------------------------------------------------

    raise RuntimeError(
        "Groq response generation ended unexpectedly."
    )