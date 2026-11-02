import logging
from collections.abc import Callable

from openai import APIError, OpenAI

from itg_rag.config import Settings
from itg_rag.errors import RAGError

logger = logging.getLogger(__name__)


def client(settings: Settings, *, embedding: bool = False) -> OpenAI:
    return OpenAI(
        api_key=settings.require_key(embedding=embedding),
        base_url=(
            settings.embedding_base_url or settings.base_url
            if embedding
            else settings.base_url
        ),
        timeout=30.0,
        max_retries=2,
    )


def call_api[T](operation: Callable[[], T]) -> T:
    try:
        return operation()
    except APIError as error:
        status = getattr(error, "status_code", None)
        description = f"HTTP {status}" if status else type(error).__name__
        logger.error("API request failed: %s", description)
        raise RAGError(f"API request failed: {description}") from None
