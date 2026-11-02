import logging
from typing import Protocol

from itg_rag.errors import RAGError
from itg_rag.schemas import SearchHit

logger = logging.getLogger(__name__)


class Embeddings(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class SearchStore(Protocol):
    def search(self, vector: list[float], top_k: int) -> list[SearchHit]: ...


def validate_question(question: str) -> str:
    question = question.strip()
    if not question or len(question) > 2000:
        raise RAGError("Question must contain between 1 and 2000 characters")
    return question


class Retriever:
    def __init__(
        self, embedder: Embeddings, store: SearchStore, top_k: int = 4
    ) -> None:
        if not 1 <= top_k <= 20:
            raise ValueError("top_k must be between 1 and 20")
        self.embedder = embedder
        self.store = store
        self.top_k = top_k

    def retrieve(self, question: str) -> list[SearchHit]:
        question = validate_question(question)
        hits = self.store.search(
            self.embedder.embed([question])[0], self.top_k
        )
        logger.info("Retrieved %d chunks", len(hits))
        return hits
