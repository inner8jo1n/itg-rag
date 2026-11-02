import logging
import math

from openai import OpenAI

from itg_rag.api import call_api
from itg_rag.errors import RAGError

logger = logging.getLogger(__name__)


def validate_vectors(vectors: list[list[float]], count: int) -> None:
    if len(vectors) != count:
        raise RAGError("Embedding API returned an unexpected vector count")
    dimensions = {len(vector) for vector in vectors}
    if len(dimensions) != 1 or 0 in dimensions:
        raise RAGError("Embedding vectors have inconsistent dimensions")
    if any(
        not all(math.isfinite(value) for value in vector)
        or not any(value != 0 for value in vector)
        for vector in vectors
    ):
        raise RAGError("Embedding vectors must be finite and non-zero")


class Embedder:
    def __init__(self, client: OpenAI, model: str) -> None:
        self.client = client
        self.model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if any(not text.strip() for text in texts):
            raise RAGError("Cannot embed empty text")
        logger.info("Embedding %d texts with %s", len(texts), self.model)
        vectors = []
        for start in range(0, len(texts), 32):
            batch = texts[start : start + 32]
            result = call_api(
                lambda batch=batch: self.client.embeddings.create(
                    model=self.model, input=batch, encoding_format="float"
                )
            )
            ordered = sorted(result.data, key=lambda item: item.index)
            if [item.index for item in ordered] != list(range(len(batch))):
                raise RAGError("Embedding API returned invalid indices")
            vectors.extend(item.embedding for item in ordered)
        validate_vectors(vectors, len(texts))
        return vectors
