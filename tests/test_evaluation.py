from pathlib import Path

import pytest
from pydantic import ValidationError

from itg_rag.errors import RAGError
from itg_rag.evaluation import DemoCase, evaluate, load_cases
from itg_rag.rag_pipeline import RAGPipeline
from itg_rag.retriever import Retriever
from itg_rag.schemas import Chunk, SearchHit


class BrokenOnceEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        if texts[0] == "broken":
            raise RAGError("API request failed: HTTP 503")
        return [[1.0, 0.0]]


class Store:
    def search(self, vector: list[float], top_k: int) -> list[SearchHit]:
        return [
            SearchHit(
                chunk=Chunk(
                    document_id="id",
                    source_name="test.txt",
                    chunk_id="chunk",
                    text="Source text for testing",
                    start=0,
                    end=23,
                ),
                distance=0.1,
            )
        ]


def test_evaluation_preserves_progress_after_failed_item() -> None:
    pipeline = RAGPipeline(Retriever(BrokenOnceEmbedder(), Store()), None)
    report = evaluate(
        pipeline,
        [
            DemoCase(
                question="broken",
                should_answer=True,
                expected_source="test.txt",
            ),
            DemoCase(
                question="working",
                should_answer=True,
                expected_source="test.txt",
            ),
        ],
        retrieval_only=True,
    )
    assert report["errors"] == 1
    assert report["total"] == 2
    assert report["expected_source_recall"] == 0.5
    assert report["rows"][1]["expected_source_found"]


@pytest.mark.parametrize("content", ["{}", '[{"question": "x"}]'])
def test_invalid_dataset_schema(tmp_path: Path, content: str) -> None:
    path = tmp_path / "questions.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValidationError):
        load_cases(path)


def test_empty_dataset(tmp_path: Path) -> None:
    path = tmp_path / "questions.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(RAGError, match="empty"):
        load_cases(path)
