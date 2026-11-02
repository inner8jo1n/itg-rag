import json
from pathlib import Path

from pydantic import Field, TypeAdapter

from itg_rag.errors import RAGError
from itg_rag.llm_client import LLMClient
from itg_rag.rag_pipeline import RAGPipeline
from itg_rag.schemas import Schema


class DemoCase(Schema):
    question: str = Field(min_length=1, max_length=2000)
    should_answer: bool
    expected_source: str | None = None
    expected_terms: list[str] = Field(default_factory=list)


def load_cases(path: Path) -> list[DemoCase]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    cases = TypeAdapter(list[DemoCase]).validate_python(raw)
    if not cases:
        raise RAGError("Demo dataset is empty")
    return cases


def evaluate(
    pipeline: RAGPipeline,
    cases: list[DemoCase],
    *,
    retrieval_only: bool = False,
    baseline: LLMClient | None = None,
) -> dict:
    rows = []
    for case in cases:
        try:
            result = None if retrieval_only else pipeline.ask(case.question)
            hits = (
                pipeline.retriever.retrieve(case.question)
                if result is None
                else result.retrieved
            )
            relevant = [
                hit for hit in hits if hit.distance <= pipeline.max_distance
            ]
            expected_found = (
                any(
                    h.chunk.source_name == case.expected_source
                    for h in relevant
                )
                if case.should_answer
                else None
            )
            row = {
                "question": case.question,
                "should_answer": case.should_answer,
                "expected_source": case.expected_source,
                "expected_source_found": expected_found,
                "expected_source_precision": (
                    sum(
                        h.chunk.source_name == case.expected_source
                        for h in relevant
                    )
                    / len(relevant)
                    if case.should_answer and relevant
                    else 0.0
                ),
                "accepted_context_count": len(relevant),
                "retrieved": [hit.model_dump() for hit in hits],
            }
            if result is not None:
                answered = result.status == "answered"
                row.update(
                    result=result.model_dump(),
                    correct_behavior=answered == case.should_answer,
                    expected_terms_found=all(
                        term.casefold() in result.answer.casefold()
                        for term in case.expected_terms
                    )
                    if answered
                    else not case.should_answer,
                )
            if baseline is not None:
                row["without_retrieval"] = baseline.complete(
                    "Отвечай на вопрос пользователя. "
                    "Если не знаешь, скажи об этом.",
                    case.question,
                    json_mode=False,
                )
            rows.append(row)
        except RAGError as error:
            rows.append({"question": case.question, "error": str(error)})
    positives = [row for row in rows if row.get("should_answer")]
    return {
        "mode": "live_retrieval" if retrieval_only else "live_rag",
        "max_distance": pipeline.max_distance,
        "top_k": pipeline.retriever.top_k,
        "total": len(rows),
        "errors": sum("error" in row for row in rows),
        "expected_terms_passed_count": sum(
            bool(row.get("expected_terms_found")) for row in rows
        )
        if not retrieval_only
        else None,
        "expected_source_recall": (
            sum(bool(row.get("expected_source_found")) for row in positives)
            / sum(case.should_answer for case in cases)
            if any(case.should_answer for case in cases)
            else None
        ),
        "correct_behavior_count": sum(
            bool(row.get("correct_behavior")) for row in rows
        )
        if not retrieval_only
        else None,
        "rows": rows,
    }
