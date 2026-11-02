import json

import pytest

from itg_rag.errors import RAGError
from itg_rag.prompts import SELECTION_SYSTEM, SUFFICIENCY_SYSTEM
from itg_rag.rag_pipeline import RAGPipeline, validate_draft
from itg_rag.retriever import Retriever
from itg_rag.schemas import Chunk, SearchHit


class FakeEmbeddings:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


class FakeStore:
    def __init__(self, hits: list[SearchHit]):
        self.hits = hits

    def search(self, vector: list[float], top_k: int) -> list[SearchHit]:
        return self.hits[:top_k]


class FakeChat:
    def __init__(self, replies: list[str]):
        self.replies = iter(replies)
        self.calls = 0
        self.answerability_calls = 0
        self.answerability_overrides: list[str] = []
        self.answerability_reply = (
            '{"answerable": true, "reason": "supported"}'
        )

    def complete(
        self, system: str, user: str, *, json_mode: bool = True
    ) -> str:
        if system in {SUFFICIENCY_SYSTEM, SELECTION_SYSTEM}:
            self.answerability_calls += 1
            if self.answerability_overrides:
                return self.answerability_overrides.pop(0)
            return self.answerability_reply
        self.calls += 1
        return next(self.replies)


@pytest.fixture
def hit() -> SearchHit:
    text = "Вернуть товар можно в течение 21 календарного дня."
    return SearchHit(
        chunk=Chunk(
            document_id="doc",
            source_name="returns.txt",
            chunk_id="id",
            text=text,
            start=0,
            end=len(text),
        ),
        distance=0.2,
    )


def reply(hit: SearchHit) -> str:
    return json.dumps(
        {
            "sufficient_context": True,
            "evidence": [
                {"chunk_id": hit.chunk.chunk_id, "quote": hit.chunk.text}
            ],
        }
    )


def pipeline(hits: list[SearchHit], chat: FakeChat) -> RAGPipeline:
    return RAGPipeline(Retriever(FakeEmbeddings(), FakeStore(hits)), chat)


def test_grounded_answer_and_sources(hit: SearchHit) -> None:
    chat = FakeChat([reply(hit)])
    result = pipeline([hit], chat).ask("Срок возврата?")
    assert result.status == "answered"
    assert result.answer == hit.chunk.text + " [S1]"
    assert result.sources[0].chunk_id == hit.chunk.chunk_id
    assert result.sources[0].quote in result.retrieved[0].chunk.text
    assert chat.calls == 1


@pytest.mark.parametrize("distance", [0.651, 1.0])
def test_weak_context_never_calls_llm(hit: SearchHit, distance: float) -> None:
    chat = FakeChat([])
    hit.distance = distance
    result = pipeline([hit], chat).ask("Что-то неизвестное")
    assert result.status == "insufficient_context"
    assert result.reason == "weak_retrieval"
    assert not result.sources
    assert chat.calls == 0


def test_empty_context() -> None:
    chat = FakeChat([])
    assert pipeline([], chat).ask("Вопрос").status == "insufficient_context"
    assert chat.calls == 0
    assert chat.answerability_calls == 0


@pytest.mark.parametrize("reason", ["different_subject", "missing_facts"])
def test_answerability_gate_skips_generation(
    hit: SearchHit, reason: str
) -> None:
    chat = FakeChat([])
    chat.answerability_reply = json.dumps(
        {"answerable": False, "reason": reason}
    )
    result = pipeline([hit], chat).ask("Вопрос о другом предмете")
    assert result.status == "insufficient_context"
    assert result.reason == "context_cannot_answer"
    assert chat.calls == 0
    assert chat.answerability_calls == 1


def test_invalid_answerability_fails_closed(hit: SearchHit) -> None:
    chat = FakeChat([])
    chat.answerability_reply = (
        '{"answerable": true, "reason": "missing_facts"}'
    )
    with pytest.raises(RAGError, match="invalid answerability twice"):
        pipeline([hit], chat).ask("Вопрос")
    assert chat.calls == 0
    assert chat.answerability_calls == 2


def test_incomplete_selected_evidence_retried(hit: SearchHit) -> None:
    chat = FakeChat([reply(hit), reply(hit)])
    chat.answerability_overrides = [
        '{"answerable": true, "reason": "supported"}',
        '{"answerable": false, "reason": "missing_facts"}',
        '{"answerable": true, "reason": "supported"}',
    ]
    assert pipeline([hit], chat).ask("Составной вопрос").status == "answered"
    assert chat.calls == 2
    assert chat.answerability_calls == 3


def test_selection_check_keeps_source_context(hit: SearchHit) -> None:
    quote = hit.chunk.text
    hit.chunk.text = "Правила магазина Полка.\n" + quote

    class RecordingChat(FakeChat):
        def complete(
            self, system: str, user: str, *, json_mode: bool = True
        ) -> str:
            if system == SELECTION_SYSTEM:
                payload = json.loads(user)
                assert payload["context"][0]["text"] == hit.chunk.text
                assert payload["selected_evidence"][0]["quote"] == quote
            return super().complete(system, user, json_mode=json_mode)

    raw = json.dumps(
        {
            "sufficient_context": True,
            "evidence": [{"chunk_id": hit.chunk.chunk_id, "quote": quote}],
        }
    )
    result = pipeline([hit], RecordingChat([raw])).ask(
        "Срок возврата в Полку?"
    )
    assert result.status == "answered"


def test_llm_can_decline_nearby_context(hit: SearchHit) -> None:
    chat = FakeChat(['{"sufficient_context": false, "evidence": []}'])
    result = pipeline([hit], chat).ask("Возврат в другой стране?")
    assert result.status == "insufficient_context"
    assert result.reason == "context_cannot_answer"


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "not json",
        "{}",
        '{"sufficient_context": "yes", "evidence": []}',
        '{"sufficient_context": true, "evidence": []}',
        '{"sufficient_context": false, "evidence": [], "answer": "invented"}',
        "x" * 24001,
    ],
)
def test_invalid_draft_fallback(hit: SearchHit, raw: str) -> None:
    chat = FakeChat([raw, reply(hit)])
    assert pipeline([hit], chat).ask("Срок возврата?").status == "answered"
    assert chat.calls == 2


def test_quote_alignment_restores_original_whitespace(hit: SearchHit) -> None:
    hit.chunk.text = "Срок возврата — 21 день.\nСохраните упаковку товара."
    raw = json.dumps(
        {
            "sufficient_context": True,
            "evidence": [
                {
                    "chunk_id": hit.chunk.chunk_id,
                    "quote": (
                        "Срок возврата — 21 день. Сохраните упаковку товара."
                    ),
                }
            ],
        }
    )
    draft = validate_draft(raw, [hit])
    assert draft.evidence[0].quote == hit.chunk.text


def test_repeated_invalid_response_fails_closed(hit: SearchHit) -> None:
    chat = FakeChat(["{}", "{}"])
    with pytest.raises(RAGError, match="invalid evidence twice"):
        pipeline([hit], chat).ask("Вопрос")


@pytest.mark.parametrize(
    "chunk_id,quote",
    [
        ("unknown", "Вернуть товар можно в течение 21 календарного дня."),
        ("id", "Вернуть товар можно через сто лет по правилам инопланетян."),
    ],
)
def test_fabricated_citations_rejected(
    hit: SearchHit, chunk_id: str, quote: str
) -> None:
    raw = json.dumps(
        {
            "sufficient_context": True,
            "evidence": [{"chunk_id": chunk_id, "quote": quote}],
        }
    )
    with pytest.raises(ValueError, match="exact quote"):
        validate_draft(raw, [hit])


def test_weak_chunks_excluded_from_llm_evidence(hit: SearchHit) -> None:
    weak = hit.model_copy(deep=True)
    weak.chunk.chunk_id = "weak"
    weak.distance = 0.9
    chat = FakeChat([reply(weak), reply(hit)])
    result = pipeline([hit, weak], chat).ask("Вопрос")
    assert result.sources[0].chunk_id == "id"
    assert len(result.retrieved) == 2
    assert chat.calls == 2


@pytest.mark.parametrize("question", [" ", "x" * 2001])
def test_invalid_question(question: str) -> None:
    with pytest.raises(RAGError, match="Question"):
        pipeline([], FakeChat([])).ask(question)
