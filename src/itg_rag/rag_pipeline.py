import json
import logging
import re
from typing import Protocol

from pydantic import ValidationError

from itg_rag.errors import RAGError
from itg_rag.prompts import (
    SELECTION_SYSTEM,
    SUFFICIENCY_SYSTEM,
    SYSTEM,
    context_prompt,
)
from itg_rag.retriever import Retriever, validate_question
from itg_rag.schemas import (
    Answer,
    Answerability,
    AnswerDraft,
    Evidence,
    SearchHit,
    Source,
)

logger = logging.getLogger(__name__)
NO_CONTEXT = "В базе знаний недостаточно информации для ответа на этот вопрос."


class Chat(Protocol):
    def complete(
        self, system: str, user: str, *, json_mode: bool = True
    ) -> str: ...


def validate_draft(raw: str, hits: list[SearchHit]) -> AnswerDraft:
    if len(raw) > 24000:
        raise ValueError("Response is too long")
    draft = AnswerDraft.model_validate_json(raw, strict=True)
    if draft.sufficient_context != bool(draft.evidence):
        raise ValueError("Evidence must match sufficient_context")
    chunks = {hit.chunk.chunk_id: hit.chunk.text for hit in hits}
    seen = set()
    resolved = []
    for evidence in draft.evidence:
        text = chunks.get(evidence.chunk_id, "")
        pattern = r"\s+".join(
            re.escape(token) for token in evidence.quote.split()
        )
        match = re.search(pattern, text)
        if not text or match is None:
            raise ValueError("Evidence must be an exact quote from its chunk")
        exact = Evidence(chunk_id=evidence.chunk_id, quote=match.group())
        key = (exact.chunk_id, exact.quote)
        if key in seen:
            raise ValueError("Duplicate evidence")
        seen.add(key)
        resolved.append(exact)
    draft.evidence = resolved
    return draft


class RAGPipeline:
    def __init__(
        self,
        retriever: Retriever,
        llm: Chat | None,
        max_distance: float = 0.65,
    ) -> None:
        if not 0 <= max_distance <= 2:
            raise ValueError("max_distance must be between 0 and 2")
        self.retriever = retriever
        self.llm = llm
        self.max_distance = max_distance

    def ask(self, question: str) -> Answer:
        question = validate_question(question)
        hits = self.retriever.retrieve(question)
        good = [hit for hit in hits if hit.distance <= self.max_distance]
        logger.info("Accepted %d/%d retrieved chunks", len(good), len(hits))
        if not good:
            return self._abstain(question, hits, "weak_retrieval")
        if self.llm is None:
            raise RAGError("Generation client is required for answering")
        if not self._answerable(question, good):
            return self._abstain(question, hits, "context_cannot_answer")
        user = context_prompt(question, good)
        draft = None
        for attempt in range(2):
            raw = self.llm.complete(SYSTEM, user)
            try:
                draft = validate_draft(raw, good)
                if draft.sufficient_context and not self._answerable(
                    question, good, draft.evidence
                ):
                    draft = None
                    raise ValueError("Selected evidence is incomplete")
                break
            except (ValueError, ValidationError) as error:
                logger.warning(
                    "Invalid grounded response (%s), attempt %d",
                    "schema"
                    if isinstance(error, ValidationError)
                    else str(error),
                    attempt + 1,
                )
                user = (
                    context_prompt(question, good)
                    + (
                        "\nОтвет не прошёл проверку схемы, цитат или полноты. "
                        "Проверь каждую часть вопроса. Добавь все нужные "
                        "цитаты, включая условия и суммы. Верни JSON по схеме."
                    )
                    + json.dumps(
                        {
                            "validation_error": (
                                "schema"
                                if isinstance(error, ValidationError)
                                else str(error)
                            ),
                            "previous_response": raw[:24000],
                        },
                        ensure_ascii=False,
                    )
                )
        if draft is None:
            raise RAGError(
                "LLM returned invalid evidence twice; no answer published"
            )
        if not draft.sufficient_context:
            return self._abstain(question, hits, "context_cannot_answer")
        by_id = {hit.chunk.chunk_id: hit for hit in good}
        sources = []
        for index, evidence in enumerate(draft.evidence, 1):
            hit = by_id[evidence.chunk_id]
            sources.append(
                Source(
                    label=f"S{index}",
                    document_id=hit.chunk.document_id,
                    source_name=hit.chunk.source_name,
                    chunk_id=hit.chunk.chunk_id,
                    distance=hit.distance,
                    quote=evidence.quote,
                )
            )
        return Answer(
            question=question,
            status="answered",
            answer="\n\n".join(f"{s.quote} [{s.label}]" for s in sources),
            sources=sources,
            retrieved=hits,
        )

    def _answerable(
        self,
        question: str,
        hits: list[SearchHit],
        evidence: list[Evidence] | None = None,
    ) -> bool:
        if self.llm is None:
            raise RAGError("Generation client is required for answering")
        user = context_prompt(question, hits, Answerability)
        system = SUFFICIENCY_SYSTEM
        if evidence is not None:
            payload = json.loads(user)
            payload["selected_evidence"] = [
                item.model_dump() for item in evidence
            ]
            user = json.dumps(payload, ensure_ascii=False)
            system = SELECTION_SYSTEM
        for attempt in range(2):
            raw = self.llm.complete(system, user)
            try:
                if len(raw) > 4000:
                    raise ValueError("Answerability response is too long")
                decision = Answerability.model_validate_json(raw, strict=True)
                if decision.answerable != (decision.reason == "supported"):
                    raise ValueError("Inconsistent answerability decision")
                logger.info("Context answerability: %s", decision.reason)
                return decision.answerable
            except (ValueError, ValidationError):
                logger.warning(
                    "Invalid answerability, attempt %d", attempt + 1
                )
                user += "\nВерни только JSON по схеме с согласованными полями."
        raise RAGError("LLM returned invalid answerability twice")

    @staticmethod
    def _abstain(question: str, hits: list[SearchHit], reason: str) -> Answer:
        logger.info("Answer declined: %s", reason)
        return Answer(
            question=question,
            status="insufficient_context",
            answer=NO_CONTEXT,
            retrieved=hits,
            reason=reason,
        )
