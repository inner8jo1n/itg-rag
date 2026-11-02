from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Schema(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class Document(Schema):
    document_id: str
    source_name: str
    text: str = Field(min_length=1)


class Chunk(Schema):
    document_id: str
    source_name: str
    chunk_id: str
    text: str = Field(min_length=1)
    start: int = Field(ge=0)
    end: int = Field(gt=0)


class SearchHit(Schema):
    chunk: Chunk
    distance: float = Field(ge=0, le=2, allow_inf_nan=False)


class Evidence(Schema):
    chunk_id: str = Field(min_length=1)
    quote: str = Field(min_length=15, max_length=1800)


class AnswerDraft(Schema):
    sufficient_context: bool
    evidence: list[Evidence] = Field(max_length=6)


class Answerability(Schema):
    answerable: bool
    reason: Literal["supported", "missing_facts", "different_subject"]


class Source(Schema):
    label: str
    document_id: str
    source_name: str
    chunk_id: str
    distance: float
    quote: str


class Answer(Schema):
    question: str
    status: Literal["answered", "insufficient_context"]
    answer: str
    sources: list[Source] = Field(default_factory=list)
    retrieved: list[SearchHit] = Field(default_factory=list)
    reason: str | None = None
