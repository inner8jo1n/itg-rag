import json

import httpx2
import pytest
from openai import OpenAI

from itg_rag.embedder import Embedder, validate_vectors
from itg_rag.errors import RAGError
from itg_rag.llm_client import LLMClient


def api(handler, retries: int = 0) -> OpenAI:
    return OpenAI(
        api_key="test-secret",
        base_url="https://test.invalid/v1",
        max_retries=retries,
        http_client=httpx2.Client(transport=httpx2.MockTransport(handler)),
    )


def test_embeddings_preserve_input_order() -> None:
    def handler(request):
        body = json.loads(request.content)
        assert body["input"] == ["first", "second"]
        return httpx2.Response(
            200,
            json={
                "data": [
                    {
                        "index": 1,
                        "embedding": [0.0, 1.0],
                        "object": "embedding",
                    },
                    {
                        "index": 0,
                        "embedding": [1.0, 0.0],
                        "object": "embedding",
                    },
                ],
                "model": "embedding-model",
                "object": "list",
                "usage": {"prompt_tokens": 2, "total_tokens": 2},
            },
        )

    with api(handler) as client:
        assert Embedder(client, "embedding-model").embed(
            ["first", "second"]
        ) == [
            [1.0, 0.0],
            [0.0, 1.0],
        ]


@pytest.mark.parametrize(
    "vectors,count",
    [
        ([], 1),
        ([[0.0]], 1),
        ([[float("nan")]], 1),
        ([[1.0], [1.0, 2.0]], 2),
        ([[float("inf")]], 1),
    ],
)
def test_bad_vectors(vectors: list[list[float]], count: int) -> None:
    with pytest.raises(RAGError):
        validate_vectors(vectors, count)


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_permanent_api_error_is_sanitized(status: int) -> None:
    calls = []

    def handler(request):
        calls.append(request)
        return httpx2.Response(status, json={"error": "test-secret echoed"})

    with api(handler, retries=2) as client:
        with pytest.raises(RAGError) as captured:
            Embedder(client, "model").embed(["text"])
        assert str(captured.value) == f"API request failed: HTTP {status}"
    assert len(calls) == 1


def test_transient_api_error_retried() -> None:
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx2.Response(503, headers={"retry-after-ms": "1"})
        return httpx2.Response(
            200,
            json={
                "id": "id",
                "created": 1,
                "model": "test",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "{}"},
                    }
                ],
            },
        )

    with api(handler, retries=2) as client:
        assert LLMClient(client, "test").complete("system", "user") == "{}"
    assert len(calls) == 2


def test_truncated_llm_reply_rejected() -> None:
    def handler(request):
        return httpx2.Response(
            200,
            json={
                "id": "id",
                "created": 1,
                "model": "test",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "length",
                        "message": {"role": "assistant", "content": "{}"},
                    }
                ],
            },
        )

    with (
        api(handler) as client,
        pytest.raises(RAGError, match="incomplete"),
    ):
        LLMClient(client, "test").complete("system", "user")


@pytest.mark.parametrize("status", [408, 409, 429, 500])
def test_temporary_errors_have_bounded_retries(status: int) -> None:
    calls = []

    def handler(request):
        calls.append(request)
        return httpx2.Response(
            status,
            headers={"retry-after-ms": "1"},
            json={"error": "private-provider-message"},
        )

    with api(handler, retries=2) as client:
        with pytest.raises(RAGError) as captured:
            Embedder(client, "model").embed(["text"])
        assert str(captured.value) == f"API request failed: HTTP {status}"
    assert len(calls) == 3


def test_connection_error_is_sanitized() -> None:
    def handler(request):
        raise httpx2.ConnectError("private-provider-message", request=request)

    with api(handler) as client:
        with pytest.raises(RAGError) as captured:
            Embedder(client, "model").embed(["text"])
        assert str(captured.value) == "API request failed: APIConnectionError"
