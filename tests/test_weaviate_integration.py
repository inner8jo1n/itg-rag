import os
import uuid

import pytest

from itg_rag.config import Settings
from itg_rag.errors import RAGError
from itg_rag.schemas import Chunk
from itg_rag.weaviate_store import WeaviateStore, connect_store

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("RUN_WEAVIATE_TESTS") != "1",
        reason="Set RUN_WEAVIATE_TESTS=1 with local Weaviate running",
    ),
]


def test_real_storage_search_upsert_and_stale_cleanup() -> None:
    settings = Settings.from_env().model_copy(
        update={"collection": "TestRag" + uuid.uuid4().hex}
    )
    first = Chunk(
        document_id="a",
        source_name="a.txt",
        chunk_id=str(uuid.uuid4()),
        text="Delivery takes two days",
        start=0,
        end=23,
    )
    second = Chunk(
        document_id="b",
        source_name="b.txt",
        chunk_id=str(uuid.uuid4()),
        text="Returns within 21 days",
        start=0,
        end=22,
    )
    with connect_store(settings) as store:
        try:
            result = store.sync([first, second], [[1.0, 0.0], [0.0, 1.0]])
            assert result["inserted"] == 2
            assert store.count() == 2
            hits = store.search([1.0, 0.0], 2)
            assert hits[0].chunk == first
            assert hits[0].distance < 0.001
            result = store.sync([first, second], [[1.0, 0.0], [0.0, 1.0]])
            assert result["inserted"] == 0
            assert result["updated"] == 2
            edited = first.model_copy(update={"text": "Updated delivery text"})
            result = store.sync([edited], [[1.0, 0.0]])
            assert result["removed"] == 1
            assert store.count() == 1
            assert store.search([1.0, 0.0], 1)[0].chunk.text == edited.text
            incompatible = WeaviateStore(
                store.client, settings.model_copy(update={"chunk_size": 1000})
            )
            with pytest.raises(RAGError, match="configuration differs"):
                incompatible.ensure()
        finally:
            store.client.collections.delete(settings.collection)
