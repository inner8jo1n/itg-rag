import logging
from contextlib import contextmanager

import weaviate
from weaviate.classes.config import (
    Configure,
    DataType,
    Property,
    VectorDistances,
)
from weaviate.classes.query import MetadataQuery
from weaviate.exceptions import WeaviateBaseError

from itg_rag.config import Settings
from itg_rag.embedder import validate_vectors
from itg_rag.errors import RAGError
from itg_rag.schemas import Chunk, SearchHit

logger = logging.getLogger(__name__)


class WeaviateStore:
    def __init__(self, client: weaviate.WeaviateClient, settings: Settings):
        self.client = client
        self.settings = settings
        self.collection = client.collections.use(settings.collection)

    def ensure(self, *, create: bool = False) -> None:
        name = self.settings.collection
        signature = self.settings.index_signature()
        if not self.client.collections.exists(name):
            if not create:
                raise RAGError("Index does not exist. Run: itg-rag ingest")
            self.client.collections.create(
                name=name,
                description=signature,
                vector_config=Configure.Vectors.self_provided(
                    vector_index_config=Configure.VectorIndex.hnsw(
                        distance_metric=VectorDistances.COSINE
                    )
                ),
                properties=[
                    Property(name=key, data_type=DataType.TEXT)
                    for key in (
                        "document_id",
                        "source_name",
                        "chunk_id",
                        "text",
                    )
                ]
                + [
                    Property(name="start", data_type=DataType.INT),
                    Property(name="end", data_type=DataType.INT),
                ],
            )
        if self.collection.config.get().description != signature:
            raise RAGError(
                "Index configuration differs: use a new "
                "WEAVIATE_COLLECTION and ingest again"
            )

    def count(self) -> int:
        return (
            self.collection.aggregate.over_all(total_count=True).total_count
            or 0
        )

    def sync(self, chunks: list[Chunk], vectors: list[list[float]]) -> dict:
        if not chunks:
            raise RAGError("Refusing to synchronize an empty corpus")
        validate_vectors(vectors, len(chunks))
        if len({chunk.chunk_id for chunk in chunks}) != len(chunks):
            raise RAGError("Duplicate chunk IDs")
        self.ensure(create=True)
        existing = {str(item.uuid) for item in self.collection.iterator()}
        current = {chunk.chunk_id for chunk in chunks}
        for chunk, vector in zip(chunks, vectors, strict=True):
            properties = chunk.model_dump()
            if chunk.chunk_id in existing:
                self.collection.data.replace(
                    uuid=chunk.chunk_id, properties=properties, vector=vector
                )
            else:
                self.collection.data.insert(
                    uuid=chunk.chunk_id, properties=properties, vector=vector
                )
        stale = existing - current
        for identifier in stale:
            self.collection.data.delete_by_id(identifier)
        count = self.count()
        if count != len(chunks):
            raise RAGError("Index object count differs from corpus size")
        logger.info("Indexed %d chunks; removed %d stale", count, len(stale))
        return {
            "indexed": count,
            "inserted": len(current - existing),
            "updated": len(current & existing),
            "removed": len(stale),
        }

    def search(self, vector: list[float], top_k: int) -> list[SearchHit]:
        self.ensure()
        validate_vectors([vector], 1)
        result = self.collection.query.near_vector(
            near_vector=vector,
            limit=top_k,
            return_metadata=MetadataQuery(distance=True),
        )
        hits = []
        for item in result.objects:
            if item.metadata.distance is None:
                raise RAGError("Weaviate did not return distance")
            hits.append(
                SearchHit(
                    chunk=Chunk.model_validate(item.properties),
                    distance=max(0.0, min(2.0, item.metadata.distance)),
                )
            )
        return hits


@contextmanager
def connect_store(settings: Settings):
    client = None
    try:
        client = weaviate.connect_to_local(
            host=settings.weaviate_host,
            port=settings.weaviate_http_port,
            grpc_port=settings.weaviate_grpc_port,
        )
        yield WeaviateStore(client, settings)
    except WeaviateBaseError:
        raise RAGError(
            "Weaviate operation failed. Check: docker compose ps; "
            "HTTP/gRPC ports; collection and embedding configuration"
        ) from None
    finally:
        if client is not None:
            client.close()
