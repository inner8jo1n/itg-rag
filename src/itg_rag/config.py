import hashlib
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field, SecretStr, model_validator

from itg_rag.errors import RAGError


class Settings(BaseModel):
    api_key: SecretStr = SecretStr("")
    base_url: str | None = None
    model: str = "gpt-4.1-mini"
    embedding_api_key: SecretStr = SecretStr("")
    embedding_base_url: str | None = None
    embedding_model: str = "text-embedding-3-small"
    knowledge_dir: Path = Path("knowledge_base")
    chunk_size: int = Field(default=900, ge=100, le=4000)
    chunk_overlap: int = Field(default=150, ge=0)
    top_k: int = Field(default=4, ge=1, le=20)
    max_distance: float = Field(default=0.65, ge=0, le=2)
    weaviate_host: str = "127.0.0.1"
    weaviate_http_port: int = Field(default=8080, ge=1, le=65535)
    weaviate_grpc_port: int = Field(default=50051, ge=1, le=65535)
    collection: str = Field(default="KnowledgeChunk", pattern=r"^[A-Z]\w*$")

    @model_validator(mode="after")
    def valid_overlap(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        return self

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv(Path.cwd() / ".env", override=False)
        names = {
            "api_key": "OPENAI_API_KEY",
            "base_url": "OPENAI_BASE_URL",
            "model": "OPENAI_MODEL",
            "embedding_api_key": "EMBEDDING_API_KEY",
            "embedding_base_url": "EMBEDDING_BASE_URL",
            "embedding_model": "EMBEDDING_MODEL",
            "knowledge_dir": "KNOWLEDGE_DIR",
            "chunk_size": "CHUNK_SIZE",
            "chunk_overlap": "CHUNK_OVERLAP",
            "top_k": "TOP_K",
            "max_distance": "MAX_DISTANCE",
            "weaviate_host": "WEAVIATE_HOST",
            "weaviate_http_port": "WEAVIATE_HTTP_PORT",
            "weaviate_grpc_port": "WEAVIATE_GRPC_PORT",
            "collection": "WEAVIATE_COLLECTION",
        }
        values = {
            field: os.environ[name]
            for field, name in names.items()
            if os.environ.get(name)
        }
        return cls.model_validate(values)

    def require_key(self, *, embedding: bool = False) -> str:
        key = self.embedding_api_key if embedding else self.api_key
        if embedding and not key.get_secret_value():
            key = self.api_key
        if not key.get_secret_value():
            raise RAGError("Set API keys in .env; see .env.example")
        return key.get_secret_value()

    def index_signature(self) -> str:
        config = {
            "format": 1,
            "embedding_model": self.embedding_model,
            "embedding_endpoint": (
                self.embedding_base_url or self.base_url or "openai"
            ),
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
        }
        return hashlib.sha256(
            json.dumps(config, sort_keys=True).encode()
        ).hexdigest()
