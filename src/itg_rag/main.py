import argparse
import json
import logging
import sys
from contextlib import ExitStack
from pathlib import Path

from pydantic import ValidationError

from itg_rag.api import client
from itg_rag.chunking import chunk_documents
from itg_rag.config import Settings
from itg_rag.embedder import Embedder
from itg_rag.errors import RAGError
from itg_rag.evaluation import evaluate, load_cases
from itg_rag.llm_client import LLMClient
from itg_rag.loader import load_documents
from itg_rag.rag_pipeline import RAGPipeline
from itg_rag.retriever import Retriever, validate_question
from itg_rag.weaviate_store import connect_store


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        description="RAG on local documents with Weaviate and cited evidence"
    )
    root.add_argument("--verbose", action="store_true")
    commands = root.add_subparsers(dest="command", required=True)
    for name in ("load", "chunks", "ingest", "stats"):
        sub = commands.add_parser(name)
        sub.add_argument("--output", type=Path)
    for name in ("search", "ask"):
        sub = commands.add_parser(name)
        sub.add_argument("question")
        sub.add_argument("--top-k", type=int)
        sub.add_argument("--output", type=Path)
    for name in ("demo", "evaluate"):
        sub = commands.add_parser(name)
        sub.add_argument(
            "--questions", type=Path, default=Path("demo/questions.json")
        )
        sub.add_argument("--retrieval-only", action="store_true")
        sub.add_argument("--compare", action="store_true")
        sub.add_argument("--output", type=Path)
    return root


def emit(value: object, path: Path | None) -> None:
    text = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    if path is None:
        print(text, end="")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        print(f"Saved: {path}")


def execute(args: argparse.Namespace, settings: Settings) -> dict | list:
    if args.command in {"load", "chunks", "ingest"}:
        documents = load_documents(settings.knowledge_dir)
        if args.command == "load":
            return [
                {
                    "document_id": doc.document_id,
                    "source_name": doc.source_name,
                    "characters": len(doc.text),
                }
                for doc in documents
            ]
        chunks = chunk_documents(
            documents, settings.chunk_size, settings.chunk_overlap
        )
        if args.command == "chunks":
            return [chunk.model_dump() for chunk in chunks]
    if args.command in {"search", "ask"}:
        validate_question(args.question)
        if args.top_k is not None:
            settings = Settings.model_validate(
                settings.model_dump() | {"top_k": args.top_k}
            )
    with ExitStack() as stack:
        store = stack.enter_context(connect_store(settings))
        if args.command == "stats":
            store.ensure()
            return {
                "collection": settings.collection,
                "objects": store.count(),
            }
        embedding_client = stack.enter_context(
            client(settings, embedding=True)
        )
        embedder = Embedder(embedding_client, settings.embedding_model)
        if args.command == "ingest":
            store.ensure(create=True)
            vectors = embedder.embed([chunk.text for chunk in chunks])
            return {"documents": len(documents), **store.sync(chunks, vectors)}
        store.ensure()
        retriever = Retriever(embedder, store, settings.top_k)
        if args.command == "search":
            return [
                hit.model_dump() for hit in retriever.retrieve(args.question)
            ]
        if args.command in {"demo", "evaluate"}:
            cases = load_cases(args.questions)
            if args.retrieval_only and args.compare:
                raise RAGError(
                    "--compare requires generation; omit --retrieval-only"
                )
        llm = None
        if args.command == "ask" or not args.retrieval_only:
            chat_client = stack.enter_context(client(settings))
            llm = LLMClient(chat_client, settings.model)
        pipeline = RAGPipeline(retriever, llm, settings.max_distance)
        if args.command == "ask":
            return pipeline.ask(args.question).model_dump()
        return evaluate(
            pipeline,
            cases,
            retrieval_only=args.retrieval_only,
            baseline=llm if args.compare else None,
        )


def main() -> int:
    args = parser().parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    for name in ("httpx", "httpx2", "httpcore", "httpcore2", "weaviate"):
        logging.getLogger(name).setLevel(logging.ERROR)
    try:
        result = execute(args, Settings.from_env())
        emit(result, args.output)
        return 1 if isinstance(result, dict) and result.get("errors") else 0
    except ValidationError as error:
        fields = ", ".join(
            ".".join(map(str, e["loc"])) for e in error.errors()
        )
        print(f"Invalid configuration or data: {fields}", file=sys.stderr)
    except (RAGError, OSError, UnicodeError, json.JSONDecodeError) as error:
        print(f"Error: {error}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
