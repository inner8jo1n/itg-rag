import json
import os
import subprocess
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

from itg_rag.api import client
from itg_rag.chunking import chunk_documents
from itg_rag.config import Settings
from itg_rag.embedder import Embedder
from itg_rag.evaluation import evaluate, load_cases
from itg_rag.llm_client import LLMClient
from itg_rag.loader import load_documents
from itg_rag.rag_pipeline import RAGPipeline
from itg_rag.retriever import Retriever
from itg_rag.weaviate_store import WeaviateStore, connect_store


def save(name: str, value: dict) -> None:
    path = Path("demo") / name
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved {path}", flush=True)


def write_demo_answers(report: dict) -> None:
    lines = [
        "# Примеры реального запуска",
        "",
        f"Проверено: {report['verified_at']}. Chat: {report['model']}; "
        f"embeddings: {report['embedding_model']}.",
        "",
        "Это сохранённые результаты live-запросов, а не offline-эмуляция.",
        "Ответы без retrieval приведены для сравнения и могут быть неверны.",
    ]
    for index, row in enumerate(report["rows"], 1):
        lines.extend(["", f"## {index}. {row['question']}", ""])
        if "error" in row:
            lines.append(f"Ошибка: {row['error']}")
            continue
        result = row["result"]
        lines.extend(
            [
                f"Статус: `{result['status']}`.",
                "",
                result["answer"],
                "",
            ]
        )
        for source in result["sources"]:
            lines.append(
                f"- [{source['label']}] `{source['source_name']}`, "
                f"chunk `{source['chunk_id']}`, "
                f"distance {source['distance']:.4f}."
            )
        if "without_retrieval" in row:
            lines.extend(["", "Без retrieval:", "", row["without_retrieval"]])
    Path("demo/answers.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    settings = Settings.from_env()
    documents = load_documents(settings.knowledge_dir)
    cases = load_cases(Path("demo/questions.json"))
    with (
        client(settings, embedding=True) as embedding_client,
        client(settings) as chat_client,
        connect_store(settings) as store,
    ):
        embedder = Embedder(embedding_client, settings.embedding_model)
        llm = LLMClient(chat_client, settings.model)
        chunks = chunk_documents(
            documents, settings.chunk_size, settings.chunk_overlap
        )
        vectors = embedder.embed([chunk.text for chunk in chunks])
        first = store.sync(chunks, vectors)
        second = store.sync(chunks, vectors)
        if second["inserted"] or second["indexed"] != len(chunks):
            raise RuntimeError("Repeated ingestion is not idempotent")
        save(
            "index_verification.json",
            {
                "documents": len(documents),
                "chunks": len(chunks),
                "embedding_dimensions": len(vectors[0]),
                "first_sync": first,
                "second_sync": second,
            },
        )
        pipeline = RAGPipeline(
            Retriever(embedder, store, settings.top_k),
            llm,
            settings.max_distance,
        )
        print("Running live RAG and no-retrieval comparison...", flush=True)
        report = evaluate(pipeline, cases, baseline=llm)
        report["verified_at"] = datetime.now(UTC).isoformat()
        report["model"] = settings.model
        report["embedding_model"] = settings.embedding_model
        save("live_results.json", report)
        write_demo_answers(report)
        print(
            f"Correct behavior: {report['correct_behavior_count']}"
            f"/{report['total']}; "
            f"errors: {report['errors']}",
            flush=True,
        )
        profiles = []
        for size, overlap, top_k in [
            (500, 80, 3),
            (900, 150, 4),
            (1400, 200, 4),
        ]:
            print(
                f"Retrieval experiment: size={size}, overlap={overlap}",
                flush=True,
            )
            profile = settings.model_copy(
                update={
                    "chunk_size": size,
                    "chunk_overlap": overlap,
                    "top_k": top_k,
                    "collection": "ExperimentRag" + uuid.uuid4().hex,
                }
            )
            experiment = WeaviateStore(store.client, profile)
            try:
                sample = chunk_documents(documents, size, overlap)
                experiment.sync(
                    sample, embedder.embed([c.text for c in sample])
                )
                evaluation = evaluate(
                    RAGPipeline(
                        Retriever(embedder, experiment, top_k),
                        None,
                        profile.max_distance,
                    ),
                    cases,
                    retrieval_only=True,
                )
                profiles.append(
                    {
                        "chunk_size": size,
                        "overlap": overlap,
                        "chunk_count": len(sample),
                        **evaluation,
                    }
                )
            finally:
                if store.client.collections.exists(profile.collection):
                    store.client.collections.delete(profile.collection)
        save("retrieval_experiments.json", {"profiles": profiles})
    subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-m", "integration"],
        env=os.environ | {"RUN_WEAVIATE_TESTS": "1"},
        check=True,
    )
    if (
        report["errors"]
        or report["correct_behavior_count"] != len(cases)
        or not all(row.get("expected_terms_found") for row in report["rows"])
    ):
        raise RuntimeError(
            "Live evaluation needs review; see demo/live_results.json"
        )


if __name__ == "__main__":
    main()
