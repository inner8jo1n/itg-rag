.PHONY: install test lint format fix typecheck check up down load chunks ingest stats search ask demo evaluate compare verify-live integration

QUESTION ?= В какой срок можно вернуть исправный товар в Полку?

install:
	uv sync --locked

test:
	uv run pytest

lint:
	uv run ruff check .

format:
	uv run ruff format .

fix:
	uv run ruff check --fix .
	uv run ruff format .

typecheck:
	uv run ty check

check:
	@./scripts/check.sh

up:
	docker compose up -d --wait

down:
	docker compose down

load:
	uv run itg-rag load

chunks:
	uv run itg-rag chunks

ingest:
	uv run itg-rag ingest

stats:
	uv run itg-rag stats

search:
	uv run itg-rag search "$(QUESTION)"

ask:
	uv run itg-rag ask "$(QUESTION)"

demo:
	uv run itg-rag demo --output results/demo.json

evaluate:
	uv run itg-rag evaluate --retrieval-only --output results/retrieval.json

compare:
	uv run itg-rag evaluate --compare --output results/comparison.json

verify-live:
	uv run python scripts/verify_live.py

integration:
	RUN_WEAVIATE_TESTS=1 uv run pytest -q -m integration
