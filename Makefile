.PHONY: install test lint format fix typecheck check

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
