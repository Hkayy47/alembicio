.PHONY: up down test lint type crash sync

sync:
	uv sync --all-extras

up:
	docker compose up -d

down:
	docker compose down

test:
	uv run pytest

lint:
	uv run ruff check alembicio tests

type:
	uv run mypy alembicio

crash:
	uv run pytest tests/crash -q
