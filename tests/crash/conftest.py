"""Fixtures for the crash harness: Postgres reachability, config, stack-readiness gate."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

_COMPOSE_DSN = "postgresql://alembicio:alembicio@localhost:5432/alembicio"
_GOLDEN = Path("examples/golden.example.jsonl").resolve()


def candidate_dsn() -> str:
    """Return the DSN to test against (env override, else the docker-compose default)."""
    return os.environ.get("DATABASE_URL", _COMPOSE_DSN)


def write_crash_config(path: Path, *, table: str, batch_size: int = 128) -> Path:
    """Write a keyless fastembed migration config for the crash corpus.

    Args:
        path: Destination yaml path.
        table: Target table name.
        batch_size: Backfill batch size.

    Returns:
        The written path.
    """
    path.write_text(
        f"""migration: crash-harness-{table}
source: {{ provider: fastembed, model: sentence-transformers/all-MiniLM-L6-v2, dim: 384 }}
target: {{ provider: fastembed, model: BAAI/bge-small-en-v1.5, dim: 384 }}
store:
  kind: pgvector
  dsn: env:DATABASE_URL
  table: {table}
  id_column: id
  text_column: content
backfill:
  batch_size: {batch_size}
  budget: {{ max_usd: 0, max_tokens: 200000000 }}
  rate_limit: {{ tpm: 0, rpm: 0 }}
  on_poison: dead_letter
verify:
  golden_queries: {_GOLDEN.as_posix()}
  gates: {{ min_recall_ratio: 1.0, k: 10, report: report.md }}
cutover: {{ mode: staged, canary_pct: 5, soak_hours: 72 }}
mapping: {{ kind: none, anchors: 4096, min_recovery: 0.95 }}
""",
        encoding="utf-8",
    )
    return path


@pytest.fixture(scope="session")
def dsn() -> str:
    """The Postgres DSN under test."""
    return candidate_dsn()


@pytest.fixture(scope="session")
def pg(dsn: str) -> Iterator[object]:
    """A live autocommit connection, or skip the test if Postgres is unreachable."""
    psycopg = pytest.importorskip("psycopg")
    try:
        conn = psycopg.connect(dsn, autocommit=True, connect_timeout=3)
    except Exception as exc:  # noqa: BLE001 - any connect failure means "skip"
        pytest.skip(f"Postgres not reachable at {dsn}: {exc}")
    try:
        yield conn
    finally:
        conn.close()


@pytest.fixture
def crash_config(tmp_path: Path) -> Path:
    """A migration config targeting the crash corpus table."""
    return write_crash_config(tmp_path / "embmigrate.yaml", table="crash_documents")


@pytest.fixture(scope="session")
def stack_ready(dsn: str, tmp_path_factory: pytest.TempPathFactory) -> bool:
    """Whether the CLI/adapter/provider execution stack is wired (probed once)."""
    from tests.crash.runner import backfill_stack_ready

    cfg = write_crash_config(
        tmp_path_factory.mktemp("stackcfg") / "embmigrate.yaml", table="crash_documents"
    )
    return backfill_stack_ready(dsn=dsn, config_path=str(cfg))
