"""Shared fixtures for docker Postgres integration tests."""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator

import psycopg
import pytest

from alembicio.adapters.pgvector import PgVectorAdapter
from alembicio.config import StoreConfig
from alembicio.core.control_plane import PostgresControlPlane

DEFAULT_DSN = "postgresql://alembicio:alembicio@localhost:5432/alembicio"


def _dsn() -> str:
    return os.environ.get("DATABASE_URL", DEFAULT_DSN)


def postgres_available() -> bool:
    """Return True if docker Postgres accepts connections."""
    try:
        with psycopg.connect(_dsn(), connect_timeout=2):
            return True
    except psycopg.Error:
        return False


requires_postgres = pytest.mark.skipif(
    not postgres_available(),
    reason="Postgres is not available (run `make up`)",
)


@pytest.fixture
def pg_dsn() -> str:
    """Return the integration Postgres DSN."""
    if not postgres_available():
        pytest.skip("Postgres is not available")
    return _dsn()


@pytest.fixture
def pg_table(pg_dsn: str) -> Iterator[str]:
    """Create an isolated documents table and tear it down after the test."""
    table = f"docs_{uuid.uuid4().hex[:10]}"
    with psycopg.connect(pg_dsn) as conn, conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute(
            f"""
            CREATE TABLE public.{table} (
              id text PRIMARY KEY,
              content text NOT NULL,
              embedding vector(3)
            )
            """
        )
        conn.commit()
    yield table
    with psycopg.connect(pg_dsn) as conn, conn.cursor() as cur:
        cur.execute(f"DROP VIEW IF EXISTS public.{table}_search")
        cur.execute(f"DROP TABLE IF EXISTS public.{table} CASCADE")
        conn.commit()


def make_adapter(
    *,
    pg_dsn: str,
    table: str,
    target_model: str = "test-target-model",
    dim: int = 3,
    mapping_mode: str = "default",
) -> PgVectorAdapter:
    """Build a configured adapter against a throwaway table."""
    store = StoreConfig(
        kind="pgvector",
        dsn=pg_dsn,
        table=table,
        id_column="id",
        text_column="content",
        old_embedding_column="embedding",
    )
    plane = PostgresControlPlane(conninfo=pg_dsn, migration_id=f"m-{table}")
    plane.ensure_schema()
    return PgVectorAdapter(
        conninfo=pg_dsn,
        store=store,
        target={"provider": "fastembed", "model": target_model, "dim": dim},
        migration_id=f"m-{table}",
        mapping_mode=mapping_mode,  # type: ignore[arg-type]
    )


def seed_rows(pg_dsn: str, table: str, rows: list[tuple[str, str]]) -> None:
    """Insert id/content rows."""
    with psycopg.connect(pg_dsn) as conn, conn.cursor() as cur:
        for doc_id, content in rows:
            cur.execute(
                f"INSERT INTO public.{table} (id, content) VALUES (%s, %s)",
                (doc_id, content),
            )
        conn.commit()


def md5_hex(text: str) -> str:
    """Mirror Postgres md5(content::text)."""
    import hashlib

    return hashlib.md5(text.encode()).hexdigest()
