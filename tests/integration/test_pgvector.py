"""Integration tests for the pgvector adapter (P4)."""

from __future__ import annotations

import threading
import time

import numpy as np
import psycopg
import pytest

from alembicio.adapters.base import VectorRecord
from alembicio.adapters.pgvector import model_slug, select_vectype
from tests.integration.conftest import make_adapter, md5_hex, requires_postgres, seed_rows

pytestmark = [pytest.mark.integration, requires_postgres]

_TARGET = {"provider": "fastembed", "model": "test-target-model", "dim": 3}


def test_prepare_is_idempotent(pg_dsn: str, pg_table: str) -> None:
    adapter = make_adapter(pg_dsn=pg_dsn, table=pg_table)
    adapter.prepare(_TARGET)
    adapter.prepare(_TARGET)
    slug = model_slug(_TARGET["model"])
    with psycopg.connect(pg_dsn) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT column_name FROM information_schema.columns
             WHERE table_name = %s AND column_name LIKE %s
            """,
            (pg_table, f"emb__{slug}%"),
        )
        cols = {row[0] for row in cur.fetchall()}
    assert f"emb__{slug}" in cols
    assert f"emb__{slug}_hash" in cols


def test_trigger_fires_on_text_update(pg_dsn: str, pg_table: str) -> None:
    adapter = make_adapter(pg_dsn=pg_dsn, table=pg_table)
    adapter.prepare(_TARGET)
    seed_rows(pg_dsn, pg_table, [("d1", "hello")])
    vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    adapter.upsert_vectors(
        [
            VectorRecord(
                doc_id="d1",
                content_hash=md5_hex("hello"),
                vector=vec,
                provenance="embedded",
            )
        ]
    )
    slug = model_slug(_TARGET["model"])
    with psycopg.connect(pg_dsn) as conn, conn.cursor() as cur:
        cur.execute(f"UPDATE public.{pg_table} SET content = %s WHERE id = %s", ("world", "d1"))
        cur.execute(f"SELECT emb__{slug}_hash FROM public.{pg_table} WHERE id = %s", ("d1",))
        assert cur.fetchone()[0] is None
        conn.commit()


def test_trigger_ignores_unrelated_column_update(pg_dsn: str, pg_table: str) -> None:
    adapter = make_adapter(pg_dsn=pg_dsn, table=pg_table)
    adapter.prepare(_TARGET)
    seed_rows(pg_dsn, pg_table, [("d1", "hello")])
    content_hash = md5_hex("hello")
    adapter.upsert_vectors(
        [
            VectorRecord(
                doc_id="d1",
                content_hash=content_hash,
                vector=np.array([1.0, 0.0, 0.0], dtype=np.float32),
                provenance="embedded",
            )
        ]
    )
    slug = model_slug(_TARGET["model"])
    with psycopg.connect(pg_dsn) as conn, conn.cursor() as cur:
        cur.execute(
            f"UPDATE public.{pg_table} SET embedding = %s WHERE id = %s",
            ("[0,1,0]", "d1"),
        )
        cur.execute(
            f"SELECT emb__{slug}_hash FROM public.{pg_table} WHERE id = %s",
            ("d1",),
        )
        assert cur.fetchone()[0] == content_hash
        conn.commit()


def test_upsert_noop_on_hash_mismatch(pg_dsn: str, pg_table: str) -> None:
    adapter = make_adapter(pg_dsn=pg_dsn, table=pg_table)
    adapter.prepare(_TARGET)
    seed_rows(pg_dsn, pg_table, [("d1", "new-text")])
    adapter.upsert_vectors(
        [
            VectorRecord(
                doc_id="d1",
                content_hash=md5_hex("stale"),
                vector=np.array([1.0, 0.0, 0.0], dtype=np.float32),
                provenance="embedded",
            )
        ]
    )
    slug = model_slug(_TARGET["model"])
    with psycopg.connect(pg_dsn) as conn, conn.cursor() as cur:
        cur.execute(f"SELECT emb__{slug} IS NULL FROM public.{pg_table} WHERE id = %s", ("d1",))
        assert cur.fetchone()[0] is True


def test_projected_rows_stay_pending(pg_dsn: str, pg_table: str) -> None:
    adapter = make_adapter(pg_dsn=pg_dsn, table=pg_table, mapping_mode="default")
    adapter.prepare(_TARGET)
    seed_rows(pg_dsn, pg_table, [("d1", "hello")])
    adapter.upsert_vectors(
        [
            VectorRecord(
                doc_id="d1",
                content_hash=md5_hex("hello"),
                vector=np.array([1.0, 0.0, 0.0], dtype=np.float32),
                provenance="projected",
            )
        ]
    )
    assert adapter.pending_count() == 1


def test_halfvec_path_at_3072(pg_dsn: str, pg_table: str) -> None:
    vectype, ops = select_vectype(3072)
    assert vectype == "halfvec"
    assert ops == "halfvec_cosine_ops"
    adapter = make_adapter(pg_dsn=pg_dsn, table=pg_table, dim=3072, target_model="big-model")
    adapter.prepare({"provider": "openai", "model": "big-model", "dim": 3072})
    info = adapter.inspect_pg()
    assert info.vectype == "halfvec"


def test_flip_read_path_atomic_under_concurrent_reader(pg_dsn: str, pg_table: str) -> None:
    adapter = make_adapter(pg_dsn=pg_dsn, table=pg_table)
    adapter.prepare(_TARGET)
    adapter.flip_read_path(active="old")
    seed_rows(pg_dsn, pg_table, [("d1", "hello")])
    errors: list[str] = []

    def reader() -> None:
        conn = psycopg.connect(pg_dsn)
        try:
            with conn.cursor() as cur:
                cur.execute("BEGIN")
                cur.execute(f"SELECT * FROM public.{pg_table}_search LIMIT 1")
                time.sleep(0.2)
                cur.execute(f"SELECT * FROM public.{pg_table}_search LIMIT 1")
                names = [desc[0] for desc in cur.description or []]
                if names.count("embedding") != 1:
                    errors.append(f"expected one embedding column, got {names}")
                cur.execute("ROLLBACK")
        finally:
            conn.close()

    thread = threading.Thread(target=reader)
    thread.start()
    time.sleep(0.05)
    adapter.flip_read_path(active="new")
    thread.join(timeout=5)
    assert not errors


def test_search_old_and_new_spaces(pg_dsn: str, pg_table: str) -> None:
    adapter = make_adapter(pg_dsn=pg_dsn, table=pg_table)
    adapter.prepare(_TARGET)
    seed_rows(pg_dsn, pg_table, [("near", "alpha"), ("far", "beta")])
    with psycopg.connect(pg_dsn) as conn, conn.cursor() as cur:
        cur.execute(
            f"UPDATE public.{pg_table} SET embedding = %s WHERE id = %s",
            ("[1,0,0]", "near"),
        )
        conn.commit()
    new_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    adapter.upsert_vectors(
        [
            VectorRecord(
                doc_id="near",
                content_hash=md5_hex("alpha"),
                vector=new_vec,
                provenance="embedded",
            )
        ]
    )
    assert adapter.search(new_vec, space="old", k=1) == ["near"]
    assert adapter.search(new_vec, space="new", k=1) == ["near"]
