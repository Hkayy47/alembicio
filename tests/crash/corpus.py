"""Synthetic corpus construction for the crash harness.

Builds a deterministic base table (id + text) that ``alembicio prepare`` later augments
with companion columns and a dual-write trigger. The corpus is seeded from a fixed RNG so
the no-crash baseline and every crashed replay operate on identical content.
"""

from __future__ import annotations

import random

import psycopg

_WORD_BANK = (
    "vector embedding migration postgres pgvector recall latency budget ledger cutover "
    "canary rollback soak reconcile checkpoint idempotent tombstone dirty backfill "
    "provenance projected anchor procrustes ridge golden overlap dimension halfvec"
)
_WORDS = _WORD_BANK.split()


def create_base_table(
    conn: psycopg.Connection[tuple[object, ...]],
    *,
    table: str,
    id_col: str,
    text_col: str,
) -> None:
    """Drop and recreate the base document table (no companion columns yet).

    Args:
        conn: Open connection.
        table: Target table name.
        id_col: Primary-key column name.
        text_col: Source text column name.

    Returns:
        None.
    """
    conn.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
    conn.execute(
        f"CREATE TABLE {table} ("
        f"  {id_col} text PRIMARY KEY,"
        f"  {text_col} text NOT NULL"
        f")"
    )


def seed_corpus(
    conn: psycopg.Connection[tuple[object, ...]],
    *,
    table: str,
    id_col: str,
    text_col: str,
    rows: int,
    seed: int,
) -> None:
    """Insert ``rows`` deterministic synthetic documents.

    Args:
        conn: Open connection.
        table: Target table name.
        id_col: Primary-key column name.
        text_col: Source text column name.
        rows: Number of rows to insert.
        seed: RNG seed for reproducible content.

    Returns:
        None.
    """
    rng = random.Random(seed)
    with (
        conn.cursor() as cur,
        cur.copy(f"COPY {table} ({id_col}, {text_col}) FROM STDIN") as copy,
    ):
        for i in range(rows):
            length = rng.randint(6, 40)
            body = " ".join(rng.choice(_WORDS) for _ in range(length))
            copy.write_row((f"doc_{i:05d}", body))


def update_random_rows(
    conn: psycopg.Connection[tuple[object, ...]],
    *,
    table: str,
    id_col: str,
    text_col: str,
    count: int,
    seed: int,
) -> list[str]:
    """Update ``count`` random rows' text (exercises the dirty/update path, I4).

    Args:
        conn: Open connection.
        table: Target table name.
        id_col: Primary-key column name.
        text_col: Source text column name.
        count: Number of rows to update.
        seed: RNG seed.

    Returns:
        The list of updated document ids.
    """
    rng = random.Random(seed)
    total = _row_count(conn, table=table)
    ids = [f"doc_{rng.randrange(total):05d}" for _ in range(count)]
    for doc_id in ids:
        new_text = "UPDATED " + " ".join(rng.choice(_WORDS) for _ in range(rng.randint(6, 40)))
        conn.execute(
            f"UPDATE {table} SET {text_col} = %s WHERE {id_col} = %s",
            (new_text, doc_id),
        )
    return ids


def _row_count(conn: psycopg.Connection[tuple[object, ...]], *, table: str) -> int:
    row = conn.execute(f"SELECT count(*) FROM {table}").fetchone()
    assert row is not None
    return int(row[0])
