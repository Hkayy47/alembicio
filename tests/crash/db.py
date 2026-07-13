"""Direct-SQL assertion battery for the crash harness (the teeth).

These helpers connect to the target Postgres and assert the post-backfill invariants
*structurally in SQL*, independent of any Python bookkeeping the worker did. They are the
oracle both for the 50-seed resume loop and for the canary bug-detection test, so they
must fail loudly (raise :class:`AssertionError`) the moment an invariant is violated.

Companion-column names are discovered from ``information_schema`` rather than hardcoded,
so the assertions stay correct whatever slug the pgvector adapter chooses (DESIGN.md §5).
"""

from __future__ import annotations

from dataclasses import dataclass

import psycopg


@dataclass(frozen=True)
class CompanionColumns:
    """The pgvector companion columns for one migration (DESIGN.md §5)."""

    slug: str
    vector_col: str
    hash_col: str
    provenance_col: str


def connect(dsn: str) -> psycopg.Connection[tuple[object, ...]]:
    """Open an autocommit connection to the target store.

    Args:
        dsn: Postgres DSN.

    Returns:
        An open autocommit connection.
    """
    return psycopg.connect(dsn, autocommit=True)


def discover_companion(
    conn: psycopg.Connection[tuple[object, ...]], *, table: str
) -> CompanionColumns:
    """Find the ``emb__{slug}`` companion columns on a table.

    Args:
        conn: Open connection.
        table: Target table name (schema ``public``).

    Returns:
        The discovered companion column names.

    Raises:
        AssertionError: If exactly one ``emb__*_hash`` column is not present.
    """
    rows = conn.execute(
        """
        SELECT column_name FROM information_schema.columns
        WHERE table_name = %s AND column_name ~ '^emb__.*_hash$'
        ORDER BY column_name
        """,
        (table,),
    ).fetchall()
    hash_cols = [str(row[0]) for row in rows]
    assert len(hash_cols) == 1, (
        f"expected exactly one emb__*_hash column on {table!r}, found {hash_cols}"
    )
    hash_col = hash_cols[0]
    slug = hash_col[len("emb__") : -len("_hash")]
    return CompanionColumns(
        slug=slug,
        vector_col=f"emb__{slug}",
        hash_col=hash_col,
        provenance_col=f"emb__{slug}_provenance",
    )


def _scalar_int(
    conn: psycopg.Connection[tuple[object, ...]],
    sql: str,
    params: tuple[object, ...],
) -> int:
    row = conn.execute(sql, params).fetchone()
    assert row is not None, f"query returned no row: {sql}"
    return int(row[0])


def assert_no_rows_pending(
    conn: psycopg.Connection[tuple[object, ...]],
    *,
    table: str,
    text_col: str,
    companion: CompanionColumns,
) -> None:
    """Assert zero rows remain pending: every companion hash equals md5 of current text.

    Args:
        conn: Open connection.
        table: Target table.
        text_col: Source text column.
        companion: Discovered companion columns.

    Returns:
        None.

    Raises:
        AssertionError: If any row's companion hash differs from ``md5(text::text)``.
    """
    pending = _scalar_int(
        conn,
        f"SELECT count(*) FROM {table} "
        f"WHERE {companion.hash_col} IS DISTINCT FROM md5({text_col}::text)",
        (),
    )
    assert pending == 0, f"{pending} rows still pending (companion hash != md5(text))"


def assert_no_projected_rows(
    conn: psycopg.Connection[tuple[object, ...]],
    *,
    table: str,
    companion: CompanionColumns,
) -> None:
    """Assert no row is still marked ``provenance = 'projected'`` (I8, non-mapping mode).

    Args:
        conn: Open connection.
        table: Target table.
        companion: Discovered companion columns.

    Returns:
        None.

    Raises:
        AssertionError: If any projected row survives.
    """
    projected = _scalar_int(
        conn,
        f"SELECT count(*) FROM {table} WHERE {companion.provenance_col} = 'projected'",
        (),
    )
    assert projected == 0, f"{projected} rows still have provenance='projected' (I8)"


def assert_state(
    conn: psycopg.Connection[tuple[object, ...]], *, migration_id: str, expected: str
) -> None:
    """Assert the durable migration state equals ``expected``.

    Args:
        conn: Open connection.
        migration_id: Migration id.
        expected: Expected state string.

    Returns:
        None.

    Raises:
        AssertionError: If the recorded state differs.
    """
    row = conn.execute(
        "SELECT state FROM alembicio.migration WHERE id = %s", (migration_id,)
    ).fetchone()
    assert row is not None, f"no migration row for {migration_id!r}"
    assert row[0] == expected, f"state is {row[0]!r}, expected {expected!r}"


def assert_no_duplicate_dead_letters(
    conn: psycopg.Connection[tuple[object, ...]], *, migration_id: str
) -> None:
    """Assert the dead-letter table holds no duplicate ``(doc_id, content_hash)`` keys.

    Args:
        conn: Open connection.
        migration_id: Migration id.

    Returns:
        None.

    Raises:
        AssertionError: If any key appears more than once.
    """
    dupes = _scalar_int(
        conn,
        """
        SELECT COALESCE(sum(c) - count(*), 0) FROM (
            SELECT count(*) AS c FROM alembicio.dead_letter
            WHERE migration_id = %s GROUP BY doc_id, content_hash
        ) g
        """,
        (migration_id,),
    )
    assert dupes == 0, f"{dupes} duplicate dead-letter rows for {migration_id!r}"


def ledger_tokens(
    conn: psycopg.Connection[tuple[object, ...]], *, migration_id: str
) -> int:
    """Return the committed ``tokens_in`` for a migration (0 if no ledger row yet)."""
    row = conn.execute(
        "SELECT tokens_in FROM alembicio.ledger WHERE migration_id = %s",
        (migration_id,),
    ).fetchone()
    return 0 if row is None else int(row[0])


def assert_ledger_within_one_batch(
    conn: psycopg.Connection[tuple[object, ...]],
    *,
    migration_id: str,
    baseline_tokens: int,
    batch_estimate_tokens: int,
    replay_floor: int,
) -> int:
    """Assert ledger ``tokens_in`` is within one batch of baseline and not below the floor.

    A crash between a durable upsert and its ledger commit under-counts by at most one
    batch (I1/I5); it must never *over*-count, and across replays it must never decrease.

    Args:
        conn: Open connection.
        migration_id: Migration id.
        baseline_tokens: ``tokens_in`` from a clean, no-crash run.
        batch_estimate_tokens: Token estimate for a single batch.
        replay_floor: The highest ``tokens_in`` seen on a previous replay.

    Returns:
        The observed ``tokens_in`` (to carry forward as the next replay floor).

    Raises:
        AssertionError: If the ledger over-counts, drops below one-batch tolerance, or
            regresses below the replay floor.
    """
    actual = ledger_tokens(conn, migration_id=migration_id)
    tolerance = batch_estimate_tokens + 1
    assert actual <= baseline_tokens, (
        f"ledger over-counted: {actual} > baseline {baseline_tokens}"
    )
    assert baseline_tokens - actual <= tolerance, (
        f"ledger under-counted by more than one batch: baseline {baseline_tokens}, "
        f"actual {actual}, tolerance {tolerance}"
    )
    assert actual >= replay_floor, (
        f"ledger regressed on replay: {actual} < previous {replay_floor}"
    )
    return actual


def assert_backfill_complete(
    conn: psycopg.Connection[tuple[object, ...]],
    *,
    migration_id: str,
    table: str,
    text_col: str,
    baseline_tokens: int,
    batch_estimate_tokens: int,
    replay_floor: int = 0,
) -> int:
    """Assert every post-backfill invariant at once (the harness oracle).

    Args:
        conn: Open connection.
        migration_id: Migration id.
        table: Target table.
        text_col: Source text column.
        baseline_tokens: ``tokens_in`` from a clean, no-crash run.
        batch_estimate_tokens: Token estimate for a single batch.
        replay_floor: Highest ``tokens_in`` seen on a previous replay.

    Returns:
        The observed ledger ``tokens_in``.

    Raises:
        AssertionError: On the first invariant that fails.
    """
    companion = discover_companion(conn, table=table)
    assert_no_rows_pending(conn, table=table, text_col=text_col, companion=companion)
    assert_no_projected_rows(conn, table=table, companion=companion)
    assert_no_duplicate_dead_letters(conn, migration_id=migration_id)
    assert_state(conn, migration_id=migration_id, expected="BACKFILLED")
    return assert_ledger_within_one_batch(
        conn,
        migration_id=migration_id,
        baseline_tokens=baseline_tokens,
        batch_estimate_tokens=batch_estimate_tokens,
        replay_floor=replay_floor,
    )
