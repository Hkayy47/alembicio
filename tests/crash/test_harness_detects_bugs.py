"""Canary: the SQL oracle must fail loudly on a checkpoint-before-write bug.

A crash harness is only worth its runtime if it *catches* the bug it exists to catch.
This test bootstraps a minimal pgvector-shaped schema and drives two in-process workers
against it -- one correct, one that advances progress (ledger/state) before the companion
writes are durable (violating I2, "durability before progress") -- and asserts that the
oracle in :mod:`tests.crash.db` passes the former and raises on the latter.

Unlike :mod:`tests.crash.test_crash_resume`, this runs today: it needs only a reachable
Postgres, not the full CLI/adapter/provider stack. When the real worker lands, the same
oracle guards it unchanged; here we substitute a hand-written buggy worker to prove the
oracle's teeth (the "monkeypatched worker" of the P3 acceptance).
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator

import pytest

from tests.crash import db

_TABLE = "canary_documents"
_MIGRATION_ID = "canary-migration"
_SLUG = "bge_small"
_HASH_COL = f"emb__{_SLUG}_hash"
_PROV_COL = f"emb__{_SLUG}_provenance"
_ROWS = 12
_BASELINE_TOKENS = 120
_BATCH_ESTIMATE = 20


def _md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _bootstrap(conn: object) -> None:
    """Create a minimal control plane and companion-column table, seeded with rows."""
    execute = conn.execute  # type: ignore[attr-defined]
    execute("CREATE SCHEMA IF NOT EXISTS alembicio")
    execute(
        "CREATE TABLE IF NOT EXISTS alembicio.migration ("
        " id text PRIMARY KEY, config_hash text, state text NOT NULL,"
        " created_at timestamptz DEFAULT now(), updated_at timestamptz DEFAULT now())"
    )
    execute(
        "CREATE TABLE IF NOT EXISTS alembicio.ledger ("
        " migration_id text PRIMARY KEY, tokens_in bigint NOT NULL DEFAULT 0,"
        " usd_est numeric NOT NULL DEFAULT 0, updated_at timestamptz DEFAULT now())"
    )
    execute(
        "CREATE TABLE IF NOT EXISTS alembicio.dead_letter ("
        " migration_id text, doc_id text, content_hash text, reason text,"
        " attempts int, last_error text, ts timestamptz DEFAULT now(),"
        " PRIMARY KEY (migration_id, doc_id, content_hash))"
    )
    execute(f"DROP TABLE IF EXISTS {_TABLE} CASCADE")
    execute(
        f"CREATE TABLE {_TABLE} ("
        f" id text PRIMARY KEY, content text NOT NULL,"
        f" {_HASH_COL} text, {_PROV_COL} text NOT NULL DEFAULT 'embedded')"
    )
    with (  # type: ignore[attr-defined]
        conn.cursor() as cur,
        cur.copy(f"COPY {_TABLE} (id, content) FROM STDIN") as copy,
    ):
        for i in range(_ROWS):
            copy.write_row((f"doc_{i:03d}", f"canary body number {i}"))

    execute(
        "INSERT INTO alembicio.migration (id, config_hash, state) VALUES (%s, %s, %s) "
        "ON CONFLICT (id) DO UPDATE SET state = EXCLUDED.state",
        (_MIGRATION_ID, "cfg", "BACKFILLING"),
    )
    execute(
        "INSERT INTO alembicio.ledger (migration_id, tokens_in) VALUES (%s, 0) "
        "ON CONFLICT (migration_id) DO UPDATE SET tokens_in = 0",
        (_MIGRATION_ID,),
    )
    execute(
        "DELETE FROM alembicio.dead_letter WHERE migration_id = %s", (_MIGRATION_ID,)
    )


def _correct_worker(conn: object) -> None:
    """A correct worker: write every companion hash durably, THEN advance progress (I2)."""
    execute = conn.execute  # type: ignore[attr-defined]
    execute(
        f"UPDATE {_TABLE} SET {_HASH_COL} = md5(content::text), {_PROV_COL} = 'embedded'"
    )
    execute(
        "UPDATE alembicio.ledger SET tokens_in = %s WHERE migration_id = %s",
        (_BASELINE_TOKENS, _MIGRATION_ID),
    )
    execute(
        "UPDATE alembicio.migration SET state = 'BACKFILLED' WHERE id = %s",
        (_MIGRATION_ID,),
    )


def _checkpoint_before_write_worker(conn: object) -> None:
    """The bug: mark the migration BACKFILLED while some rows are still unwritten (I2)."""
    execute = conn.execute  # type: ignore[attr-defined]
    # Only the first half get their companion hash written...
    execute(
        f"UPDATE {_TABLE} SET {_HASH_COL} = md5(content::text) "
        f"WHERE id < %s",
        (f"doc_{_ROWS // 2:03d}",),
    )
    # ...but progress is checkpointed as if the whole run completed.
    execute(
        "UPDATE alembicio.ledger SET tokens_in = %s WHERE migration_id = %s",
        (_BASELINE_TOKENS, _MIGRATION_ID),
    )
    execute(
        "UPDATE alembicio.migration SET state = 'BACKFILLED' WHERE id = %s",
        (_MIGRATION_ID,),
    )


def _ledger_overcount_worker(conn: object) -> None:
    """A worker that double-counts a replayed batch (ledger over baseline, I1)."""
    _correct_worker(conn)
    conn.execute(  # type: ignore[attr-defined]
        "UPDATE alembicio.ledger SET tokens_in = %s WHERE migration_id = %s",
        (_BASELINE_TOKENS + _BATCH_ESTIMATE + 5, _MIGRATION_ID),
    )


@pytest.fixture
def canary_db(pg: object) -> Iterator[object]:
    """Bootstrap the canary schema on the shared connection for each test."""
    _bootstrap(pg)
    yield pg


def _run_oracle(conn: object, *, replay_floor: int = 0) -> int:
    return db.assert_backfill_complete(
        conn,
        migration_id=_MIGRATION_ID,
        table=_TABLE,
        text_col="content",
        baseline_tokens=_BASELINE_TOKENS,
        batch_estimate_tokens=_BATCH_ESTIMATE,
        replay_floor=replay_floor,
    )


def test_oracle_passes_on_correct_completion(canary_db: object) -> None:
    """Sanity: a correctly completed run passes every assertion (no false positives)."""
    _correct_worker(canary_db)
    observed = _run_oracle(canary_db)
    assert observed == _BASELINE_TOKENS


def test_oracle_detects_checkpoint_before_write(canary_db: object) -> None:
    """The headline canary: progress advanced before writes are durable is caught."""
    _checkpoint_before_write_worker(canary_db)
    with pytest.raises(AssertionError, match="pending"):
        _run_oracle(canary_db)


def test_oracle_detects_ledger_overcount(canary_db: object) -> None:
    """A double-counted (replayed) batch pushes the ledger over baseline and is caught."""
    _ledger_overcount_worker(canary_db)
    with pytest.raises(AssertionError, match="over-counted"):
        _run_oracle(canary_db)


def test_oracle_detects_ledger_regression_on_replay(canary_db: object) -> None:
    """A ledger that moves backward across replays is caught (monotonicity, I5)."""
    _correct_worker(canary_db)
    with pytest.raises(AssertionError, match="regressed"):
        _run_oracle(canary_db, replay_floor=_BASELINE_TOKENS + 1)


def test_oracle_detects_wrong_state(canary_db: object) -> None:
    """A run left in the wrong state (e.g. still BACKFILLING) is caught."""
    conn = canary_db
    conn.execute(  # type: ignore[attr-defined]
        f"UPDATE {_TABLE} SET {_HASH_COL} = md5(content::text)"
    )
    conn.execute(  # type: ignore[attr-defined]
        "UPDATE alembicio.ledger SET tokens_in = %s WHERE migration_id = %s",
        (_BASELINE_TOKENS, _MIGRATION_ID),
    )
    # state deliberately left at BACKFILLING.
    with pytest.raises(AssertionError, match="state is"):
        _run_oracle(conn)
