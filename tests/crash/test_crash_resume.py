"""The 50-seed crash/resume loop against real Postgres (INVARIANTS I13).

For each seed: seed a 5k-row corpus, run ``init``/``prepare``, take a no-crash baseline
of ``ledger.tokens_in``, then run ``backfill`` as a subprocess and SIGKILL it mid-run at
the injected fault window; resume with ``backfill --resume`` to completion. After
completion, assert *directly in SQL* (see :mod:`tests.crash.db`): zero rows pending;
every companion hash equals md5 of current text; ledger tokens within one batch of
baseline and never lower across the replay; no duplicate dead-letters; state == BACKFILLED.

This test drives the real ``alembicio`` binary end-to-end, so it activates only once the
pgvector adapter, fastembed provider, and ``init``/``prepare``/``backfill`` verbs are
wired. Until then :func:`tests.crash.runner.backfill_stack_ready` reports the stack as
unwired and the test skips cleanly rather than passing vacuously.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from tests.crash import corpus, db, runner

_SEEDS = list(range(50))
_ROWS = 5_000
_BATCH_SIZE = 128
_TABLE = "crash_documents"
_ID_COL = "id"
_TEXT_COL = "content"
_MIGRATION_ID = f"crash-harness-{_TABLE}"


def _batch_estimate_tokens(conn: object, baseline_tokens: int) -> int:
    """Estimate one batch's token cost from the baseline (batches ~= rows / batch_size)."""
    batches = max(1, _ROWS // _BATCH_SIZE)
    return max(1, baseline_tokens // batches)


@pytest.mark.parametrize("seed", _SEEDS)
def test_crash_resume_preserves_invariants(
    seed: int,
    dsn: str,
    pg: object,
    crash_config: Path,
    stack_ready: bool,
) -> None:
    """One seed of run -> kill -> resume -> complete, verified structurally in SQL."""
    if not stack_ready:
        pytest.skip(
            "backfill execution stack (pgvector adapter / fastembed provider / CLI "
            "init+prepare+backfill) is not wired yet; harness skips until it lands"
        )

    rng = random.Random(seed)
    conn = pg  # live autocommit connection from the fixture
    config_path = str(crash_config)

    # 1. Fresh deterministic corpus for this seed.
    corpus.create_base_table(conn, table=_TABLE, id_col=_ID_COL, text_col=_TEXT_COL)
    corpus.seed_corpus(
        conn, table=_TABLE, id_col=_ID_COL, text_col=_TEXT_COL, rows=_ROWS, seed=seed
    )

    # 2. init + prepare (idempotent; safe to re-run).
    assert runner.run_verb("init", dsn=dsn, config_path=config_path).returncode == 0
    assert runner.run_verb("prepare", dsn=dsn, config_path=config_path).returncode == 0

    # 3. No-crash baseline: run backfill to completion, record the ledger, then reset the
    #    companion state so the crashed run starts from the same pending set.
    assert runner.run_verb("backfill", dsn=dsn, config_path=config_path).returncode == 0
    baseline_tokens = db.ledger_tokens(conn, migration_id=_MIGRATION_ID)
    batch_estimate = _batch_estimate_tokens(conn, baseline_tokens)
    _reset_progress(conn)

    # 4. Crash run: kill mid-backfill at a random fault window.
    fault_after = rng.randint(1, max(1, _ROWS // _BATCH_SIZE - 1))
    killed = runner.run_backfill_with_kill(
        dsn=dsn,
        config_path=config_path,
        fault_after_n_batches=fault_after,
        intra_batch_sleep_ms=rng.randint(50, 300),
        kill_after_seconds=rng.uniform(0.5, 4.0),
    )

    # 5. Resume to completion (safe from any crash point, I2/I6).
    resumed = runner.run_verb(
        "backfill", dsn=dsn, config_path=config_path, extra_args=("--resume",)
    )
    assert resumed.returncode == 0, resumed.stderr

    # 6. Structural SQL oracle: every post-backfill invariant holds.
    observed = db.assert_backfill_complete(
        conn,
        migration_id=_MIGRATION_ID,
        table=_TABLE,
        text_col=_TEXT_COL,
        baseline_tokens=baseline_tokens,
        batch_estimate_tokens=batch_estimate,
        replay_floor=0,
    )
    assert observed <= baseline_tokens
    # The crash must actually have happened for this seed to be meaningful.
    assert killed.killed, "process finished before the injected kill window"


def _reset_progress(conn: object) -> None:
    """Clear companion hashes and control-plane progress to re-run from a clean pending set."""
    companion = db.discover_companion(conn, table=_TABLE)  # type: ignore[arg-type]
    conn.execute(  # type: ignore[attr-defined]
        f"UPDATE {_TABLE} SET {companion.hash_col} = NULL, "
        f"{companion.vector_col} = NULL, {companion.provenance_col} = 'embedded'"
    )
    conn.execute(  # type: ignore[attr-defined]
        "UPDATE alembicio.ledger SET tokens_in = 0, usd_est = 0 WHERE migration_id = %s",
        (_MIGRATION_ID,),
    )
    conn.execute(  # type: ignore[attr-defined]
        "DELETE FROM alembicio.dead_letter WHERE migration_id = %s", (_MIGRATION_ID,)
    )
    conn.execute(  # type: ignore[attr-defined]
        "UPDATE alembicio.migration SET state = 'PREPARED' WHERE id = %s", (_MIGRATION_ID,)
    )
