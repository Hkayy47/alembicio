"""Postgres control plane (DESIGN.md §3, D1)."""

from __future__ import annotations

from typing import Any

import psycopg
from psycopg import sql
from psycopg.rows import dict_row

from alembicio.core.state import MigrationState, assert_transition
from alembicio.core.worker import ControlPlane, LedgerSnapshot

_SCHEMA = "alembicio"


class PostgresControlPlane:
    """Migration state, ledger, and dead-letter persistence in Postgres."""

    def __init__(self, *, conninfo: str, migration_id: str) -> None:
        """Initialize against a migration id.

        Args:
            conninfo: Postgres DSN.
            migration_id: Migration identifier from yaml.
        """
        self._conninfo = conninfo
        self._migration_id = migration_id

    def _connect(self) -> psycopg.Connection[Any]:
        return psycopg.connect(self._conninfo, row_factory=dict_row)

    def ensure_schema(self) -> None:
        """Create the alembicio control-plane schema if missing."""
        ddl = f"""
        CREATE SCHEMA IF NOT EXISTS {_SCHEMA};
        CREATE TABLE IF NOT EXISTS {_SCHEMA}.migration (
          id           text PRIMARY KEY,
          config_hash  text NOT NULL DEFAULT '',
          state        text NOT NULL DEFAULT 'CREATED',
          created_at   timestamptz NOT NULL DEFAULT now(),
          updated_at   timestamptz NOT NULL DEFAULT now()
        );
        CREATE TABLE IF NOT EXISTS {_SCHEMA}.ledger (
          migration_id text PRIMARY KEY REFERENCES {_SCHEMA}.migration(id),
          tokens_in    bigint NOT NULL DEFAULT 0,
          usd_est      numeric NOT NULL DEFAULT 0,
          updated_at   timestamptz NOT NULL DEFAULT now()
        );
        CREATE TABLE IF NOT EXISTS {_SCHEMA}.dead_letter (
          migration_id text NOT NULL,
          doc_id       text NOT NULL,
          content_hash text NOT NULL,
          reason       text NOT NULL,
          attempts     int  NOT NULL DEFAULT 1,
          last_error   text,
          ts           timestamptz NOT NULL DEFAULT now(),
          PRIMARY KEY (migration_id, doc_id, content_hash)
        );
        CREATE TABLE IF NOT EXISTS {_SCHEMA}.read_state (
          migration_id text PRIMARY KEY,
          active       text NOT NULL DEFAULT 'old',
          canary_pct   int  NOT NULL DEFAULT 0,
          seed         bigint NOT NULL DEFAULT 0
        );
        """
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(ddl)
            cur.execute(
                sql.SQL(
                    """
                    INSERT INTO {}.migration (id, state)
                    VALUES (%s, 'CREATED')
                    ON CONFLICT (id) DO NOTHING
                    """
                ).format(sql.Identifier(_SCHEMA)),
                (self._migration_id,),
            )
            cur.execute(
                sql.SQL(
                    """
                    INSERT INTO {}.ledger (migration_id)
                    VALUES (%s)
                    ON CONFLICT (migration_id) DO NOTHING
                    """
                ).format(sql.Identifier(_SCHEMA)),
                (self._migration_id,),
            )
            conn.commit()

    def get_state(self) -> MigrationState:
        """Return the durable migration state."""
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                sql.SQL("SELECT state FROM {}.migration WHERE id = %s").format(
                    sql.Identifier(_SCHEMA)
                ),
                (self._migration_id,),
            )
            row = cur.fetchone()
            if row is None:
                return "CREATED"
            state = str(row["state"])
            return state  # type: ignore[return-value]

    def set_state(self, state: MigrationState, /) -> None:
        """Transition to ``state``, validating legality."""
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                sql.SQL("SELECT state FROM {}.migration WHERE id = %s FOR UPDATE").format(
                    sql.Identifier(_SCHEMA)
                ),
                (self._migration_id,),
            )
            row = cur.fetchone()
            current: MigrationState = (
                str(row["state"]) if row else "CREATED"  # type: ignore[assignment]
            )
            assert_transition(current, state)
            cur.execute(
                sql.SQL(
                    """
                    INSERT INTO {}.migration (id, config_hash, state)
                    VALUES (%s, '', %s)
                    ON CONFLICT (id) DO UPDATE
                      SET state = EXCLUDED.state, updated_at = now()
                    """
                ).format(sql.Identifier(_SCHEMA)),
                (self._migration_id, state),
            )
            conn.commit()

    def commit_spend(self, *, tokens: int, usd: float) -> None:
        """Add committed spend to the monotone ledger (I5)."""
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                    UPDATE {}.ledger
                       SET tokens_in = tokens_in + %s,
                           usd_est = usd_est + %s,
                           updated_at = now()
                     WHERE migration_id = %s
                    """
                ).format(sql.Identifier(_SCHEMA)),
                (tokens, usd, self._migration_id),
            )
            conn.commit()

    def record_dead_letter(
        self, *, doc_id: str, content_hash: str, reason: str, error: str
    ) -> None:
        """Idempotently record a poison key."""
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                    INSERT INTO {}.dead_letter
                      (migration_id, doc_id, content_hash, reason, attempts, last_error)
                    VALUES (%s, %s, %s, %s, 1, %s)
                    ON CONFLICT (migration_id, doc_id, content_hash) DO UPDATE
                      SET attempts = {}.dead_letter.attempts + 1,
                          last_error = EXCLUDED.last_error,
                          ts = now()
                    """
                ).format(sql.Identifier(_SCHEMA), sql.Identifier(_SCHEMA)),
                (self._migration_id, doc_id, content_hash, reason, error),
            )
            conn.commit()

    def spend(self) -> LedgerSnapshot:
        """Return the current committed spend."""
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    "SELECT tokens_in, usd_est FROM {}.ledger WHERE migration_id = %s"
                ).format(sql.Identifier(_SCHEMA)),
                (self._migration_id,),
            )
            row = cur.fetchone()
            if row is None:
                return LedgerSnapshot(tokens_in=0, usd_est=0.0)
            return LedgerSnapshot(
                tokens_in=int(row["tokens_in"]),
                usd_est=float(row["usd_est"]),
            )

    def dead_letter_keys(self) -> set[tuple[str, str]]:
        """Return ``(doc_id, content_hash)`` keys in the dead-letter table."""
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    "SELECT doc_id, content_hash FROM {}.dead_letter WHERE migration_id = %s"
                ).format(sql.Identifier(_SCHEMA)),
                (self._migration_id,),
            )
            rows = cur.fetchall()
            return {(str(r["doc_id"]), str(r["content_hash"])) for r in rows}


def as_control_plane(plane: PostgresControlPlane) -> ControlPlane:
    """Narrow a concrete plane to the worker protocol."""
    return plane
