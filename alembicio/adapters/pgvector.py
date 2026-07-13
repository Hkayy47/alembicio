"""Postgres/pgvector store adapter (DESIGN.md §5, D4/D7/D10/D11)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import psycopg
from psycopg import sql
from psycopg.rows import dict_row

from alembicio.adapters.base import (
    DocRecord,
    ModelSpec,
    ReconcileReport,
    StoreInfo,
    VectorRecord,
)
from alembicio.config import EmbMigrateConfig, StoreConfig

_VECTOR_INDEX_DIM_LIMIT = 2000
_SCHEMA = "alembicio"


@dataclass(frozen=True)
class PgVectorInspectInfo:
    """Extended inspect payload for pgvector-specific doctor checks."""

    dims: int
    row_count: int
    pending_count: int
    disk_estimate_bytes: int
    pgvector_version: str
    vectype: str
    indexable: bool
    text_available: bool
    projected_row_count: int


def model_slug(model: str) -> str:
    """Sanitize a model id into a safe SQL identifier suffix."""
    slug = re.sub(r"[^a-z0-9_]+", "_", model.lower()).strip("_")
    return slug[:48] or "target"


def select_vectype(dim: int) -> tuple[str, str]:
    """Return ``(pg_type, ops_class)`` for a target dimensionality."""
    if dim <= _VECTOR_INDEX_DIM_LIMIT:
        return "vector", "vector_cosine_ops"
    return "halfvec", "halfvec_cosine_ops"


def vector_literal(vector: npt.NDArray[np.float32]) -> str:
    """Format a float32 vector for pgvector input."""
    parts = ",".join(f"{float(x):.8g}" for x in vector.tolist())
    return f"[{parts}]"


class PgVectorAdapter:
    """Postgres/pgvector implementation of :class:`~alembicio.adapters.base.StoreAdapter`."""

    def __init__(
        self,
        *,
        conninfo: str,
        store: StoreConfig,
        target: ModelSpec,
        migration_id: str,
        mapping_mode: Literal["default", "mapping_only"] = "default",
    ) -> None:
        """Wire the adapter to a table and target model.

        Args:
            conninfo: Postgres DSN.
            store: Store section from embmigrate.yaml.
            target: Target model specification.
            migration_id: Active migration id for dead-letter exclusion.
            mapping_mode: Whether projected vectors count as complete.
        """
        self._conninfo = conninfo
        self._store = store
        self._target = target
        self._migration_id = migration_id
        self._mapping_mode = mapping_mode
        self._slug = model_slug(target["model"])
        self._emb_col = f"emb__{self._slug}"
        self._hash_col = f"emb__{self._slug}_hash"
        self._prov_col = f"emb__{self._slug}_provenance"
        self._vectype, self._ops = select_vectype(target["dim"])

    @classmethod
    def from_config(cls, config: EmbMigrateConfig) -> PgVectorAdapter:
        """Construct from a validated migration config."""
        return cls(
            conninfo=config.store.dsn,
            store=config.store,
            target={
                "provider": config.target.provider,
                "model": config.target.model,
                "dim": config.target.dim,
            },
            migration_id=config.migration,
            mapping_mode=config.mapping.mode,
        )

    def _connect(self) -> psycopg.Connection[Any]:
        return psycopg.connect(self._conninfo, row_factory=dict_row)

    def _table(self) -> sql.Identifier:
        return sql.Identifier("public", self._store.table)

    def _pending_sql(self) -> sql.Composed:
        """SQL fragment for the pending predicate (D11)."""
        hash_expr = sql.SQL(self._store.content_hash_expr or "md5('')")
        projected_pending = sql.SQL(
            "({prov} = 'projected' AND {mode})"
        ).format(
            prov=sql.Identifier(self._prov_col),
            mode=sql.Literal(self._mapping_mode != "mapping_only"),
        )
        hash_pending = sql.SQL("({hash_col} IS DISTINCT FROM ({hash_expr}))").format(
            hash_col=sql.Identifier(self._hash_col),
            hash_expr=hash_expr,
        )
        return sql.SQL("({hash_pending} OR {projected_pending})").format(
            hash_pending=hash_pending,
            projected_pending=projected_pending,
        )

    def inspect(self) -> StoreInfo:
        """Return store summary for doctor/status."""
        details = self.inspect_pg()
        return StoreInfo(
            dims=details.dims,
            row_count=details.row_count,
            pending_count=details.pending_count,
            disk_estimate_bytes=details.disk_estimate_bytes,
        )

    def _fetch_int(self, cur: psycopg.Cursor[Any], key: str = "n") -> int:
        row = cur.fetchone()
        if row is None:
            return 0
        return int(row[key])

    def inspect_pg(self) -> PgVectorInspectInfo:
        """Return pgvector-specific inspect details."""
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
            ext = cur.fetchone()
            pgvector_version = str(ext["extversion"]) if ext else "unknown"

            cur.execute(
                sql.SQL("SELECT COUNT(*) AS n FROM {}").format(self._table())
            )
            row_count = self._fetch_int(cur)

            pending = self.pending_count()

            cur.execute(
                sql.SQL("SELECT pg_total_relation_size({}::regclass) AS bytes").format(
                    sql.Literal(f"public.{self._store.table}")
                )
            )
            disk = self._fetch_int(cur, key="bytes")

            text_available = self._store.text_column is not None
            projected = self._projected_count(cur)

        indexable = self._target["dim"] <= _VECTOR_INDEX_DIM_LIMIT or pgvector_version >= "0.7"
        return PgVectorInspectInfo(
            dims=self._target["dim"],
            row_count=row_count,
            pending_count=pending,
            disk_estimate_bytes=disk,
            pgvector_version=pgvector_version,
            vectype=self._vectype,
            indexable=indexable,
            text_available=text_available,
            projected_row_count=projected,
        )

    def _projected_count(self, cur: psycopg.Cursor[Any]) -> int:
        cur.execute(
            sql.SQL(
                """
                SELECT COUNT(*) AS n FROM {}
                WHERE {prov} = 'projected'
                """
            ).format(self._table(), prov=sql.Identifier(self._prov_col)),
        )
        try:
            return self._fetch_int(cur)
        except psycopg.Error:
            return 0

    def prepare(self, target: ModelSpec) -> None:
        """Add target columns and install the dual-write trigger (idempotent)."""
        dim = target["dim"]
        vectype, _ = select_vectype(dim)
        emb_col = self._emb_col
        hash_col = self._hash_col
        prov_col = self._prov_col
        text_col = self._store.text_column or "content"
        fn_name = f"alembicio_dirty__{self._slug}"
        trig_name = f"alembicio_dirty__{self._slug}"

        ddl = sql.SQL(
            """
            CREATE EXTENSION IF NOT EXISTS vector;
            ALTER TABLE {table}
              ADD COLUMN IF NOT EXISTS {emb} {vectype}({dim}),
              ADD COLUMN IF NOT EXISTS {hash} text,
              ADD COLUMN IF NOT EXISTS {prov} text NOT NULL DEFAULT 'embedded';
            CREATE OR REPLACE FUNCTION {schema}.{fn}() RETURNS trigger AS $$
            BEGIN
              IF TG_OP = 'INSERT'
                 OR NEW.{text_col} IS DISTINCT FROM OLD.{text_col} THEN
                NEW.{emb} := NULL;
                NEW.{hash} := NULL;
                NEW.{prov} := 'embedded';
              END IF;
              RETURN NEW;
            END $$ LANGUAGE plpgsql;
            DROP TRIGGER IF EXISTS {trig} ON {table};
            CREATE TRIGGER {trig}
            BEFORE INSERT OR UPDATE ON {table}
            FOR EACH ROW EXECUTE FUNCTION {schema}.{fn}();
            """
        ).format(
            table=self._table(),
            emb=sql.Identifier(emb_col),
            vectype=sql.SQL(vectype),
            dim=sql.Literal(dim),
            hash=sql.Identifier(hash_col),
            prov=sql.Identifier(prov_col),
            schema=sql.Identifier(_SCHEMA),
            fn=sql.Identifier(fn_name),
            text_col=sql.Identifier(text_col),
            trig=sql.Identifier(trig_name),
        )
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(_SCHEMA)))
            cur.execute(ddl)
            conn.commit()

    def pending_count(self) -> int:
        """Count rows not backfill-complete per the pending predicate."""
        pending = self._pending_sql()
        query = sql.SQL(
            """
            SELECT COUNT(*) AS n FROM {table}
            WHERE {pending}
            """
        ).format(table=self._table(), pending=pending)
        with self._connect() as conn, conn.cursor() as cur:
            try:
                cur.execute(query)
                return self._fetch_int(cur)
            except psycopg.Error:
                return 0

    def claim_batch(self, *, limit: int) -> list[DocRecord]:
        """Claim pending rows with ``FOR UPDATE SKIP LOCKED``."""
        text_col = self._store.text_column
        id_col = self._store.id_column
        hash_expr = self._store.content_hash_expr or "md5('')"
        pending = self._pending_sql()

        query = sql.SQL(
            """
            SELECT {id_col} AS doc_id,
                   {text_col} AS text,
                   ({hash_expr}) AS content_hash
              FROM {table}
             WHERE {pending}
               AND NOT EXISTS (
                     SELECT 1 FROM {schema}.dead_letter dl
                      WHERE dl.migration_id = %s
                        AND dl.doc_id = {id_col}::text
                        AND dl.content_hash = ({hash_expr})::text
                   )
             ORDER BY {id_col}
             LIMIT {lim}
             FOR UPDATE SKIP LOCKED
            """
        ).format(
            id_col=sql.Identifier(id_col),
            text_col=sql.Identifier(text_col) if text_col else sql.SQL("NULL"),
            hash_expr=sql.SQL(hash_expr),
            table=self._table(),
            pending=pending,
            schema=sql.Identifier(_SCHEMA),
            lim=sql.Literal(limit),
        )

        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(query, (self._migration_id,))
            rows = cur.fetchall()
            conn.commit()
            return [
                DocRecord(
                    doc_id=str(row["doc_id"]),
                    text=row["text"],
                    content_ref=None,
                    content_hash=str(row["content_hash"]),
                )
                for row in rows
            ]

    def upsert_vectors(self, batch: list[VectorRecord]) -> None:
        """Idempotently write vectors guarded by the canonical content hash (I4)."""
        if not batch:
            return
        id_col = self._store.id_column
        hash_expr = self._store.content_hash_expr or "md5('')"
        vectype = self._vectype

        update = sql.SQL(
            """
            UPDATE {table}
               SET {emb} = %s::{vectype},
                   {hash} = %s,
                   {prov} = %s
             WHERE {id_col} = %s
               AND ({hash_expr}) = %s
            """
        ).format(
            table=self._table(),
            emb=sql.Identifier(self._emb_col),
            hash=sql.Identifier(self._hash_col),
            prov=sql.Identifier(self._prov_col),
            id_col=sql.Identifier(id_col),
            hash_expr=sql.SQL(hash_expr),
            vectype=sql.SQL(vectype),
        )

        with self._connect() as conn, conn.cursor() as cur:
            for record in batch:
                cur.execute(
                    update,
                    (
                        vector_literal(record["vector"]),
                        record["content_hash"],
                        record["provenance"],
                        record["doc_id"],
                        record["content_hash"],
                    ),
                )
            conn.commit()

    def reconcile(self) -> ReconcileReport:
        """Verify structural guarantees; pg deletes/updates are trigger-driven (I3/I4)."""
        dirty = 0
        with self._connect() as conn, conn.cursor() as cur:
            hash_expr = self._store.content_hash_expr or "md5('')"
            cur.execute(
                sql.SQL(
                    """
                    SELECT COUNT(*) AS n FROM {table}
                    WHERE {hash} IS DISTINCT FROM ({hash_expr})
                       OR ({prov} = 'projected' AND {mapping_only} = false)
                    """
                ).format(
                    table=self._table(),
                    hash=sql.Identifier(self._hash_col),
                    hash_expr=sql.SQL(hash_expr),
                    prov=sql.Identifier(self._prov_col),
                    mapping_only=sql.Literal(self._mapping_mode == "mapping_only"),
                ),
            )
            dirty = self._fetch_int(cur)
        return ReconcileReport(
            tombstones_applied=0,
            dirty_requeued=dirty,
            orphans_removed=0,
        )

    def build_index(self, *, concurrently: bool = True) -> None:
        """Build an HNSW index on the target embedding column."""
        idx_name = f"idx_{self._store.table}_{self._slug}"
        concurrently_sql = sql.SQL("CONCURRENTLY") if concurrently else sql.SQL("")
        stmt = sql.SQL(
            """
            CREATE INDEX {concurrently} IF NOT EXISTS {idx}
            ON {table} USING hnsw ({emb} {ops})
            """
        ).format(
            concurrently=concurrently_sql,
            idx=sql.Identifier(idx_name),
            table=self._table(),
            emb=sql.Identifier(self._emb_col),
            ops=sql.SQL(self._ops),
        )
        conn = psycopg.connect(self._conninfo, autocommit=True)
        try:
            with conn.cursor() as cur:
                cur.execute(stmt)
        finally:
            conn.close()

    def search(
        self,
        vector: npt.NDArray[np.float32],
        *,
        space: Literal["old", "new"],
        k: int,
    ) -> list[str]:
        """Search old or new space by cosine distance."""
        id_col = self._store.id_column
        if space == "new":
            emb_col = self._emb_col
            vectype = self._vectype
        else:
            if not self._store.old_embedding_column:
                msg = "old_embedding_column is not configured"
                raise ValueError(msg)
            emb_col = self._store.old_embedding_column
            vectype = "vector"

        query = sql.SQL(
            """
            SELECT {id_col}::text AS doc_id
              FROM {table}
             WHERE {emb} IS NOT NULL
             ORDER BY {emb} <=> %s::{vectype}
             LIMIT %s
            """
        ).format(
            id_col=sql.Identifier(id_col),
            table=self._table(),
            emb=sql.Identifier(emb_col),
            vectype=sql.SQL(vectype),
        )
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(query, (vector_literal(vector), k))
            return [str(row["doc_id"]) for row in cur.fetchall()]

    def flip_read_path(
        self,
        *,
        active: Literal["old", "new"],
        canary_pct: int = 0,
        seed: int = 0,
    ) -> None:
        """Atomically swap the search view and update read_state (D7)."""
        columns = self._list_columns()
        if not columns:
            msg = f"table {self._store.table} has no columns"
            raise ValueError(msg)

        old_emb = self._store.old_embedding_column or "embedding"
        emb_source = self._emb_col if active == "new" else old_emb
        select_parts: list[sql.Composed | sql.Identifier] = []
        for col in columns:
            if col == old_emb:
                select_parts.append(
                    sql.SQL("{emb} AS embedding").format(
                        emb=sql.Identifier(emb_source if active == "new" else old_emb)
                    )
                )
            elif col == self._emb_col and col != old_emb:
                continue
            else:
                select_parts.append(sql.Identifier(col))
        select_list = sql.SQL(", ").join(select_parts)
        view = sql.Identifier("public", f"{self._store.table}_search")

        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                sql.SQL("CREATE OR REPLACE VIEW {view} AS SELECT {cols} FROM {table}").format(
                    view=view,
                    cols=select_list,
                    table=self._table(),
                )
            )
            cur.execute(
                sql.SQL(
                    """
                    INSERT INTO {schema}.read_state (migration_id, active, canary_pct, seed)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (migration_id) DO UPDATE
                      SET active = EXCLUDED.active,
                          canary_pct = EXCLUDED.canary_pct,
                          seed = EXCLUDED.seed
                    """
                ).format(schema=sql.Identifier(_SCHEMA)),
                (self._migration_id, active, canary_pct, seed),
            )
            conn.commit()

    def _list_columns(self) -> list[str]:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT a.attname AS column_name
                  FROM pg_attribute a
                  JOIN pg_class c ON a.attrelid = c.oid
                  JOIN pg_namespace n ON c.relnamespace = n.oid
                 WHERE c.relname = %s
                   AND n.nspname = 'public'
                   AND a.attnum > 0
                   AND NOT a.attisdropped
                 ORDER BY a.attnum
                """,
                (self._store.table,),
            )
            return [str(row["column_name"]) for row in cur.fetchall()]

    def decommission(self) -> None:
        """Drop migration artifacts from the user table."""
        slug = self._slug
        fn_name = f"alembicio_dirty__{slug}"
        trig_name = f"alembicio_dirty__{slug}"
        view = sql.Identifier("public", f"{self._store.table}_search")

        stmt = sql.SQL(
            """
            DROP TRIGGER IF EXISTS {trig} ON {table};
            DROP FUNCTION IF EXISTS {schema}.{fn}();
            DROP VIEW IF EXISTS {view};
            ALTER TABLE {table}
              DROP COLUMN IF EXISTS {emb},
              DROP COLUMN IF EXISTS {hash},
              DROP COLUMN IF EXISTS {prov};
            DELETE FROM {schema}.read_state WHERE migration_id = %s;
            """
        ).format(
            trig=sql.Identifier(trig_name),
            table=self._table(),
            schema=sql.Identifier(_SCHEMA),
            fn=sql.Identifier(fn_name),
            view=view,
            emb=sql.Identifier(self._emb_col),
            hash=sql.Identifier(self._hash_col),
            prov=sql.Identifier(self._prov_col),
        )
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(stmt, (self._migration_id,))
            conn.commit()
