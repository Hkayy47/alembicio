# DESIGN.md — alembicio architecture (Phase 0 output)

Status: v0.1 design, ratified before first line of core code. Companion documents:
INVARIANTS.md (law), docs/research-report.md (problem + landscape), PROMPTS.md (how each
phase gets built). Diagram: docs/state-machine.svg.

## 1. Purpose and MVP scope

alembicio migrates a production vector store from one embedding model to another with
zero downtime and verified retrieval quality: declarative migration file → dual-write →
checkpointed, budget-aware, crash-resumable backfill → golden-query verification →
staged cutover → soak → rollback or decommission. It runs, verifies, and leaves; it does
not become a resident service.

**In scope for v0.1 (two-week MVP):** pgvector adapter; providers `openai`, `gemini`,
`fastembed` (local, keyless demo); preserve-chunks mode; Procrustes + ridge mapping;
recall/MRR/overlap report; cutover/rollback/decommission; `doctor`.
**v0.2:** Qdrant adapter (dual collection + alias flip + write shim + reconciliation),
mapping recovery tables published in docs.
**Explicit non-goals for now:** re-chunk mode (re-ingestion with ID remapping), serving
proxies, scheduling daemons, >2 backends, MLP adapters, Batch-API submit/poll mode
(config key reserved, implementation deferred).

## 2. State machine

```
CREATED → PREPARED → BACKFILLING ⟲ → BACKFILLED → VERIFIED → CANARY → CUTOVER → SOAKING → DONE
                 │                       │            │          │         │
                 │                       │            └──────────┴─────────┴──→ ROLLED_BACK
                 └── PAUSED_BUDGET / PAUSED_ERROR (resumable) ──→ BACKFILLING
```

| From | Verb | To | Guard |
|---|---|---|---|
| (none) | `init` | CREATED | config parses; no existing migration with same name |
| CREATED | `doctor` | CREATED | read-only; records estimate snapshot |
| CREATED | `prepare` | PREPARED | doctor checks pass (I10); idempotent re-run is a no-op |
| PREPARED / PAUSED_* | `backfill` | BACKFILLING | budget remaining > 0 |
| BACKFILLING | (worker) | BACKFILLED | `reconcile()` then `pending_count = 0` re-checked atomically with state write; target index build completed (or `verify.allow_unindexed: true`); zero rows with target `provenance = 'projected'` unless `mapping.mode: mapping_only` |
| BACKFILLING | (budget hit) | PAUSED_BUDGET | clean batch boundary (I5); no partial-batch ledger commit |
| BACKFILLED | `verify` | VERIFIED | gates in I9 pass; report written; `projected_row_count` recorded |
| VERIFIED | `cutover --canary N` | CANARY | N in [1,99]; gates still satisfied; `projected_row_count = 0` unless `mapping_only`; index ready |
| VERIFIED / CANARY | `cutover` | CUTOVER | gates still satisfied (re-run verify or fast-path recheck); index ready |
| CANARY / CUTOVER | (clock) | SOAKING | soak window begins at 100% |
| CANARY / CUTOVER / SOAKING | `rollback` | ROLLED_BACK | always legal; flips read path back |
| SOAKING | `decommission` | DONE | soak elapsed; typed confirmation |

Every verb re-entrant per I6. `status` renders this table's current row plus progress,
spend, dead-letter counts, and index build phase.

## 3. Control plane

**Where migration state lives (Decision D1):** when the target store is Postgres, the
control plane is a `alembicio` schema *in the same database*, so progress markers and
vector writes can share transactions — the strongest possible exactly-once story. For
non-transactional stores (Qdrant), the control plane is a local SQLite file under
`--state-dir` (checked into nothing, backed up by the operator), and exactly-once is
achieved via idempotent upserts + reconciliation instead of shared transactions.

Tables (Postgres dialect; SQLite mirrors):

```sql
CREATE SCHEMA IF NOT EXISTS alembicio;
CREATE TABLE alembicio.migration (
  id           text PRIMARY KEY,          -- from yaml `migration:`
  config_hash  text NOT NULL,             -- sha256 of canonicalized yaml
  state        text NOT NULL,             -- enum per §2
  created_at   timestamptz NOT NULL DEFAULT now(),
  updated_at   timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE alembicio.ledger (
  migration_id text REFERENCES alembicio.migration(id),
  tokens_in    bigint NOT NULL DEFAULT 0,  -- monotone (I5)
  usd_est      numeric NOT NULL DEFAULT 0, -- monotone (I5)
  updated_at   timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (migration_id)
);
CREATE TABLE alembicio.dead_letter (
  migration_id text NOT NULL,
  doc_id       text NOT NULL,
  content_hash text NOT NULL,
  reason       text NOT NULL,              -- token_limit | empty | provider_4xx | ...
  attempts     int  NOT NULL,
  last_error   text,
  ts           timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (migration_id, doc_id, content_hash)
);
CREATE TABLE alembicio.read_state (
  migration_id text PRIMARY KEY,
  active       text NOT NULL,              -- 'old' | 'new'
  canary_pct   int  NOT NULL DEFAULT 0,
  seed         bigint NOT NULL
);
CREATE TABLE alembicio.checkpoint (        -- used by offset-based backends (Qdrant)
  migration_id text NOT NULL,
  shard_id     text NOT NULL,
  last_offset  text NOT NULL,
  updated_at   timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (migration_id, shard_id)
);
```

## 4. Core protocols

```python
class DocRecord(TypedDict):
    doc_id: str
    text: str | None          # None when only content_ref is available
    content_ref: str | None
    content_hash: str

class VectorRecord(TypedDict):
    doc_id: str
    content_hash: str
    vector: "np.ndarray"       # float32, shape (d,)
    provenance: Literal["embedded", "projected"]

class StoreAdapter(Protocol):
    def inspect(self) -> StoreInfo: ...                     # dims, counts, text availability, disk estimate
    def prepare(self, target: ModelSpec) -> None: ...       # add column/collection, install capture (trigger/shim)
    def pending_count(self) -> int: ...                     # rows not backfill-complete per §5 pending predicate
    def claim_batch(self, *, limit: int) -> list[DocRecord]: ...   # pg: SKIP LOCKED; qdrant: scroll from checkpoint
    def upsert_vectors(self, batch: list[VectorRecord]) -> None: ...  # idempotent (I1)
    def reconcile(self) -> ReconcileReport: ...             # deletes-win sweep (I3), dirty re-enqueue (I4)
    def build_index(self, *, concurrently: bool = True) -> None: ...
    def search(self, vector: "np.ndarray", *, space: Literal["old", "new"], k: int) -> list[str]: ...
    def flip_read_path(self, *, active: Literal["old", "new"], canary_pct: int = 0) -> None: ...
    def decommission(self) -> None: ...
```

The worker composes these: `claim_batch → budget.reserve (in-memory only) →
provider.embed_batch → store.upsert_vectors → ledger.commit`, with token-bucket pacing,
exponential backoff, a circuit breaker per provider, and dead-lettering per I-series
rules. `ledger.commit` runs only after `upsert_vectors` durably satisfies or
dead-letters every item in the batch; a reserved batch that exhausts budget mid-embed is
rolled back (no ledger commit, no checkpoint advance). Transition to `BACKFILLED` runs
`reconcile()`, re-checks `pending_count = 0`, and writes state in one critical section.
The worker owns no backend- or provider-specific logic.

## 5. pgvector adapter (v0.1 flagship)

**Why pgvector first (Decision D4):** triggers give change-capture for free, and the
dual-*column* pattern makes deletes/updates structurally consistent — the new vector
lives in the same row as the document, so a deleted row deletes both spaces atomically
and I3 is satisfied by construction.

**Schema per migration** (slug = sanitized target model id):

```sql
ALTER TABLE {table}
  ADD COLUMN IF NOT EXISTS emb__{slug} {vectype}({dims}),
  ADD COLUMN IF NOT EXISTS emb__{slug}_hash text,
  ADD COLUMN IF NOT EXISTS emb__{slug}_provenance text NOT NULL DEFAULT 'embedded';
```

**Canonical hash:** yaml field `content_hash_expr` (SQL fragment, default
`md5({text_column}::text)`) is the only hash definition. `doctor` compiles it once;
the adapter uses the same fragment in pending, claim, upsert guards, and companion-hash
writes (Decision D2, D11).

**Companion hash is the idempotency record (Decision D10):** a row is *done* iff
`emb__{slug}_hash = ({content_hash_expr})` AND (`emb__{slug}_provenance = 'embedded'`
OR `mapping.mode = mapping_only`). No separate done-set; the record is transactional
with the row.

**Resume is predicate-driven (Decision D11):** pending ⇔
`(emb__{slug}_hash IS DISTINCT FROM ({content_hash_expr}))
 OR (emb__{slug}_provenance = 'projected' AND mapping.mode != mapping_only))`.
Workers claim with `SELECT ... FOR UPDATE SKIP LOCKED` on the pending predicate, so
parallel workers and crash-resume need no shard bookkeeping on Postgres. (Qdrant uses
`alembicio.checkpoint` scroll offsets instead; see §6 for advance ordering.)

**Dual-write capture (trigger):**

```sql
CREATE OR REPLACE FUNCTION alembicio_dirty__{slug}() RETURNS trigger AS $$
BEGIN
  IF TG_OP = 'INSERT'
     OR NEW.{text_col} IS DISTINCT FROM OLD.{text_col} THEN
    NEW.emb__{slug} := NULL;
    NEW.emb__{slug}_hash := NULL;
    NEW.emb__{slug}_provenance := 'embedded';
  END IF;
  RETURN NEW;
END $$ LANGUAGE plpgsql;

CREATE TRIGGER alembicio_dirty__{slug}
BEFORE INSERT OR UPDATE ON {table}
FOR EACH ROW EXECUTE FUNCTION alembicio_dirty__{slug}();
```

**Dimension/index constraint (surfaced by `doctor`):** pgvector's HNSW/IVFFlat index on
the `vector` type supports at most 2000 dimensions. For 3072-dim targets
(text-embedding-3-large, gemini-embedding-001 default) the adapter uses
`halfvec(3072)` with `halfvec_cosine_ops` (pgvector ≥ 0.7), or the user opts into MRL
truncation (e.g., `dims: 1536`) in the yaml. `doctor` refuses configs that would build
an unindexable column. `{vectype}` above is `vector` or `halfvec` accordingly.

**Index build:** after backfill reaches 100% (or a configured threshold),
`CREATE INDEX CONCURRENTLY ... USING hnsw (emb__{slug} {ops})` with a documented
`maintenance_work_mem` recommendation printed by `status`.

**Index readiness gate:** `build_index(concurrently=True)` MUST complete (or fail
loudly) before `verify` or `cutover` unless `verify.allow_unindexed: true`. `status`
reports index build phase (`BUILDING`, `VALID`, `INVALID`) separately from backfill
percent.

**Cutover (Decision D7):** the atomic primitive is a view swap in one transaction
using an **explicit column list** (never `SELECT *`) — e.g.
`CREATE OR REPLACE VIEW {table}_search AS
 SELECT {id_col}, {text_col}, …, {old_emb_col} AS embedding FROM {table}` flipped to
`… emb__{slug} AS embedding …`. Request-level canary requires app cooperation:
alembicio writes `alembicio.read_state` and ships a ten-line helper
(`alembicio.runtime.choose_space(request_key)`) that hashes a request key against
`canary_pct`/`seed`. README states that 0/100 flips need only the view, while
percentage canary needs the helper; apps reading the base table directly are unsupported
during migration. Rollback = the same view swap in reverse; dual-write trigger stays
installed until decommission, so rollback is lossless.

**Decommission:** drop trigger, function, companion columns (including provenance), and
`read_state` row; `VACUUM` advice printed.

## 6. Qdrant adapter (v0.2, designed now so the protocol is honest)

Dual collection configured for the target model; **alias flip** as the native cutover
primitive; dual-write via a thin SDK shim (a wrapper over `qdrant_client` the app
imports during the migration window) because Qdrant has no server-side triggers;
tombstone table in the SQLite control plane for I3; a post-backfill **reconciliation
scroll** compares point versions/payloads and repairs drift the shim missed.
Named-vectors mode (Qdrant ≥ 1.18) is used when the existing collection already uses
named vectors; otherwise dual collection — the two blessed vendor patterns from
Qdrant's migration tutorial; the adapter automates the choice.

**Checkpoint ordering (I2, D11):** `alembicio.checkpoint.last_offset` advances only in
the same SQLite transaction as `ledger.commit`, and both run only after
`upsert_vectors` returns success for every non-dead-lettered point in the batch (Qdrant
upsert acknowledged). Crash before that transaction leaves offset unchanged.

**Dual-write vs vendor tutorial (D7):** Qdrant's manual tutorial disables dual-write
after alias flip. Alembicio **keeps** dual-write through soak so rollback remains
lossless; README must say so explicitly. Decommission drops the shim.

**Tombstones (I3):** delete paths write `(doc_id, deleted_at)` to SQLite before or with
Qdrant delete; `upsert_vectors` MUST skip ids with tombstones newer than the batch claim
timestamp; reconciliation removes orphan points in the new collection/vector name.

## 7. Providers and budgeter

`openai`, `gemini`, `fastembed` behind `EmbeddingProvider`. Token counting: provider
tokenizer when available, else a conservative chars/3.5 estimate flagged as estimate in
the ledger. Budgeter = token bucket (capacity `tpm`, refill `tpm/60` per second; second
bucket for `rpm`) + in-memory pre-reservation against remaining budget (I5); ledger
commit follows durable upsert per §4. Retries: exponential backoff with jitter on
429/5xx, circuit breaker opens after N consecutive failures and pauses the run as
`PAUSED_ERROR` (resumable). Per-model price table lives in one constants module with
yaml override `pricing.usd_per_mtok`.

## 8. Verification (eval/)

`golden.jsonl` lines: `{"q": str, "expect": [doc_id, ...], "k": int}`. Runner embeds
each query with old and new models, searches the respective spaces via
`StoreAdapter.search`, computes: known-item recall@k (the gate, I9), MRR, overlap@k and
rank correlation between old/new lists (context, not a gate), and p50/p95 search
latency. Output: `report.md` with a machine-readable `report.json` twin including
`projected_row_count` (I8). Synthetic golden sets (LLM-generated known-item pairs) are
supported behind `alembicio golden synth` and are labeled synthetic in the report per
I9.

## 9. Mapping module (mapping/, NumPy-only per I12)

Anchors: default 4096 rows sampled stratified by text-length decile; embed with both
models → `A (n×d_s)`, `B (n×d_t)`; 80/20 fit/held-out split. **Procrustes:** `M = AᵀB`,
`U, Σ, Vᵀ = svd(M)`, `Q = U Vᵀ` (rectangular allowed — handles 768→3072). **Ridge:**
`W = (AᵀA + λI)⁻¹ AᵀB`, λ selected on held-out. Recovery metric = held-out known-item
recall ratio vs. true target embeddings; gate per I8. Mapping backfill writes
`provenance = "projected"`; such rows remain pending for true re-embed per §5 pending
predicate unless `mapping.mode: mapping_only`. Artifacts saved as `.npz` with metadata
`{source_model, target_model, dims, n_anchors, recovery, created_at}`. Unit anchor
test: a synthetic random rotation of Gaussian data must be recovered to `np.allclose`
precision before the module is trusted on real pairs.

## 10. Failure taxonomy → handling

| Failure | Handling | Invariant |
|---|---|---|
| Crash mid-batch (kill -9, OOM) | resume from predicate/checkpoint; replays are no-ops | I1, I2 |
| Provider 429 / 5xx storm | backoff + circuit breaker → PAUSED_ERROR, resumable | I5, I6 |
| Doc updated during backfill | trigger nulls companion hash; stale write can't satisfy | I4 |
| Doc deleted during backfill | pg: structural; qdrant: tombstone + reconcile | I3 |
| Poison doc (over limit, empty, 4xx) | dead_letter with reason; run continues | §3 table |
| Budget exceeded | clean stop at batch boundary → PAUSED_BUDGET; no partial ledger commit | I5 |
| Dims > index limit | doctor refuses; halfvec or MRL guidance | I10 |
| Projected rows treated as done | pending predicate includes provenance; verify reports count | I8 |
| Insert between reconcile and BACKFILLED | atomic reconcile + pending re-check + state write | I6 |
| Index not ready at verify/cutover | gate unless `verify.allow_unindexed`; status shows phase | I9, I10 |
| Verification gate fails | cutover refused; report explains | I9 |
| Regression discovered post-cutover | rollback = read-path flip; dual-write kept through soak | I6, I7 |

## 11. Decisions log

- **D1** Control plane co-located in Postgres when target is Postgres; SQLite otherwise.
- **D2** Idempotency key = `(doc_id, content_hash, target_model_id)`; `content_hash` comes from a single yaml `content_hash_expr` used everywhere.
- **D3** Preserve-chunks mode only in MVP; re-chunk mode is v2 with explicit ID-remap design.
- **D4** pgvector first: triggers = free CDC; dual-column dissolves delete/update races.
- **D5** Mapping + metrics are NumPy-only; core stays torch/scipy/sklearn-free.
- **D6** Two backends max in year one; third adapters arrive via frozen Protocol + shared scenario matrix.
- **D7** Cutover primitive is an atomic flip (explicit-column view swap / alias); percentage canary is app-assisted via `read_state` helper — documented honestly, not faked; dual-write kept through soak (deliberate extension of Qdrant tutorial for lossless rollback).
- **D8** Budget ledger persists monotonically with progress; in-memory pre-reservation before embed, ledger commit only after durable upsert.
- **D9** Batch-API (submit/poll/merge) reserved in yaml, deferred past MVP.
- **D10** Companion hash column *is* the done-record on pgvector (transactional with the row); completion also requires `provenance = 'embedded'` unless `mapping_only`.
- **D11** Postgres resume is predicate-driven (`content_hash_expr` + provenance + SKIP LOCKED); Qdrant offset checkpoints advance only post-upsert in the same SQLite txn as ledger commit.
- **D12** Demo path is keyless: fastembed MiniLM → bge-small on docker Postgres; CI never needs paid keys.

## 12. Open questions (tracked, not blocking)

Batch-API orchestration shape; re-chunk ID-remap design; Weaviate vs. Milvus as adapter
three (demand-driven); multi-tenant/namespace scoping; whether `golden synth` should
ship in core or `[demo]`.
