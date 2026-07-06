# INVARIANTS.md — the law of alembicio

Every agent session (Cursor, Claude Code, claude.ai) and every human PR is bound by this
file. If code and this file disagree, the code is wrong. Changes to this file require a
dedicated PR that touches nothing else.

## I1. Exactly-once effects

Delivery is at-least-once (retries, crashes, and resumes happen). *Effects* are
exactly-once. The idempotency key is `(doc_id, content_hash, target_model_id)`,
where `content_hash` is the single canonical expression configured in yaml
(`content_hash_expr`) and computed identically by `doctor`, the adapter pending
predicate, claim_batch, and upsert_vectors. Re-processing an already-succeeded key
MUST be a no-op. All vector writes MUST be idempotent upserts keyed by the idempotency
key. Replay safety is defined over store effects (vector upserts, companion-hash writes,
ledger increments for keys already durably recorded) and dead-letter keys; provider HTTP
retries may incur duplicate API spend and are bounded by budget counters, not by
exactly-once effects. It is never acceptable to double-charge the budget ledger for
replayed work that was already recorded.

## I2. Durability before progress

A checkpoint (or, on pgvector, a companion-hash write — see D10/D11 in DESIGN.md) may
only be committed after every write it covers is durable in the target store. Resume
derives work from durable state only. Progress markers are monotonic; they never move
backward.

## I3. Deletes win (tombstone precedence)

A delete observed at logical time *t* supersedes any backfill write for that document,
regardless of arrival order. On dual-column backends (pgvector) this is structural — the
vector lives in the deleted row. On dual-collection backends (Qdrant), deletes write
tombstones, backfill MUST check tombstones before upsert, and reconciliation MUST remove
any vector whose source document no longer exists.

## I4. Dirty wins (update precedence)

An update during backfill changes `content_hash` and marks the row dirty. A backfill
write computed from stale content MUST NOT satisfy the dirty row: satisfaction is
defined as "stored companion hash equals the canonical content_hash (`content_hash_expr`
evaluated on the current row)," never as "a write happened."

## I5. Budget monotonicity and hard stops

Token and cost counters only increase, persist with progress state, and survive crashes.
When a configured budget is exceeded, the worker stops cleanly at a batch boundary,
transitions to `PAUSED_BUDGET`, and remains fully resumable. Budgets are enforced
*before* spending, using the provider token estimate for the batch. A batch whose embed
step finishes after the budget ceiling is reached MUST NOT commit ledger progress; the
entire batch is retried or dead-lettered on resume.

## I6. State-machine legality

Only the transitions listed in DESIGN.md §2 are legal. Every CLI verb is idempotent:
running `prepare` twice is a no-op; `backfill --resume` is safe from any crash point;
`verify` never mutates vector state. `cutover` refuses unless verification gates passed
(override requires `--force` plus typed confirmation). `rollback` is legal from
`CANARY`, `CUTOVER`, and `SOAKING`. `decommission` is legal only after the soak window
and requires typing the migration id.

## I7. Read-path atomicity (adapter scope)

`flip_read_path` and `StoreAdapter.search(..., space=…)` expose exactly one active
space per call. The pgvector search view MUST enumerate columns explicitly (never
`SELECT *`) so exactly one embedding column is visible to readers. Integration tests
MUST assert: (a) concurrent readers during view swap never observe duplicate or mixed
embedding columns in one result row, and (b) `search(space="old")` and
`search(space="new")` read only their respective storage. Request-level canary splits
traffic by request via `read_state` and `choose_space()`; apps that bypass the
view/helper are out of scope and MUST be documented as such in README.

## I8. Mapping mode is honest

Projected vectors carry `provenance = "projected"`. Rows whose target-space provenance
is `"projected"` are never treated as backfill-complete unless `mapping.mode:
mapping_only` is set in config. `pending_count`, `claim_batch`, and the pgvector pending
predicate MUST treat such rows as pending. True re-embeddings MUST overwrite projected
vectors during backfill. A finished migration (`state = DONE`, or `BACKFILLED`/`VERIFIED`/
`CUTOVER`/`SOAKING` when `mapping.mode: mapping_only`) contains zero rows where target
provenance is `"projected"`. The verify report MUST include `projected_row_count`;
`cutover` refuses when `projected_row_count > 0` unless `mapping_only`. Cutover in
mapping mode refuses if held-out recovery is below `mapping.min_recovery`.

## I9. Verification gates

`cutover` requires: known-item `recall@k(new) >= gates.min_recall_ratio ×
recall@k(old)` on the golden set, a generated report artifact, and (unless
`mapping.mode: mapping_only`) `projected_row_count = 0`. Synthetic golden sets are
labeled as synthetic in the report.

## I10. Data preconditions

A migration refuses to start unless every row exposes retrievable source text
(`text_column` or `content_ref`) and a computable `content_hash` via the configured
`content_hash_expr` (default: `md5({text_column}::text)`). The same expression MUST be
used for the companion-hash column, pending predicate, claim predicate, and upsert guard
— `doctor` verifies this single definition, along with provider auth, target dims vs.
store schema (including the pgvector 2000-dim index limit — see DESIGN.md §5), and
transient disk headroom.

## I11. Safety defaults

Destructive verbs are dry-run by default. Secrets appear only via `env:` indirection in
YAML — never literals, never logged. Embedding vectors and fitted mapping matrices are
treated as sensitive data (see SECURITY.md).

## I12. Engineering conventions (binding on agents and humans)

- Never rename or remove a released public symbol; never change a released function
  signature in a breaking way. New parameters are keyword-only with sensible defaults so
  existing call sites require zero changes.
- When an agent modifies a file, it outputs the entire file — no diffs-as-prose, no
  elided sections.
- No placeholder comments, no `TODO`, no `...` stubs in committed code. Every function
  is complete and runnable.
- Type annotations on every function signature. Docstrings (one-line summary + Args /
  Returns) on every public method.
- `mapping/` and `eval/metrics` are NumPy-only. No scipy, sklearn, or torch anywhere in
  the core package. Heavy demo dependencies (fastembed, docker fixtures) live in
  `[demo]` / `[dev]` extras.

## I13. Testing law

No merge into `core/` (state, worker, checkpoint, reconcile, cutover) without: (a)
property tests covering the change, (b) the crash-injection suite green, and (c) an
adversarial-review summary pasted into the PR description (prompt P8). Weakening,
skipping, or deleting a failing test to get green is prohibited; fix the code or fix the
spec via a dedicated INVARIANTS/DESIGN PR.
