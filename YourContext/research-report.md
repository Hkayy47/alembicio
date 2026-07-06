# Alembicio — Research Report & Build Plan

**"Alembic for vectors": a DB-agnostic, zero-downtime embedding model migration orchestrator.**
Prepared July 5, 2026. All landscape claims verified against live sources (linked in §8).

---

## 1. Executive summary

Every embedding model defines its own private coordinate system. A vector database is therefore silently welded to one model version, and when that model is deprecated — which vendors now do on a roughly annual cadence — search breaks while every dashboard stays green. Google shut down `embedding-001` (Aug 14, 2025) and `text-embedding-004` (Jan 14, 2026); Azure deprecated `text-embedding-ada-002` (Jan 4, 2025); Google's own docs state that `gemini-embedding-001` and `gemini-embedding-2` occupy incompatible spaces and that upgrading requires re-embedding everything.

The response so far: every vendor publishes a *manual tutorial* (Qdrant's blue-green walkthrough, Google Cloud's dual-column pattern), one Postgres-only tool automates re-embedding without migration semantics (pgai Vectorizer), one startup sells learned projections between spaces (Schift), one EMNLP 2025 paper formalizes the adapter approach (Drift-Adapter), and teams keep spec'ing the orchestration in-house (e.g., OpenViking issue #1523, April 2026, which reads like alembicio's feature list written by a stranger).

Nobody has shipped the obvious open-source thing: **declarative migration files → dual-write → checkpointed, resumable, budget-aware backfill → golden-query recall verification → staged cutover → rollback**, working across more than one database. That is alembicio. The projection math (orthogonal Procrustes in pure NumPy) is the CS+math flex, offered honestly as a stopgap mode with published recovery numbers; the exactly-once orchestration is the systems flex and the actual moat.

Verdict unchanged from your notes: difficulty ~3.5/5, traction ~3.5/5, and the reason it's absent from project-idea lists — boring ops pain — is exactly why it's real.

---

## 2. The problem from first principles

### 2.1 What an embedding actually is

An embedding model is a function `f_θ : text → R^d`. Retrieval works because training (typically a contrastive objective like InfoNCE) arranges the outputs so that semantically related texts have high inner product / cosine similarity and unrelated texts don't. A vector database then builds an approximate-nearest-neighbor (ANN) index — HNSW, IVF, etc. — over the corpus vectors so that "find the most similar" runs in milliseconds instead of a linear scan.

The crucial subtlety: the training loss depends only on **relative** geometry — inner products among the model's *own* outputs. Nothing in the objective pins down absolute coordinates.

### 2.2 Why two models' vectors are mutually meaningless

**Rotation invariance.** For any orthogonal matrix `Q ∈ O(d)` (a rotation/reflection), `⟨Qx, Qy⟩ = ⟨x, y⟩` for all x, y. So if `f` achieves some training loss, the model `g = Q ∘ f` achieves *exactly the same loss*. The coordinate system is unidentified: even a perfect retraining of the *same* architecture on the *same* data lands in an arbitrarily rotated copy of the space, before you account for different random init, data order, tokenizer, architecture, or objective — all of which change the intrinsic geometry too, not just its orientation.

**What cross-model cosine actually computes.** Query `q` embedded by model B, compared against corpus vectors from model A, is a dot product between coordinates expressed in unrelated bases. For high-dimensional vectors with effectively independent coordinates, cosine similarity concentrates near 0 with spread on the order of `1/√d` — at d = 1536 or 3072, that's noise around zero. Ranking by noise returns essentially random neighbors, hence the 0.000-recall benchmark you observed. Monitoring doesn't catch it because every layer is "healthy": the API returns 200s, the index answers queries, latency is fine. Only *relevance* died.

**Dimensionality is a coincidence, not a contract.** Two models emitting 3072-dim vectors share a shape, not a meaning — axis 517 in one space has no relationship to axis 517 in the other. The canonical proof is now first-party: Google's embedding docs state outright that gemini-embedding-001 and gemini-embedding-2 spaces are incompatible, that their outputs cannot be directly compared, and that upgrading requires re-embedding all existing data — same vendor, same default 3072 dims.

**Matryoshka (MRL) is orthogonal to this problem.** MRL-trained models (gemini-embedding-001 and -2 among them) let you truncate one model's vector to a useful prefix (3072 → 1536 → 768). That's dimension flexibility *within* a single model's space. It does nothing for cross-model compatibility, though it matters operationally (choosing a target dimensionality during migration).

**But the spaces aren't unrelated in structure.** The Platonic Representation Hypothesis line of work, and vec2vec concretely, show that different encoders converge to similar *latent semantic structure*: vec2vec translates between spaces with no paired data at all, reaching cosine similarity up to 0.92 against ground-truth targets. Two consequences for alembicio: (a) *supervised* linear maps (you have unlimited paired data — just embed the same texts twice) work well enough to be a useful bridge; (b) embeddings leak content — vec2vec demonstrates attribute inference and inversion from vectors alone — so treat vector stores and any mapping artifacts as sensitive data, and say so in the docs.

### 2.3 Why this is an operations crisis, not an inconvenience

**Deprecations are the forcing function.** Verified timeline of embedding-model retirements and forced moves:

| Date | Event | Consequence |
|---|---|---|
| Jan 4, 2024 | OpenAI shuts down all first-generation embedding models | First mass forced migration (to ada-002) |
| Jan 2024 | OpenAI ships text-embedding-3-small/large | ada-002 becomes legacy; 3-small is ~5× cheaper and far better on multilingual (54.9 vs 31.4 MIRACL) |
| Jan 4, 2025 | Azure deprecates text-embedding-ada-002 (retirement follows) | The most widely deployed embedding model in the world enters end-of-life |
| Jul 2025 | gemini-embedding-001 GA (3072-dim, MRL, $0.15/M tokens) | Replacement target announced with deprecation notice attached |
| Aug 14, 2025 | Google shuts down embedding-001 (and gemini-embedding-exp-03-07) | Forced move #1 in the Gemini ecosystem |
| Jan 14, 2026 | Google shuts down text-embedding-004 (Gemini API **and** Vertex) | 768 → 3072 dim change; real projects break with "model not found" and must delete/recreate indices (documented in the wild, e.g. anthropics/claude-code issue #23557) |
| 2026 | gemini-embedding-2 ships (multimodal) | Google docs: space incompatible with gemini-embedding-001; re-embed everything |

Every RAG system older than ~18 months has either done this migration or has it queued. "At least once per team" is conservative.

**The silent-failure mode.** The nastiest version isn't the hard 404 — it's the *partial* state: config updated to the new model for queries while the corpus still holds old vectors (or half of each, if someone re-embeds in place). Search "works," quality is garbage, and there is no error to alert on. The OpenViking maintainers describe exactly this: in-place reindex causes user-visible degradation during the window, with queries hitting a half-old-half-new vector set.

### 2.4 The cost model of a migration (why naive approaches break)

Worked example: 1M documents, ~800 tokens each → 800M tokens to re-embed.

- **API cost:** text-embedding-3-large at $0.13/M → ~$104. gemini-embedding-001 at $0.15/M → ~$120. Cheap in dollars — the constraint is *throughput and correctness*, not price. Batch APIs cut cost ~50% but add up-to-24h latency windows, which changes the orchestration (submit → poll → merge).
- **Rate limits set the floor on wall-clock:** at 1M tokens/min you cannot finish in under ~13.3 hours; at 5M TPM, ~2.7 hours — before retries, throttling storms, and provider hiccups. This is where "8–9 hours for a million docs" comes from, and why a crash at hour 7 without checkpoints is a catastrophe.
- **ANN index rebuild:** HNSW construction over millions of high-dim vectors takes minutes-to-hours and real memory (pgvector: tune `maintenance_work_mem`, parallel workers, and always `CREATE INDEX CONCURRENTLY` on a live table).
- **Transient 2× storage:** dual columns / dual collections mean vectors + index exist twice until decommission. 1M × 3072 fp32 ≈ 12 GB of raw vectors per copy, plus index. A preflight disk check is not optional (alembicio's `doctor` does this).
- **Live writes don't pause:** documents are inserted, updated, and deleted during the multi-hour window. Without dual-write and reconciliation, the new space is stale the moment backfill finishes. Qdrant's own tutorial concedes the manual version can't handle deletes/partial updates mid-flight and tells you to pause them — a real orchestrator must do better (tombstone precedence + a reconciliation sweep).
- **Chunking drift (the hidden trap):** a new model usually means a new tokenizer and different max input length. If chunk boundaries were token-based, "re-embed each chunk" is no longer well-defined — the honest operation is re-*ingest*. See §5.

Correctness requirements that fall out of this: **exactly-once effect** per (document, target model) — achieved with idempotency keys `(doc_id, content_hash, target_model)` and idempotent upserts, so at-least-once delivery + dedup gives exactly-once outcomes; **durable checkpoints** `(migration_id, shard, last_committed_offset)` written transactionally with (or after) the vector writes they cover; **poison-document handling** (dead-letter queue, not a stalled run); **budget hard-stops** (max USD, max tokens) that fail safe and resumably.

---

## 3. Landscape: who is solving this, and how far they get

### 3.1 Vector DB vendors — manual patterns, published as tutorials

**Qdrant** publishes an official operations tutorial, "Migrate to a New Embedding Model," describing two zero-downtime options: (1) **blue-green** — create a second collection configured for the new model, dual-write every incoming upsert to both, background-scroll through points re-embedding into the new collection, then flip search traffic (via a **collection alias**, Qdrant's purpose-built cutover primitive) and disable dual writes; (2) on Qdrant ≥ 1.18 with **named vectors**, add the new model as a second named vector in place — easier and cheaper when your collection was configured for it. The tutorial explicitly warns that deletes and partial updates must be paused during the manual migration or the sets diverge. Everything is you-write-the-code: offsets, resume, verification, rollback.

**Postgres/pgvector** has the **dual-column pattern**, taught step-by-step by a Google Cloud engineer (April 2026, on AlloyDB): add `embedding_v2 vector(new_dim)` (the `vector(n)` type is fixed-dimension, so a dim change *requires* a new column), backfill in background jobs, run a **golden dataset** of critical queries against both columns, put the read path behind a **feature flag**, canary 5–10% of traffic, roll back instantly by toggling the flag, drop the old column in a later cleanup. Again: a hand-rolled procedure, not a tool.

**Elasticsearch / others:** dense_vector dims are immutable → changing models means delete-and-recreate indices plus full re-ingest (observed in the wild in the claude-code breakage issue). Pinecone's integrated-inference indexes are pinned to a model at creation → new model, new index. Weaviate similarly supports multiple named vectors per object, giving a dual-space primitive but no orchestration. The pattern across all vendors: primitives yes, orchestrator no.

### 3.2 Postgres ecosystem — pgai Vectorizer (closest incumbent)

Timescale (now TigerData)'s **pgai Vectorizer** is the nearest thing to "solved," and studying it sharpens alembicio's wedge. It makes embeddings **declarative**: `ai.create_vectorizer(...)` on a table, and a background worker creates embeddings and keeps them synced as rows change — "like declaring an index, but the vectorizer manages the embeddings." It supports **multiple embeddings of the same data with different models** for testing/experimentation, handles model failures and rate limits in the worker, and since April 2025 runs against **any** Postgres (self-hosted, RDS, Supabase) as a Python CLI/library.

What it doesn't do — the gap alembicio lives in: it's **Postgres-only**; it has **no golden-query recall/MRR verification** (you can create a second vectorizer, but nothing measures whether the new model is *better* or even *sane*); it has **no staged cutover / canary / rollback semantics** (the read-path flip is your problem); no **cost/token budgets**; and it wants to own the embedding lifecycle going forward (a worker you adopt), rather than being a **bounded migration** you run and finish. pgai is "embeddings as managed state, forever, in Postgres"; alembicio is "a migration with a beginning, a verification, and an end, on whatever DB you already have."

### 3.3 Commercial — Schift (and the direction of travel)

**Schift** (schift.io) has commercialized the learned-mapping approach, and they're further along than a stealth experiment: a CLI (`brew install schift`), SDKs in Python/JS/Go/Java, a unified embed API across providers, and an `upgrade(db=..., to=...)` call that switches an entire vector DB to a new model **without re-embedding**, via projection matrices they claim recover **99.7%** of retrieval quality "in minutes instead of days." Their blog documents the pain well (developer-forum quotes about being stuck on ada-002 with interlinked IDs; a documented text-embedding-004 → 3-large migration writeup). Treat the 99.7% as a vendor benchmark on favorable pairs — the peer-reviewed range (below) is 95–99%, pair-dependent — but their existence validates two things: the pain is worth paying for, and **no OSS standard exists** (or they couldn't sell this as a product).

Also note the platform trend: DB vendors absorbing embedding management (Pinecone integrated inference, Weaviate's embedding service, MongoDB's Voyage acquisition, pgai on Postgres). This *reduces* future migrations for teams who buy in early on one platform — and is the main long-term competitive risk to alembicio (§5) — but it doesn't help the enormous installed base with app-managed embeddings, and it deepens vendor lock-in, which cuts the other way.

### 3.4 Research — the theory is settled enough to build on

**Drift-Adapter** (Harshil Vejendla, EMNLP 2025 main, arXiv:2509.23471) is the formal treatment of the stopgap: learn a lightweight transform bridging spaces so the **existing ANN index keeps serving** while full re-embedding is deferred. Direction of mapping: **new queries → legacy space**. Three parameterizations evaluated — **Orthogonal Procrustes, Low-Rank Affine, compact Residual MLP** — trained on a small sample of paired old/new embeddings. Results on MTEB text corpora and a 1M-item CLIP upgrade: **95–99% of full re-embedding's Recall@10 / MRR**, **< 10 μs** added query latency, **> 100×** less recompute than re-indexing, with robustness analyses across drift severity, training-pair count, and billion-scale extrapolation. This is alembicio's mapping mode, with a citation instead of a promise.

**vec2vec** (Jha, Zhang & Shmatikov, arXiv:2505.12540) proves the *unsupervised* version: translations between spaces with **no paired data**, via a shared latent (their "Strong Platonic Representation Hypothesis"), hitting cosine similarity up to 0.92 to ground-truth targets and enabling attribute inference / inversion from vectors alone. For alembicio this is (a) theoretical reassurance that supervised Procrustes is the easy case, and (b) a security disclosure to include: embedding vectors are not anonymized data.

Lineage worth citing in the README for the math-inclined: Schönemann (1966) closed-form orthogonal Procrustes; the cross-lingual word-vector mapping line (Mikolov et al. 2013 linear maps; Xing et al. 2015 orthogonality; MUSE/Conneau et al. 2017 adversarial + Procrustes refinement); relative representations (Moschella et al. 2022) for anchor-based model stitching; Platonic Representation Hypothesis (Huh et al. 2024).

### 3.5 Adjacent tools that don't solve it

- **vector-io / VDF** (AI Northstar Tech): export/import across ~9 vector DBs via the Vector Dataset Format, plus a `reembed_vdf` CLI that re-embeds an exported snapshot offline with a new model. It's DB↔DB portability and offline batch re-embed — no live dual-write, no checkpoint/resume against a production store, no verification, no cutover. Complementary (alembicio could even emit/accept VDF later), not competing.
- **drift-spark** (github.com/aayush4vedi/drift-spark): Spark-native embedding lifecycle — `embed`, `watch`, `migrate` — with a reference implementation of Drift-Adapter's Procrustes upgrade path. Closest OSS spiritual sibling, but coupled to a Spark runtime; teams with a Postgres or Qdrant and no Spark cluster (most RAG teams) aren't served. Watch it; consider interop.
- **LangChain / LlamaIndex ingestion pipelines:** can re-run ingestion with a new embedder, with doc-store dedup — but no dual-write, no verification, no cutover; you're rebuilding from source, offline.
- **In-house rebuilds:** OpenViking issue #1523 (April 2026) specs blue-green sets, dual-write, `ov reindex --resume/--rollback/--abort`, `--retry-failed`, dry-run disk-space checks, health gates. This is the strongest evidence of unmet demand: sophisticated teams are writing alembicio's design doc inside their own issue trackers.

### 3.6 Gap matrix

| Capability | Vendor tutorials (DIY) | pgai Vectorizer | Schift | Drift-Adapter / drift-spark | vector-io | **alembicio** |
|---|---|---|---|---|---|---|
| DB-agnostic | — (per-DB recipe) | ✗ Postgres-only | ~ (pgvector + ?) | ✗ Spark-coupled (OSS impl) | ✓ many DBs | ✓ pgvector + Qdrant (v1) |
| Declarative migration file | ✗ | ~ (SQL config) | ✗ (API calls) | ✗ | ✗ | ✓ embmigrate.yaml |
| True re-embedding orchestration | manual | ✓ (sync-forever model) | ✗ (projection instead) | ✗ (defers it) | offline only | ✓ (bounded migration) |
| Dual-write during backfill | manual | ✓ (trigger-based sync) | ✗ | ✗ | ✗ | ✓ |
| Checkpointed, crash-resumable | manual | ~ (worker queue) | n/a | ✗ | ✗ | ✓ exactly-once effects |
| Rate-limit + cost budget | ✗ | ~ (rate-limit handling, no budget) | n/a | ✗ | ✗ | ✓ hard-stop budgets |
| Golden-query recall verification | manual suggestion | ✗ | opaque | offline eval in paper | ✗ | ✓ recall@k / MRR / overlap report |
| Staged cutover + rollback | manual (flags/aliases) | ✗ | ✗ | ✗ | ✗ | ✓ canary %, soak, one-command rollback |
| Projection stopgap mode | ✗ | ✗ | ✓ (closed) | ✓ (paper/reference) | ✗ | ✓ NumPy Procrustes, gated + honest |
| Open source | n/a | ✓ | ✗ | ✓ | ✓ | ✓ |

---

## 4. Solution space

### 4.1 The real fix: orchestrated re-embedding (alembicio core)

The migration is a **state machine**; every state is durable and every transition is resumable:

```
PLAN (doctor) → PREPARE → BACKFILL ⟲ → VERIFY → CUTOVER (canary→100%) → SOAK → DECOMMISSION
                              ↑______________rollback = flip read path back______________|
```

- **PLAN / `doctor`:** estimate tokens, dollars, wall-clock at the configured TPM, and transient disk; live-check provider auth, target model dims, and store schema headroom. Refuse to start what can't finish.
- **PREPARE:** pgvector → `ALTER TABLE ... ADD COLUMN emb__<model> vector(d)` plus a **dirty-flag trigger** (`needs_embed = true` on insert/update) — Postgres triggers give you change-capture for free, which is precisely why pgvector is the right first adapter. Qdrant → new collection (or new named vector on ≥ 1.18) + register the app's writes through a thin **dual-write SDK shim** (Qdrant has no triggers), with a **reconciliation scan** after backfill to catch anything the shim missed. Aliases are created here for the later flip.
- **BACKFILL:** shard the keyspace; each worker loop = claim batch → skip rows whose `(doc_id, content_hash, target_model)` already succeeded → embed (token-bucket rate limiter; budget counters; exponential backoff; circuit breaker) → idempotent upsert of vectors → commit checkpoint `(migration_id, shard, offset)`. Crash anywhere, resume anywhere; poison docs go to a dead-letter table with reasons. Optional Batch-API mode (submit/poll/merge) for 50% cost cuts.
- **VERIFY:** the phrase "prove recall didn't drop" needs precision, because *old and new rankings are supposed to differ* — that's the upgrade. Three honest measurements: (1) **known-item golden queries** (query → the doc that must be in top-k): new recall@k must be ≥ old — this is the gate; (2) **overlap@k / rank correlation** between old and new result lists — informational, expect moderate divergence; (3) **MRR/nDCG deltas** if you have labeled relevance. Golden sets come from real query logs when available, else LLM-generated known-item pairs from the corpus. Output is a report artifact (markdown/HTML) you can paste into a PR.
- **CUTOVER:** pgvector → flip the read path (a view swap or app-side flag alembicio manages) at a canary percentage, then 100%; Qdrant → **alias flip**, its native zero-downtime primitive. Dual-write continues.
- **SOAK:** configurable window (default 72h–7d) with both spaces maintained; rollback is a read-path flip back, instant, lossless.
- **DECOMMISSION:** drop the old column/collection and the trigger/shim. Migration over — alembicio leaves; nothing of it stays resident (contrast with pgai's forever-worker model).

**Exactly-once, precisely:** delivery is at-least-once (retries happen); *effects* are exactly-once via idempotency keys + idempotent upserts + checkpoints that only advance after their covered writes are durable. This is standard event-sourcing discipline (your internship maps directly) — and it's the part vendors' tutorials wave at with "resumable offsets."

### 4.2 The stopgap: projections between spaces (mapping mode)

**Orthogonal Procrustes, closed form (Schönemann 1966).** Sample n anchor texts, embed with both models to get paired matrices `A ∈ R^{n×d_s}` (source) and `B ∈ R^{n×d_t}` (target). Solve `min_{Q: QᵀQ=I} ‖AQ − B‖_F`: compute `M = AᵀB`, SVD `M = UΣVᵀ`, then `Q = UVᵀ`. One SVD of a d×d (or d_s×d_t, rectangular case — an orthonormal Stiefel frame, which also handles dimension changes like 768→3072) — pure `numpy.linalg`, milliseconds, no torch/scipy/sklearn, which matches your session constraints exactly. A few thousand anchors suffice; use held-out anchors to measure recovery, never the training anchors.

**The ladder, when orthogonal isn't enough:** + isotropic scaling (trace formula) → ridge-regularized affine `W = (AᵀA + λI)⁻¹AᵀB` → low-rank affine → small residual MLP (Drift-Adapter's three, in ascending capacity). Ship orthogonal + ridge in v1 (both NumPy-closed-form); leave MLP out until someone asks — it drags in a training loop and undermines the "boring and auditable" positioning.

**Direction matters.** Two distinct uses: (a) **new queries → old space** (Drift-Adapter's choice): deploy the new query model *today*, keep the old index serving, defer backfill — quality ceiling is the *old* model's geometry; (b) **old corpus → new space**: start serving from the new index immediately with projected placeholders that real re-embeddings progressively overwrite — the "bridge during backfill" mode, which is the natural fit inside alembicio's pipeline (projected vectors are just backfill rows with `provenance = projected`, replaced as true embeddings land).

**Why the ceiling is real (never oversell this).** A linear map cannot create distinctions the source geometry doesn't contain: if the new model separates two concepts the old model conflated, no `Q` recovers that separation from old vectors. Add nonlinear residuals between real model pairs, anisotropy/hubness mismatches, and out-of-distribution anchor drift, and you get the honest picture: peer-reviewed recovery is **95–99% of full re-embedding on favorable pairs** (Drift-Adapter), vendor-claimed 99.7% (Schift, their benchmark). Alembicio's rule: mapping mode **measures recovery on held-out anchors from the user's own corpus and refuses cutover below a configurable threshold** (default 0.95 known-item recall ratio), and the README states plainly that projection is a bridge, not a destination. Honesty is a differentiator against a closed vendor selling the opposite story. Security note in docs, citing vec2vec: embedding vectors (and mapping matrices) can leak document content; treat both as sensitive.

### 4.3 Structural mitigations people confuse with solutions

- **Store the raw text (or a pointer to it) + a content hash, always.** Non-negotiable precondition — Qdrant's tutorial assumes text in payloads for the same reason. `doctor` verifies this and fails early if vectors are orphaned from source text.
- **Hybrid retrieval as a bridge:** keeping BM25/keyword search in the ranking blend puts a floor under quality during the window. Recommend it in docs; don't build it.
- **MRL truncation:** flexibility within one model family only; relevant to choosing target dims, not to compatibility.
- **Self-hosted OSS models as an escape hatch:** a model you host is never deprecated *at* you. Real tradeoff (quality stagnation vs. sovereignty) — worth a docs page, and it's why the demo uses local models.
- **DB-owned embeddings (pgai / integrated inference):** genuinely reduces future migrations *if* you're on that platform and accept the lock-in; doesn't help the installed base, which is alembicio's market.

### 4.4 The interface: embmigrate.yaml + CLI

```yaml
# embmigrate.yaml
migration: ada002-to-3large-2026q3
source:
  provider: openai
  model: text-embedding-ada-002
  dim: 1536
target:
  provider: openai
  model: text-embedding-3-large
  dim: 3072
store:
  kind: pgvector
  dsn: env:DATABASE_URL
  table: documents
  id_column: id
  text_column: content          # or content_ref for external text
backfill:
  batch_size: 256
  budget: { max_usd: 150, max_tokens: 900_000_000 }
  rate_limit: { tpm: 4_000_000, rpm: 4_000 }
  on_poison: dead_letter
verify:
  golden_queries: golden.jsonl   # known-item pairs
  gates: { min_recall_at_10_ratio: 1.0, report: report.md }
cutover:
  mode: staged
  canary_pct: 5
  soak_hours: 72
mapping:                          # optional stopgap
  kind: procrustes                # procrustes | ridge | none
  anchors: 4096
  min_recovery: 0.95
```

CLI verbs: `alembicio init | doctor | prepare | backfill [--resume] | verify | cutover [--canary N] | rollback | decommission | status`. Every verb idempotent; every verb dry-runnable.

---

## 5. Constraints, traps, and honest risks

**Adapter sprawl kills solo devs.** Two backends for the first year: **pgvector** (triggers = free change-capture; dual-column is the cleanest possible pattern) and **Qdrant** (aliases = purpose-built cutover; named vectors on ≥ 1.18 or dual collections otherwise). Both primitives are vendor-documented, so the adapters implement blessed patterns rather than fighting the store. Everything else — Weaviate, Milvus, Elastic, Pinecone — is a "help wanted" label and a stable `StoreAdapter` protocol, not a promise.

**Chunking drift is the ugliest real-world wrinkle.** New tokenizer + new max-input-length ⇒ old chunk boundaries may be invalid or suboptimal, and chunk IDs stop being stable. MVP position: **preserve-chunks mode** (re-embed existing chunks verbatim; warn loudly if any chunk exceeds the target model's input limit, truncate-with-flag or dead-letter). **Re-chunk mode** (re-ingest from source with ID remapping and parent-doc bookkeeping) is v2 — scoping it out of the MVP is the difference between two weeks and two months.

**The CDC asymmetry.** Postgres gives triggers; Qdrant gives nothing server-side, so live writes during a Qdrant migration require either an app-side dual-write shim (a ~50-line wrapper around qdrant-client that alembicio provides) or acceptance of a reconciliation-only strategy (final scroll comparing versions/timestamps). Ship both; document the tradeoff. Never claim magic interception.

**Deletes and updates during backfill.** Tombstone precedence: a delete observed after a backfill write must win; an update marks the row dirty and re-enqueues. The reconciliation sweep at the end of BACKFILL is what turns "we paused deletes" (Qdrant's manual advice) into "we didn't have to."

**Eval-set honesty.** LLM-generated golden queries are biased toward what the corpus makes easy; say so in the report artifact, prefer real query logs, and show overlap@k as context rather than pretending recall@k on synthetic pairs is ground truth.

**Never claim mappings match re-embedding.** Gate cutover on measured recovery from the user's own held-out anchors; publish a recovery table per model pair in the repo (this doubles as content marketing that Schift structurally can't match, since their number is a sales claim).

**Competitive risk, stated plainly.** The platforms are absorbing embedding management (pgai on Postgres; integrated inference on managed DBs). If every DB eventually owns re-embedding, alembicio's window narrows to: cross-DB shops, self-hosted/OSS-model shops, lock-in-averse teams, and everyone mid-migration *today*. That's still large, and the verification/cutover/rollback semantics remain undone even where auto-re-embedding exists. But build for a 12–24 month relevance window, ship fast, and let the recall-report artifact be the thing people remember.

**Ops safety defaults:** dry-run by default for destructive verbs; budget hard-stops; secrets only via env indirection (`env:VAR` in YAML, never literals); `decommission` requires typed confirmation; vectors treated as sensitive data (vec2vec citation in SECURITY.md).

**Engineering conventions (binding for all generated code):** full-file outputs when modifying; never rename public symbols; new params keyword-only with defaults; type annotations on every signature; docstrings (one-line + Args/Returns) on public methods; the mapping/eval math module is **NumPy-only** (SVD via `numpy.linalg`) — no scipy/sklearn/torch anywhere in the core package; heavyweight demo deps (fastembed / sentence-transformers, docker fixtures) live in `[demo]`/`[dev]` extras.

---

## 6. Build workflow: Cursor × Claude

### 6.1 Division of labor

Three surfaces, one repo, zero conflicts (CLAUDE.md and .cursor/rules coexist):

- **claude.ai (Claude Fable 5):** the thinking surface. Design docs, invariants, adversarial reviews of correctness-critical diffs, launch writing. Fable 5 is Anthropic's highest-capability generally available model (GA June 9, 2026); it's also ~2× Opus pricing via API, so it's a scalpel, not a daily driver.
- **Claude Code (terminal or desktop; Claude family — Opus 4.8 for the hard runs, Sonnet 5 otherwise):** the repo-scale agent. Long TDD loops, multi-file refactors, "make the crash-injection suite pass," test triage. Reads CLAUDE.md every session.
- **Cursor (IDE):** the everyday loop. Sonnet 5 as the pinned agent default (close-to-Opus agentic quality at $2/$10 launch pricing, superseding Sonnet 4.6); Composer 2.5 / Auto pool for scaffolding and chores (cheaper included pool); Opus 4.8 pinned for stubborn debugging; GPT-5.3 Codex as a cross-model second opinion; Fusion powering Tab. Fable 5 exists in Cursor's picker but bills ~2× Opus — prefer doing Fable-grade work in claude.ai/Claude Code sessions where it's a deliberate act.

### 6.2 Model-to-task map

| # | Task | Surface | Model | Notes |
|---|---|---|---|---|
| 0 | DESIGN.md, INVARIANTS.md, YAML schema, `StoreAdapter`/`EmbeddingProvider` protocols | claude.ai | Fable 5 | One long extended-thinking session; outputs are the repo's constitution |
| 1 | Scaffold: pyproject, typer CLI, ruff+mypy+pytest, docker-compose (pg16+pgvector, qdrant), CI | Cursor agent | Composer 2.5 / Auto | Cheap pool; review lightly |
| 2 | Checkpoint store + resumable worker | Claude Code | Opus 4.8 | Hypothesis property tests **written and approved first**; then implementation until green |
| 3 | Crash-injection harness (kill -9 mid-batch; assert no dupes, no gaps, budget monotone) | Claude Code | Opus 4.8 | The harness is the product's credibility |
| 4 | pgvector adapter (prepare/trigger, backfill, `CREATE INDEX CONCURRENTLY`, cutover flip) | Cursor agent | Sonnet 5 | Integration tests against docker Postgres |
| 5 | Providers + token-bucket budgeter (openai, gemini, fastembed-local) | Cursor agent | Sonnet 5 | Record/replay HTTP fixtures; no live keys in CI |
| 6 | Eval harness: recall@k, MRR, overlap@k, report renderer | Claude Code | Opus 4.8 | Unit tests against hand-computed metric cases |
| 7 | Procrustes/ridge mapping module (NumPy-only) | claude.ai → Cursor | Fable 5 derive/verify, Sonnet 5 glue | Sanity test: recover a synthetic random rotation to machine precision before real pairs |
| 8 | Adversarial review of any diff touching checkpoint/cutover/reconciliation | claude.ai | Fable 5 | Standing prompt: "construct an interleaving of crash, retry, concurrent write, delete that violates exactly-once or loses a tombstone" |
| 9 | Stubborn-bug second opinion | Cursor | GPT-5.3 Codex | Cross-model triangulation |
| 10 | README, docs site, launch post, recovery-table writeups | Cursor / claude.ai | Sonnet 5 draft → Fable 5 edit | Ops tools are sold by their docs |
| — | Tab autocomplete all day | Cursor | Fusion | Free speed |

### 6.3 Guardrails for agent-written correctness code

1. **INVARIANTS.md is law, mirrored verbatim into `.cursor/rules` and `CLAUDE.md`** so every agent session — either tool — carries: exactly-once effect semantics; checkpoint-after-durable-write ordering; tombstone precedence; budget monotonicity; idempotent CLI verbs; the engineering conventions from §5.
2. **Tests before core.** Property tests (Hypothesis: random batch sizes, injected failures at every step index, shuffled interleavings) and the kill -9 harness exist and are agreed on before the worker is written. Agents then code to the harness, not to vibes.
3. **Small diffs, mandatory review lane.** Anything under `core/checkpoint`, `core/cutover`, `core/reconcile` gets a Fable 5/Opus adversarial pass before merge; everything else can flow on Sonnet-5-and-CI.
4. **Two-backend discipline enforced structurally:** adapters implement a frozen `Protocol`; the test suite runs the *same* migration scenario matrix against both docker backends, so a third adapter later is "implement protocol, inherit the matrix."

### 6.4 Two-week MVP calendar

| Day | Deliverable |
|---|---|
| 1 | Fable 5 design session → DESIGN.md, INVARIANTS.md, embmigrate.yaml v0 schema, protocols |
| 2 | Scaffold + CI + docker-compose green (Composer/Auto) |
| 3–4 | Checkpoint store + worker skeleton, property tests passing (Claude Code + Opus 4.8) |
| 5 | Providers + token-bucket + budget hard-stops; fastembed local provider (Sonnet 5) |
| 6–7 | pgvector adapter end-to-end: prepare → backfill → index build on docker corpus |
| 8 | Crash-injection harness green: kill -9 at random offsets × 50 runs, zero dupes/gaps; reconciliation sweep |
| 9 | Eval harness + report artifact (recall@k / MRR / overlap@k) |
| 10 | cutover / rollback / decommission verbs + `status` |
| 11 | `doctor`: cost/time/disk estimator + live checks |
| 12 | Demo: 100k synthetic docs, MiniLM → bge-small via fastembed (no API keys), mid-run kill -9 + resume on camera, recall report; record asciinema |
| 13 | README + docs (Sonnet 5 draft, Fable 5 edit), example configs, SECURITY.md |
| 14 | Polish, tag v0.1.0, launch post ("Google deleted text-embedding-004 in January. Here's the tool that should have existed.") |
| wk 3–4 | Qdrant adapter (dual-collection + alias flip + SDK shim + reconciliation), mapping mode GA with recovery tables |

Demo constraints honored: free local models only (fastembed / sentence-transformers in `[demo]` extras), docker everything, zero paid keys required to reproduce the README gif.

---

## 7. GitHub metadata

**About** (≤350 chars):

> Alembic for vectors — zero-downtime embedding model migrations. Declarative YAML migrations, dual-write, checkpointed & resumable re-embedding with rate-limit/cost budgets, golden-query recall verification, staged cutover, one-command rollback. pgvector + Qdrant. Optional Procrustes projection as an honest stopgap.

**Tagline options:**
1. "Your embedding model just got deprecated. Migrate a million vectors with zero downtime — and prove recall didn't drop."
2. "Embedding models die. Your search shouldn't."
3. "Schema migrations got Alembic. Vector migrations got a tutorial. Until now."

**Topics:** `embeddings` · `vector-database` · `pgvector` · `qdrant` · `rag` · `semantic-search` · `database-migrations` · `mlops` · `zero-downtime` · `retrieval-augmented-generation`

**README opener:**

```markdown
# alembicio

**Alembic for vectors.** Embedding models get deprecated on an annual cadence now
(Google shut down text-embedding-004 in Jan 2026; Azure retired ada-002). Vectors
from two models are mutually meaningless — even at identical dimensionality — so
your vector DB is silently welded to a dead model, and every dashboard stays green
while users get garbage.

`alembicio` migrates it properly: declarative migration files, dual-write,
checkpointed re-embedding that survives crashes and rate limits, golden-query
recall verification, staged cutover, one-command rollback. Runs once, verifies,
leaves. pgvector and Qdrant.

    pip install alembicio
    alembicio init && alembicio doctor && alembicio backfill
    alembicio verify && alembicio cutover --canary 5
```

**Suggested repo layout:** `alembicio/{cli,core/{state,checkpoint,worker,budget,reconcile},adapters/{pgvector,qdrant},providers/{openai,gemini,fastembed},mapping/{procrustes,ridge},eval/{golden,metrics,report}}` + `tests/{property,crash,integration}` + `examples/` + `docs/`.

---

## 8. Sources

**Deprecations / incompatibility (first-party where possible)**
- Google Developers Blog — gemini-embedding-001 GA; embedding-001 EOL Aug 14 2025, text-embedding-004 EOL Jan 14 2026, MRL, $0.15/M: https://developers.googleblog.com/gemini-embedding-available-gemini-api/
- Google Gemini API docs — "embedding spaces between gemini-embedding-001 and gemini-embedding-2 are incompatible … you must re-embed": https://ai.google.dev/gemini-api/docs/embeddings
- Breakage in the wild — anthropics/claude-code#23557 (768→3072, ES indices recreated): https://github.com/anthropics/claude-code/issues/23557 ; firebase/genkit#4551: https://github.com/firebase/genkit/issues/4551
- Azure — ada-002 deprecated Jan 4 2025 (Microsoft Q&A / model lifecycle): https://learn.microsoft.com/en-us/answers/questions/5622204/ ; lifecycle policy: https://learn.microsoft.com/en-us/azure/foundry/openai/concepts/model-retirements
- OpenAI deprecations page (first-gen embeddings shutdown Jan 4 2024; notice policy): https://platform.openai.com/docs/deprecations

**Vendor migration patterns**
- Qdrant, "Migrate to a New Embedding Model" (blue-green, dual-write, alias flip, named-vector option ≥1.18, pause-deletes caveat): https://qdrant.tech/documentation/tutorials-operations/embedding-model-migration/
- Google Cloud community (R. Samborski, Apr 2026) — dual-column on AlloyDB, golden dataset, feature-flag canary, rollback: https://medium.com/google-cloud/migrating-vector-embeddings-in-production-without-downtime-8a0464af6f55

**Incumbents / adjacent**
- Schift — org + product claims (projection matrices, 99.7% recovery, SDKs): https://github.com/schift-io ; blog: https://schift.io/blog/why-vector-migration-matters/
- TigerData pgai Vectorizer — declarative embeddings, auto-sync, multi-model experimentation, any-Postgres: https://github.com/timescale/pgai ; https://www.tigerdata.com/blog/pgai-vectorizer-now-works-with-any-postgres-database ; https://pypi.org/project/pgai/
- vector-io / VDF (+ reembed_vdf): https://github.com/AI-Northstar-Tech/vector-io
- drift-spark (Spark-native lifecycle, Drift-Adapter reference impl): https://github.com/aayush4vedi/drift-spark
- In-house reinvention — volcengine/OpenViking#1523 (blue-green, dual-write, resume/rollback/abort, dry-run disk checks): https://github.com/volcengine/OpenViking/issues/1523

**Research**
- Vejendla, "Drift-Adapter" (EMNLP 2025 main; Procrustes / low-rank affine / residual MLP; 95–99% recovery; <10μs; >100× recompute reduction): https://arxiv.org/abs/2509.23471 ; https://aclanthology.org/2025.emnlp-main.805/
- Jha, Zhang, Shmatikov, "Harnessing the Universal Geometry of Embeddings" (vec2vec; unsupervised translation; cos ≤ 0.92; security implications): https://arxiv.org/abs/2505.12540 ; https://vec2vec.github.io/
- Background lineage (from general literature): Schönemann 1966 (orthogonal Procrustes closed form); Mikolov et al. 2013 / Xing et al. 2015 / Conneau et al. 2017 (cross-lingual mapping); Moschella et al. 2022 (relative representations); Huh et al. 2024 (Platonic Representation Hypothesis)

**Tooling (workflow section)**
- Cursor models & pricing (Sonnet 5, Opus 4.8, Fable 5, GPT-5.5/5.3-Codex, Gemini 3.1/3.5, Composer 2.5, Auto pool): https://cursor.com/docs/models-and-pricing
- Anthropic model docs (Fable 5 GA Jun 9 2026; Opus 4.8; Sonnet 5): https://platform.claude.com/docs/en/about-claude/models/overview
- Sonnet 5 in Cursor guidance (defaults, when to escalate to Opus 4.8): https://apidog.com/blog/claude-sonnet-5-cursor/
