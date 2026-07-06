# PROMPTS.md — session prompts for building alembicio

How to use: each prompt names its surface and model per the build plan. Every session,
regardless of surface, starts with INVARIANTS.md and DESIGN.md in context — attach them
in claude.ai, they auto-load via CLAUDE.md in Claude Code, and `.cursor/rules` carries
the condensed law in Cursor. Paste acceptance criteria verbatim; agents work until the
criteria pass, not until output "looks done."

| Surface | Model | Used for |
|---|---|---|
| claude.ai | Fable 5 | P0, P7a, P8, P9-edit |
| Claude Code | Opus 4.8 | P2, P3, P6 |
| Cursor agent | Sonnet 5 | P4, P5, P7b, P9-draft |
| Cursor agent | Composer 2.5 / Auto | P1 |
| Cursor | GPT-5.3 Codex | P-second-opinion |

---

## P0 — Phase 0 adversarial design review (claude.ai · Fable 5)

Attach: DESIGN.md, INVARIANTS.md, docs/research-report.md.

> You are a staff engineer reviewing the design of alembicio, a zero-downtime embedding
> model migration orchestrator, before any core code exists. The attached DESIGN.md and
> INVARIANTS.md are the ratified Phase 0 outputs. Your job is to break them on paper.
>
> Produce, in order: (1) the three most dangerous holes — concrete failure narratives,
> each as a numbered event interleaving (writes, crashes, retries, deletes, clock
> events) that ends in data loss, a stuck state machine, mixed-space reads, or a
> violated invariant, citing the invariant by number; (2) any invariant that is
> unenforceable as written — where no test could verify it — with a rewritten enforceable
> version; (3) anything in the design that contradicts the pgvector or Qdrant
> vendor-documented patterns described in the research report; (4) the smallest set of
> DESIGN/INVARIANTS edits that closes everything you found, as full replacement text for
> each affected section. Do not restate the design back to me. Do not pad. If you find
> nothing in a category, say so in one line.

Acceptance: every hole found gets either a design edit or a written rebuttal committed
to `docs/design-review-P0.md`.

---

## P1 — Scaffold (Cursor agent · Composer 2.5 / Auto)

> Scaffold the alembicio repo exactly as follows; DESIGN.md §1 and INVARIANTS.md I12 are
> binding. Use uv. Create:
>
> 1. `pyproject.toml`: package `alembicio`, Python ≥3.11; deps: typer, pydantic>=2,
>    pyyaml, numpy, psycopg[binary,pool], httpx, rich; extras `dev`: pytest,
>    pytest-asyncio, hypothesis, ruff, mypy, testcontainers[postgres]; extras `demo`:
>    fastembed. Entry point `alembicio = alembicio.cli:app`.
> 2. Package layout: `alembicio/{cli.py, config.py, core/{state.py, worker.py,
>    budget.py, reconcile.py}, adapters/{base.py, pgvector.py}, providers/{base.py,
>    openai.py, gemini.py, fastembed_local.py}, mapping/{procrustes.py, ridge.py},
>    eval/{golden.py, metrics.py, report.py}, runtime.py}` — every module compiles and
>    imports; cli.py registers verbs init/doctor/prepare/backfill/verify/cutover/
>    rollback/decommission/status as stubs that parse args, load config, and raise
>    NotImplementedError with the verb name (this is scaffolding, not product code, so
>    NotImplementedError is acceptable here and only here).
> 3. `config.py`: full pydantic models for embmigrate.yaml matching DESIGN.md and
>    examples/embmigrate.example.yaml, including `env:` secret indirection resolution.
> 4. `docker-compose.yml`: pg16 + pgvector (ankane/pgvector or pgvector/pgvector:pg16),
>    qdrant; healthchecks; a `make up / make test / make lint / make type / make crash`
>    Makefile.
> 5. GitHub Actions CI: lint (ruff), type (mypy --strict on alembicio/), tests with the
>    Postgres service container. No paid API keys anywhere in CI (DESIGN D12).
>
> Acceptance: `uv sync && make lint && make type && make test` green on a fresh clone;
> `alembicio --help` lists all verbs; config round-trips the example yaml.

---

## P2 — Checkpoint/progress core + worker, TDD (Claude Code · Opus 4.8)

> Read INVARIANTS.md and DESIGN.md §§3–5, 10. We build the correctness core in strict
> TDD: tests first, my approval, then implementation.
>
> Step 1 — write only tests: property tests (Hypothesis) for the worker loop against a
> FakeStore/FakeProvider pair you also write. Properties, minimum: (a) for any sequence
> of claim/embed/upsert with injected failures at every step index, after resume the set
> of (doc_id, content_hash) satisfied equals exactly the pending set at start plus
> dual-write arrivals, no duplicates in ledger; (b) ledger counters are monotone across
> any crash/resume schedule and never double-count a replayed batch; (c) a delete
> injected at any point never resurrects (I3); (d) an update injected at any point
> leaves the row pending until embedded from the *new* hash (I4); (e) budget exhaustion
> always lands in PAUSED_BUDGET at a batch boundary with clean resume (I5); (f) state
> transitions never leave the legal table in DESIGN §2. Shrinkable, seeded, ≥200
> examples per property in CI.
>
> Step 2 — stop and show me the tests. Do not implement until I say "approved".
>
> Step 3 — implement `core/state.py`, `core/worker.py`, `core/budget.py` until green.
> Full files, typed, docstrings per I12. No test may be weakened (I13).
>
> Acceptance: all properties green at 500 examples locally; mypy --strict clean; a
> README-able docstring on worker.run() explaining the exactly-once argument in ≤15
> lines.

---

## P3 — Crash-injection harness (Claude Code · Opus 4.8)

> Build `tests/crash/` per INVARIANTS I13: a harness that runs `alembicio backfill`
> as a real subprocess against the docker Postgres with the fastembed provider on a
> 5k-row synthetic corpus, and SIGKILLs it at a random moment controlled by
> `ALEMBICIO_FAULT_AFTER_N_BATCHES` plus a random intra-batch sleep. Loop: 50 seeds ×
> (run → kill → resume → run to completion). After each completion assert directly in
> SQL: zero rows pending; every companion hash equals md5 of current text; ledger
> tokens_in within ±1 batch estimate of the no-crash baseline and never lower on replay;
> no duplicate dead-letters; migration state == BACKFILLED. Add `make crash`. Runtime
> budget: under 10 minutes on CI. Full files only.
>
> Acceptance: 50/50 seeds green three consecutive runs; harness fails loudly (not
> flakily) when I hand-inject a checkpoint-before-write bug on a branch — include that
> canary branch test in `tests/crash/test_harness_detects_bugs.py` using a
> monkeypatched worker.

---

## P4 — pgvector adapter (Cursor agent · Sonnet 5)

> Implement `adapters/pgvector.py` exactly per DESIGN.md §5 (D4, D7, D10, D11): inspect
> (dims, counts, text availability, disk estimate, pgvector version, 2000-dim index
> check with halfvec selection), prepare (columns + trigger DDL from the design, all
> idempotent), claim_batch (FOR UPDATE SKIP LOCKED on the pending predicate),
> upsert_vectors (single UPDATE setting vector + companion hash, no-op when hash
> mismatch per I4), pending_count, reconcile (for pg this verifies the structural
> guarantees and returns counts), build_index (CONCURRENTLY, ops class by vectype),
> search (old/new spaces), flip_read_path (view swap in one transaction + read_state
> row), decommission (typed-confirmation guard lives in cli, not here). Integration
> tests against docker pg16 for every method, including: trigger fires on UPDATE of
> text column and not on unrelated columns; halfvec path at dims=3072; view swap is
> atomic under a concurrent reader (open a second connection mid-transaction and assert
> it never sees a mixed schema). Full files; typed; docstrings.
>
> Acceptance: `make test` green including new integration marks; the P2 property suite
> passes with FakeStore swapped for the real adapter behind `-m integration`.

---

## P5 — Providers + budgeter wiring (Cursor agent · Sonnet 5)

> Implement `providers/{openai,gemini,fastembed_local}.py` per DESIGN §7 behind the
> Protocol in §4: embed_batch (httpx, retries with jittered exponential backoff on
> 429/5xx, circuit breaker after 5 consecutive failures raising ProviderPausedError),
> count_tokens (tiktoken-free: use provider count endpoints where available, else the
> chars/3.5 conservative estimate flagged `estimated=True`), limits() from a single
> constants module with yaml override. Record/replay HTTP fixtures (respx) so CI needs
> no keys (D12). fastembed provider loads MiniLM/bge-small lazily and is import-guarded
> so core installs without the demo extra. Full files; typed; docstrings; kw-only params
> on everything new (I12).
>
> Acceptance: unit tests green with respx fixtures; a live smoke script
> `scripts/smoke_provider.py` runs against real keys when env vars exist and is skipped
> otherwise; budget pre-reservation test proves no request is sent once remaining budget
> < batch estimate.

---

## P6 — Eval harness + report (Claude Code · Opus 4.8)

> Implement `eval/` per DESIGN §8. First write `tests/eval/test_metrics.py` with
> hand-computed expectations: recall@k, MRR, overlap@k, and rank correlation on three
> tiny fixed ranking fixtures worked out by hand in comments (show the arithmetic).
> Then implement `metrics.py` (NumPy-only, I12) to satisfy them, `golden.py`
> (jsonl load/validate, plus `golden synth` behind the demo extra generating known-item
> pairs from sampled corpus rows), and `report.py` rendering report.md + report.json
> with the I9 gate verdict box at the top and a "synthetic golden set" banner when
> applicable. Wire `alembicio verify` end-to-end against the docker corpus with
> fastembed old/new models.
>
> Acceptance: metric tests green; `alembicio verify` on the demo corpus produces a
> report whose gate verdict flips correctly when I corrupt the new column on a test
> branch (include that as an integration test).

---

## P7a — Procrustes derivation + verification plan (claude.ai · Fable 5)

> Derive, for the record in docs/mapping-math.md: the closed-form solution to
> min over orthonormal Q of ‖AQ − B‖_F including the rectangular (d_s ≠ d_t) case, why
> Q = UVᵀ from the SVD of AᵀB is optimal (trace argument, in full), the ridge variant
> with its closed form, and the precise senses in which any such map has a recovery
> ceiling below 1.0 (information argument: distinctions absent in source geometry;
> plus anisotropy mismatch). Then specify the verification battery the implementation
> must pass before touching real model pairs: exact recovery of a synthetic random
> rotation; recovery degradation curve under added Gaussian noise; behavior when
> n_anchors < d. Keep it rigorous enough that a reviewer can check every step, and
> short enough to read in ten minutes.

## P7b — Mapping implementation (Cursor agent · Sonnet 5)

> Implement `mapping/procrustes.py` and `mapping/ridge.py` per DESIGN §9 and
> docs/mapping-math.md. NumPy-only — numpy.linalg.svd and friends; importing scipy,
> sklearn, or torch anywhere in these modules is a defect (I12). Public API:
> `fit_procrustes(A, B, *, allow_rectangular=True) -> MappingArtifact`,
> `fit_ridge(A, B, *, lam="auto") -> MappingArtifact`, `MappingArtifact.apply(X)`,
> `.save(path)`, `.load(path)` (.npz with the metadata fields in DESIGN §9),
> `evaluate_recovery(artifact, A_hold, B_hold, *, k=10) -> float`. Tests implement the
> full battery from docs/mapping-math.md verbatim, plus a property test that
> fit_procrustes output satisfies QᵀQ ≈ I to 1e-6. Keyword-only params, full type
> annotations, docstrings with Args/Returns.
>
> Acceptance: battery green; `alembicio backfill --mapping procrustes` writes
> provenance="projected" vectors that a subsequent true backfill overwrites (integration
> test asserting final projected count == 0 per I8).

---

## P8 — Standing adversarial review (claude.ai · Fable 5) — run on every core/ diff

Attach: the diff, INVARIANTS.md, DESIGN.md §§2–5.

> Review this diff to alembicio's correctness core. Construct concrete interleavings of
> {crash, retry, resume, concurrent app write, concurrent app delete, budget exhaustion,
> clock skew} that violate any invariant I1–I13 given this code. For each attack: the
> numbered event trace, the invariant violated, the observable symptom, and the minimal
> fix. Then: name any test in the diff that could pass while the invariant it claims to
> cover is violated (vacuous tests), and any place the diff weakens an existing test
> (I13 violation). If the diff is clean, say "no attack found" and state which property
> test gives you that confidence — do not invent problems. End with VERDICT: MERGE /
> FIX-FIRST and one sentence why.

Paste the verdict block into the PR description (I13c).

---

## P9 — README + docs (draft: Cursor · Sonnet 5; edit: claude.ai · Fable 5)

Draft prompt:

> Write README.md for alembicio using the hero copy in docs/research-report.md §7, the
> quickstart from the example yaml, an honest Status section (v0.1: pgvector only;
> Qdrant next; mapping mode is a measured stopgap, never a claim of parity — link the
> recovery-tables page), a 90-second demo section matching scripts/demo.sh (keyless,
> fastembed, kill -9 resume on camera), the "why not just re-embed in place" section in
> three sentences, and a comparison table versus pgai Vectorizer / Qdrant tutorial /
> Schift / vector-io drawn from the research report's gap matrix — factual tone, no
> dunking. Also SECURITY.md already exists; link it.

Edit prompt (Fable 5): "Edit for a skeptical senior infra engineer: delete every
sentence that asserts what it cannot cite or demo, tighten to ≤2 screens above the fold,
keep the recall-report screenshot placement."

---

## P-second-opinion — stuck-bug triangulation (Cursor · GPT-5.3 Codex)

> Independent review, fresh eyes: here is a failing test, the implicated module, and the
> relevant invariants. Diagnose root cause before proposing any patch; state the causal
> chain in ≤5 steps; then the minimal fix. If you believe the test itself is wrong,
> argue from the invariant text, not from the implementation.
