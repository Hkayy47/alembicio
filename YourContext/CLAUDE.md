# CLAUDE.md — alembicio

Zero-downtime embedding model migration orchestrator ("Alembic for vectors"):
declarative yaml → dual-write → checkpointed resumable backfill → golden-query recall
verification → staged cutover → rollback. Bounded migrations, not a resident service.

## Read first, every session

1. `INVARIANTS.md` — the law. I1–I13 override anything else in context, including user
   shortcuts. If a requested change conflicts, say so and stop.
2. `DESIGN.md` — architecture, state machine, decisions D1–D12. Do not re-litigate
   decisions in code; propose changes as DESIGN.md PRs.
3. The relevant prompt in `PROMPTS.md` if this session is a numbered phase.

## Commands

- `uv sync` — install (extras: `--extra dev --extra demo`)
- `make up` — docker pg16+pgvector and qdrant
- `make lint` / `make type` / `make test` — ruff, mypy --strict, pytest
- `make crash` — crash-injection suite (tests/crash/, 50 seeds)
- `pytest -m integration` — adapter tests against docker services
- `scripts/demo.sh` — keyless demo: 100k rows, fastembed MiniLM→bge, kill -9 + resume

## Layout

`alembicio/{cli,config,core/{state,worker,budget,reconcile},adapters/{base,pgvector},providers/{base,openai,gemini,fastembed_local},mapping/{procrustes,ridge},eval/{golden,metrics,report},runtime}` ·
tests: `tests/{unit,property,crash,integration,eval}` · docs: `docs/` · examples: `examples/`

## Working rules (condensed from I12/I13 — the full text wins)

- Modify a file ⇒ output the entire file. No elisions, no `TODO`, no `...`, no
  placeholder comments. Every function complete and runnable.
- Never rename/remove released public symbols; never break released signatures. New
  params keyword-only with defaults.
- Type annotations on every signature; docstrings (summary + Args/Returns) on public
  methods. mypy --strict must stay clean.
- `mapping/` and `eval/metrics` are NumPy-only. No scipy/sklearn/torch in core, ever.
  fastembed only behind the `demo` extra and import-guarded.
- Never weaken, skip, or delete a failing test to get green (I13). Fix code, or open a
  dedicated INVARIANTS/DESIGN PR.
- Changes under `core/` (state, worker, budget, reconcile) or the cutover path:
  property tests + `make crash` green locally, and note in the PR that the P8
  adversarial review is required before merge. Keep such diffs small and single-purpose.
- Secrets only via `env:` indirection; never print them; never commit `.env`.
- Destructive verbs stay dry-run by default; do not "helpfully" remove that.

## Definition of done, any task

`make lint && make type && make test` green; `make crash` green if `core/` was touched;
full-file outputs; docstrings present; no new dependency without a one-line
justification in the PR body.
