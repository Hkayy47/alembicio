# alembicio

**Alembic for vectors.** Embedding models get deprecated on an annual cadence now
(Google shut down `text-embedding-004` in Jan 2026; Azure retired `ada-002`). Vectors
from two models are mutually meaningless — even at identical dimensionality — so your
vector DB is silently welded to a dead model, and every dashboard stays green while
users get garbage.

`alembicio` migrates it properly: declarative migration files, dual-write, checkpointed
re-embedding that survives crashes and rate limits, golden-query recall verification,
staged cutover, one-command rollback. It runs once, verifies, and leaves.

```bash
pip install alembicio
alembicio init && alembicio doctor
alembicio prepare && alembicio backfill      # crash it; `backfill --resume` continues
alembicio verify                              # recall@k / MRR report, gate enforced
alembicio cutover --canary 5 && alembicio cutover
alembicio decommission                        # after the soak window
```

## Status

Pre-alpha, built in the open. v0.1 targets **pgvector** end-to-end (dual-column,
trigger-based dual-write, predicate-driven resume, halfvec for >2000-dim models,
view-swap cutover). **Qdrant** (dual collection + alias flip) is v0.2. Mapping mode
(orthogonal Procrustes / ridge, pure NumPy) ships as a *measured stopgap*: it gates on
recovery computed from your own held-out anchors and never claims parity with true
re-embedding.

## Why not just re-embed in place?

Because the window is hours long at API rate limits, writes don't pause, and a query
against a half-old/half-new column returns confident garbage with zero errors to alert
on. See `docs/research-report.md` for the full problem analysis, the landscape (Qdrant's
manual tutorial, the dual-column pattern, pgai Vectorizer, Schift, Drift-Adapter), and
why this tool exists.

Start here: `DESIGN.md` (architecture) · `INVARIANTS.md` (the law) · `PROMPTS.md`
(how each phase gets built) · `SECURITY.md` (embeddings are sensitive data).
