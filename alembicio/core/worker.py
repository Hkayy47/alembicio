"""Resumable, budget-aware backfill worker loop (DESIGN.md §4, INVARIANTS I1-I6).

The worker owns no backend- or provider-specific logic. It composes four collaborators
-- a :class:`BackfillStore`, an :class:`~alembicio.providers.base.EmbeddingProvider`, a
:class:`ControlPlane`, and a :class:`~alembicio.core.budget.BudgetTracker` -- into the
exactly-once pipeline of DESIGN.md §4:

    claim_batch -> budget.reserve (in-memory) -> embed -> upsert_vectors -> ledger.commit

Every durability decision is delegated to the store (whose companion-hash write *is* the
done-record on pgvector, D10) and the control plane (whose ledger/state writes are
transactional with progress). :func:`drive_backfill` is written against narrow Protocols
precisely so it can be property-tested against in-memory fakes with injected crashes.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from typing import Protocol, TypedDict

import numpy as np
import numpy.typing as npt

from alembicio.adapters.base import DocRecord, ReconcileReport, VectorRecord
from alembicio.config import EmbMigrateConfig
from alembicio.core.budget import BudgetExhaustedError, BudgetTracker, TokenBucket
from alembicio.core.state import RESUMABLE_ENTRY, MigrationState
from alembicio.providers.base import (
    EmbeddingProvider,
    PoisonInputError,
    ProviderPausedError,
    TransientProviderError,
)


class WorkerError(RuntimeError):
    """Raised for worker misconfiguration or an unwired execution backend."""


# Crash-injection contract (honoured only when the env vars are set; a no-op otherwise).
# The crash-injection harness (tests/crash/) sets these to hold the process in the
# sharpest exactly-once window -- after a batch is durably upserted but before its ledger
# row commits -- so an external SIGKILL lands there. Production never sets them.
FAULT_AFTER_N_BATCHES_ENV = "ALEMBICIO_FAULT_AFTER_N_BATCHES"
FAULT_INTRA_BATCH_SLEEP_MS_ENV = "ALEMBICIO_FAULT_INTRA_BATCH_SLEEP_MS"


def _env_int(name: str) -> int | None:
    """Return an int-valued environment variable, or None if unset/invalid."""
    raw = os.environ.get(name)
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _maybe_inject_fault(
    batches_upserted: int, *, sleeper: Callable[[float], None]
) -> None:
    """Hold the process in the post-upsert/pre-commit window for crash injection.

    Active only when ``ALEMBICIO_FAULT_AFTER_N_BATCHES`` is set (tests). Once at least
    that many batches have been durably upserted, it sleeps for
    ``ALEMBICIO_FAULT_INTRA_BATCH_SLEEP_MS`` so a harness SIGKILL lands after the durable
    write but before the ledger commit -- the sharpest test of exactly-once resume.

    Args:
        batches_upserted: Number of batches durably upserted so far this run.
        sleeper: Sleep function, injectable for tests.

    Returns:
        None.
    """
    threshold = _env_int(FAULT_AFTER_N_BATCHES_ENV)
    if threshold is None or batches_upserted < threshold:
        return
    sleep_ms = _env_int(FAULT_INTRA_BATCH_SLEEP_MS_ENV) or 0
    if sleep_ms > 0:
        sleeper(sleep_ms / 1000.0)


class LedgerSnapshot(TypedDict):
    """Point-in-time view of committed spend."""

    tokens_in: int
    usd_est: float


class BackfillStore(Protocol):
    """The store surface the worker depends on (a subset of ``StoreAdapter``).

    Durability lives entirely behind these four methods: ``upsert_vectors`` must be an
    idempotent, guarded write (I1/I3/I4), and ``claim_batch``/``pending_count`` must
    derive pending work from durable state only (I2), excluding dead-lettered keys.
    """

    def pending_count(self) -> int:
        """Return the number of rows still requiring a fresh embed (excludes dead-letters)."""
        ...

    def claim_batch(self, *, limit: int) -> list[DocRecord]:
        """Claim up to ``limit`` pending, non-dead-lettered rows for embedding."""
        ...

    def upsert_vectors(self, batch: list[VectorRecord]) -> None:
        """Idempotently persist vectors keyed by ``(doc_id, content_hash)`` (I1/I3/I4)."""
        ...

    def reconcile(self) -> ReconcileReport:
        """Apply the deletes-win sweep and dirty re-enqueue before completion."""
        ...


class ControlPlane(Protocol):
    """Progress, spend, and dead-letter surface (Postgres schema or SQLite mirror).

    Implementations MUST reject illegal state transitions (DESIGN.md §2 via
    :func:`alembicio.core.state.assert_transition`) and keep ``commit_spend`` monotone.
    """

    def get_state(self) -> MigrationState:
        """Return the durable migration state."""
        ...

    def set_state(self, state: MigrationState, /) -> None:
        """Transition to ``state``, validating legality; raise on an illegal move."""
        ...

    def commit_spend(self, *, tokens: int, usd: float) -> None:
        """Add committed spend to the monotone ledger (I5)."""
        ...

    def record_dead_letter(
        self, *, doc_id: str, content_hash: str, reason: str, error: str
    ) -> None:
        """Idempotently record a poison key so the run can continue (no duplicates)."""
        ...

    def spend(self) -> LedgerSnapshot:
        """Return the current committed spend."""
        ...


def _text_of(record: DocRecord) -> str:
    """Return the inline text for a claimed row.

    Args:
        record: A claimed document row.

    Returns:
        The row's text.

    Raises:
        WorkerError: If only a ``content_ref`` is available (not supported in the core
            loop; content-ref fetch is an adapter concern).
    """
    text = record["text"]
    if text is None:
        msg = f"doc {record['doc_id']}: content_ref fetch is not supported in the core loop"
        raise WorkerError(msg)
    return text


def _backoff_seconds(consecutive_errors: int) -> float:
    """Return an exponential backoff delay (capped) for a transient-failure streak.

    Args:
        consecutive_errors: Number of consecutive transient failures so far.

    Returns:
        Seconds to wait before the next attempt.
    """
    return float(min(30.0, 0.5 * (2 ** max(0, consecutive_errors - 1))))


def drive_backfill(
    *,
    store: BackfillStore,
    provider: EmbeddingProvider,
    control_plane: ControlPlane,
    budget: BudgetTracker,
    target_model: str,
    target_dims: int,
    batch_size: int,
    usd_per_mtok: float,
    max_consecutive_errors: int = 5,
    pacer: TokenBucket | None = None,
    sleeper: Callable[[float], None] = time.sleep,
) -> MigrationState:
    """Drive the backfill loop to completion, a budget pause, or an error pause.

    The loop is the exactly-once pipeline of DESIGN.md §4. Its correctness rests on two
    ordering rules that hold across any crash point: (1) ``upsert_vectors`` is the only
    durability event and it is idempotent and guarded, so replays are no-ops and stale
    or deleted writes cannot satisfy a row (I1/I3/I4); (2) ``commit_spend`` follows the
    durable upsert, so a crash in between under-counts by at most one batch and never
    double-counts on resume (I5). Budget is reserved *before* the embed, so a ceiling
    hit stops cleanly at a batch boundary with no partial ledger commit.

    Args:
        store: Durable claim/upsert/reconcile surface.
        provider: Embedding provider.
        control_plane: Progress/spend/dead-letter surface.
        budget: Monotone spend tracker with pre-reservation.
        target_model: Target model identifier passed to the provider.
        target_dims: Target embedding dimensionality.
        batch_size: Maximum rows claimed per batch.
        usd_per_mtok: Price used to estimate per-batch USD spend.
        max_consecutive_errors: Circuit-breaker threshold for transient failures.
        pacer: Optional token-bucket pacer (skipped when ``None`` or unthrottled).
        sleeper: Sleep function for backoff/pacing, injectable for tests.

    Returns:
        The terminal state for this invocation: ``BACKFILLED``, ``PAUSED_BUDGET``, or
        ``PAUSED_ERROR``.

    Raises:
        WorkerError: If invoked from a non-resumable state.
    """
    entry = control_plane.get_state()
    if entry not in RESUMABLE_ENTRY:
        msg = f"cannot backfill from state {entry}; expected one of {sorted(RESUMABLE_ENTRY)}"
        raise WorkerError(msg)
    control_plane.set_state("BACKFILLING")

    consecutive_errors = 0
    batches_upserted = 0
    while True:
        batch = store.claim_batch(limit=batch_size)
        if not batch:
            # Completion critical section: reconcile, re-check pending, and write state
            # with no interleaving (I6 -- "insert between reconcile and BACKFILLED").
            store.reconcile()
            if store.pending_count() == 0:
                control_plane.set_state("BACKFILLED")
                return "BACKFILLED"
            # reconcile re-enqueued dirty rows (I4); drain again.
            continue

        texts = [_text_of(record) for record in batch]
        est_tokens = provider.count_tokens(texts, model=target_model)
        est_usd = est_tokens / 1_000_000 * usd_per_mtok

        # Reserve BEFORE spending (I5). A ceiling hit stops at this batch boundary with
        # nothing embedded and nothing committed.
        try:
            budget.reserve(tokens=est_tokens, usd=est_usd)
        except BudgetExhaustedError:
            control_plane.set_state("PAUSED_BUDGET")
            return "PAUSED_BUDGET"

        if pacer is not None:
            pacer.acquire(est_tokens, sleeper=sleeper)

        try:
            matrix = provider.embed_batch(texts, model=target_model, dims=target_dims)
        except PoisonInputError as poison:
            for index in poison.bad_indices:
                record = batch[index]
                control_plane.record_dead_letter(
                    doc_id=record["doc_id"],
                    content_hash=record["content_hash"],
                    reason=poison.reason,
                    error=str(poison),
                )
            consecutive_errors = 0
            continue
        except TransientProviderError:
            consecutive_errors += 1
            if consecutive_errors >= max_consecutive_errors:
                control_plane.set_state("PAUSED_ERROR")
                return "PAUSED_ERROR"
            sleeper(_backoff_seconds(consecutive_errors))
            continue
        except ProviderPausedError:
            control_plane.set_state("PAUSED_ERROR")
            return "PAUSED_ERROR"

        consecutive_errors = 0
        records: list[VectorRecord] = [
            VectorRecord(
                doc_id=record["doc_id"],
                content_hash=record["content_hash"],
                vector=_row(matrix, index),
                provenance="embedded",
            )
            for index, record in enumerate(batch)
        ]

        # Durability point. A hard crash here (kill -9) is NOT caught: it propagates out
        # and resume re-derives work from whatever was durably written.
        store.upsert_vectors(records)
        batches_upserted += 1

        # Crash-injection window (tests only): durable write done, ledger not yet.
        _maybe_inject_fault(batches_upserted, sleeper=sleeper)

        # Ledger commit strictly after durable upsert (I1/I5). The control-plane ledger
        # is the durable source of truth; the BudgetTracker is its in-memory mirror,
        # re-hydrated from the durable ledger on resume. A crash inside commit_spend
        # loses both (the mirror update below never runs), so the two stay consistent.
        control_plane.commit_spend(tokens=est_tokens, usd=est_usd)
        budget.commit(tokens=est_tokens, usd=est_usd)


def _row(matrix: npt.NDArray[np.float32], index: int) -> npt.NDArray[np.float32]:
    """Return a contiguous float32 copy of row ``index`` from an embedding matrix.

    Args:
        matrix: The ``(n, d)`` embedding matrix.
        index: Row index.

    Returns:
        A ``(d,)`` float32 vector.
    """
    return np.ascontiguousarray(matrix[index], dtype=np.float32)


# --- Execution backend factories -------------------------------------------------------
# run() assembles real collaborators from config. The store, provider, and control-plane
# backends are built by later phases (pgvector adapter, providers, control plane); until
# then these factories raise a precise error rather than silently doing nothing. The
# tested core -- drive_backfill above -- has no dependency on them.


def _build_store(config: EmbMigrateConfig) -> BackfillStore:
    """Construct the store adapter for the configured backend."""
    if config.store.kind == "pgvector":
        from alembicio.adapters.pgvector import PgVectorAdapter

        return PgVectorAdapter.from_config(config)
    msg = f"store backend {config.store.kind!r} is not wired for execution yet"
    raise WorkerError(msg)


def _build_provider(config: EmbMigrateConfig) -> EmbeddingProvider:
    """Construct the embedding provider for the configured target."""
    overrides = config.pricing.usd_per_mtok if config.pricing else None
    provider = config.target.provider
    if provider == "openai":
        from alembicio.providers.openai import OpenAIProvider

        return OpenAIProvider(pricing_overrides=overrides)
    if provider == "gemini":
        from alembicio.providers.gemini import GeminiProvider

        return GeminiProvider(pricing_overrides=overrides)
    if provider == "fastembed":
        from alembicio.providers.fastembed_local import FastEmbedLocalProvider

        return FastEmbedLocalProvider(pricing_overrides=overrides)
    msg = f"provider {provider!r} is not wired for execution yet"
    raise WorkerError(msg)


def _build_control_plane(config: EmbMigrateConfig) -> ControlPlane:
    """Construct the control plane for the configured backend."""
    if config.store.kind == "pgvector":
        from alembicio.core.control_plane import PostgresControlPlane

        plane = PostgresControlPlane(
            conninfo=config.store.dsn,
            migration_id=config.migration,
        )
        plane.ensure_schema()
        return plane
    msg = f"control plane for {config.store.kind!r} is not wired for execution yet"
    raise WorkerError(msg)


def run(config: EmbMigrateConfig, *, resume: bool = False) -> None:
    """Execute the backfill worker until paused or complete.

    Exactly-once argument (I1/I2). The only durable done-record is the target companion
    hash, written transactionally with the vector by ``upsert_vectors`` (D10). Work is
    *derived*, never remembered: ``claim_batch`` re-selects rows whose companion hash
    differs from the canonical ``content_hash`` (D11), so replaying an already-satisfied
    key selects nothing and is a no-op. The ledger commits only after that durable
    upsert, so a crash between the two under-counts spend by at most one batch and never
    double-counts on resume; a crash before it leaves the row pending for a clean retry.
    Updates null the companion hash (I4), so a stale in-flight write cannot satisfy a
    dirty row; deletes remove the row structurally (I3), so a late write cannot
    resurrect it. Budget is reserved before spend and committed after, so a ceiling hit
    stops cleanly at a batch boundary in ``PAUSED_BUDGET`` (I5). Hence resume from any
    crash point converges to the same durable set with monotone, non-duplicated spend.

    Args:
        config: Validated migration configuration.
        resume: When True, continue from durable pending state only.

    Returns:
        None.
    """
    store = _build_store(config)
    provider = _build_provider(config)
    control_plane = _build_control_plane(config)

    limits = provider.limits(model=config.target.model)
    budget = BudgetTracker(config.backfill.budget)
    spend = control_plane.spend()
    budget.tokens_in = spend["tokens_in"]
    budget.usd_est = spend["usd_est"]
    pacer: TokenBucket | None = None
    if config.backfill.rate_limit.tpm > 0:
        pacer = TokenBucket(
            capacity=config.backfill.rate_limit.tpm,
            refill_per_sec=config.backfill.rate_limit.tpm / 60.0,
        )

    drive_backfill(
        store=store,
        provider=provider,
        control_plane=control_plane,
        budget=budget,
        target_model=config.target.model,
        target_dims=config.target.dim,
        batch_size=config.backfill.batch_size,
        usd_per_mtok=limits["usd_per_mtok"],
        pacer=pacer,
    )
