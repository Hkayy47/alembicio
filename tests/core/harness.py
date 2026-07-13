"""Scenario driver: run the worker over a crash/mutation schedule to convergence.

A scenario reuses one :class:`~tests.core.fakes.GlobalSchedule` across resume rounds so
crash points and concurrent mutations share a single global timeline. Each round
re-hydrates the :class:`~alembicio.core.budget.BudgetTracker` from the durable ledger,
exactly as a real resume would.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from alembicio.config import BudgetConfig
from alembicio.core.budget import BudgetTracker
from alembicio.core.state import MigrationState
from alembicio.core.worker import drive_backfill
from tests.core.fakes import (
    CrashInjected,
    FakeControlPlane,
    FakeProvider,
    FakeStore,
    GlobalSchedule,
    Session,
)

# A mutation spec is a (global_step, op-tuple) pair, where op-tuple is one of:
#   ("insert", doc_id, text) | ("update", doc_id, text) | ("delete", doc_id)
MutationSpec = tuple[int, tuple[str, ...]]

_BIG_USD = 1e9
_BIG_TOKENS = 10**12


@dataclass
class ScenarioResult:
    """Everything a property test needs to assert on after a scenario runs."""

    session: Session
    store: FakeStore
    cp: FakeControlPlane
    provider: FakeProvider
    schedule: GlobalSchedule
    final_state: MigrationState | None


def _bind_mutations(
    store: FakeStore, mutations: Sequence[MutationSpec] | None
) -> list[tuple[int, Callable[[], None]]]:
    """Turn declarative mutation specs into callables bound to ``store``.

    Args:
        store: The store the mutations act on.
        mutations: Declarative ``(step, op)`` specs.

    Returns:
        A list of ``(step, thunk)`` pairs for :class:`GlobalSchedule`.
    """
    bound: list[tuple[int, Callable[[], None]]] = []
    for step, op in mutations or []:
        kind = op[0]
        if kind == "insert":
            bound.append((step, lambda o=op: store.insert(o[1], o[2])))
        elif kind == "update":
            bound.append((step, lambda o=op: store.update(o[1], o[2])))
        elif kind == "delete":
            bound.append((step, lambda o=op: store.delete(o[1])))
        else:  # pragma: no cover - guards a typo in a test spec
            raise ValueError(f"unknown mutation kind {kind!r}")
    return bound


def _rehydrated_budget(
    cp: FakeControlPlane, *, max_usd: float, max_tokens: int
) -> BudgetTracker:
    """Build a BudgetTracker seeded from the durable ledger (models resume)."""
    budget = BudgetTracker(BudgetConfig(max_usd=max_usd, max_tokens=max_tokens))
    budget.tokens_in = cp.tokens_in
    budget.usd_est = cp.usd_est
    return budget


def new_scenario(
    docs: dict[str, str],
    *,
    dim: int = 8,
    provider: FakeProvider | None = None,
    initial: MigrationState = "PREPARED",
) -> ScenarioResult:
    """Create a fresh scenario over ``docs`` with an inert (no-op) schedule.

    Args:
        docs: Initial ``doc_id -> text`` corpus.
        dim: Embedding dimensionality.
        provider: Optional pre-configured provider (poison/transient faults).
        initial: Starting migration state.

    Returns:
        A :class:`ScenarioResult` with a not-yet-run store/control-plane pair.
    """
    session = Session()
    dead: set[tuple[str, str]] = set()
    store = FakeStore(docs, dead_letters=dead, session=session)
    cp = FakeControlPlane(session, dead_letters=dead, initial=initial)
    provider = provider or FakeProvider(dim=dim)
    return ScenarioResult(
        session=session,
        store=store,
        cp=cp,
        provider=provider,
        schedule=GlobalSchedule(),
        final_state=None,
    )


def run(
    scenario: ScenarioResult,
    *,
    batch_size: int,
    dim: int = 8,
    crash_steps: Sequence[int] | None = None,
    mutations: Sequence[MutationSpec] | None = None,
    max_usd: float = _BIG_USD,
    max_tokens: int = _BIG_TOKENS,
    usd_per_mtok: float = 0.0,
    max_rounds: int | None = None,
) -> ScenarioResult:
    """Run the worker to convergence under a crash/mutation schedule.

    Rounds repeat until the migration reaches ``BACKFILLED`` or a pause
    (``PAUSED_BUDGET``/``PAUSED_ERROR``). A crash resumes; a pause returns for the caller
    to handle (e.g. raise the budget and :func:`drain`).

    Args:
        scenario: A scenario from :func:`new_scenario`.
        batch_size: Rows per claim.
        dim: Embedding dimensionality.
        crash_steps: Global steps at which to inject a ``kill -9``.
        mutations: Declarative concurrent mutations.
        max_usd: USD ceiling.
        max_tokens: Token ceiling.
        usd_per_mtok: Price used for per-batch USD estimation.
        max_rounds: Safety bound on resume rounds.

    Returns:
        The same :class:`ScenarioResult`, updated with the final state.
    """
    schedule = GlobalSchedule(
        crash_steps=crash_steps,
        mutations=_bind_mutations(scenario.store, mutations),
    )
    scenario.session.controller = schedule
    scenario.schedule = schedule
    rounds = max_rounds or (len(list(crash_steps or [])) + len(scenario.store.rows) + 8)

    final: MigrationState | None = None
    for _ in range(rounds):
        if scenario.cp.get_state() == "BACKFILLED":
            break
        budget = _rehydrated_budget(scenario.cp, max_usd=max_usd, max_tokens=max_tokens)
        try:
            final = drive_backfill(
                store=scenario.store,
                provider=scenario.provider,
                control_plane=scenario.cp,
                budget=budget,
                target_model="fake-target",
                target_dims=dim,
                batch_size=batch_size,
                usd_per_mtok=usd_per_mtok,
                sleeper=lambda _seconds: None,
            )
        except CrashInjected:
            final = None
            continue
        if final in ("BACKFILLED", "PAUSED_BUDGET", "PAUSED_ERROR"):
            break

    scenario.final_state = final if final is not None else scenario.cp.get_state()
    return scenario


def drain(
    scenario: ScenarioResult,
    *,
    batch_size: int,
    dim: int = 8,
    provider: FakeProvider | None = None,
    max_usd: float = _BIG_USD,
    max_tokens: int = _BIG_TOKENS,
    usd_per_mtok: float = 0.0,
    max_rounds: int | None = None,
) -> MigrationState:
    """Resume with no faults (fresh schedule) until BACKFILLED or a stable pause.

    Args:
        scenario: A previously-run scenario (possibly paused).
        batch_size: Rows per claim.
        dim: Embedding dimensionality.
        provider: Optional replacement provider (e.g. one that no longer fails).
        max_usd: USD ceiling (raise it to clear a budget pause).
        max_tokens: Token ceiling.
        usd_per_mtok: Price used for per-batch USD estimation.
        max_rounds: Safety bound on resume rounds.

    Returns:
        The final migration state.
    """
    scenario.session.controller = GlobalSchedule()
    if provider is not None:
        scenario.provider = provider
    rounds = max_rounds or (len(scenario.store.rows) + 8)
    for _ in range(rounds):
        if scenario.cp.get_state() == "BACKFILLED":
            break
        budget = _rehydrated_budget(scenario.cp, max_usd=max_usd, max_tokens=max_tokens)
        state = drive_backfill(
            store=scenario.store,
            provider=scenario.provider,
            control_plane=scenario.cp,
            budget=budget,
            target_model="fake-target",
            target_dims=dim,
            batch_size=batch_size,
            usd_per_mtok=usd_per_mtok,
            sleeper=lambda _seconds: None,
        )
        if state == "BACKFILLED":
            return "BACKFILLED"
        if state in ("PAUSED_BUDGET", "PAUSED_ERROR"):
            return state
    scenario.final_state = scenario.cp.get_state()
    return scenario.cp.get_state()
