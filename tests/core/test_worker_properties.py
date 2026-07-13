"""Property tests for the backfill worker loop (INVARIANTS I1-I6, DESIGN.md §4).

Each property is exercised against the in-memory FakeStore/FakeProvider/FakeControlPlane
under Hypothesis-generated crash and mutation schedules. The fakes model the pgvector
exactly-once machine (companion hash = done-record, structural deletes, dirty-null
updates), so a green property here is a statement about the invariants, not the fakes.
"""

from __future__ import annotations

import contextlib
import string

from hypothesis import assume, given
from hypothesis import strategies as st

from alembicio.config import BudgetConfig
from alembicio.core.budget import BudgetTracker
from alembicio.core.worker import drive_backfill
from tests.core import harness
from tests.core.fakes import (
    CrashInjected,
    FakeControlPlane,
    FakeProvider,
    FakeStore,
    GlobalSchedule,
    Session,
    content_hash,
)

_TEXT = st.text(alphabet=string.ascii_letters + string.digits + " ", max_size=40)
_BATCH = st.integers(min_value=1, max_value=8)
_CRASHES = st.lists(st.integers(min_value=1, max_value=150), max_size=6)


def _corpus(*, min_size: int = 1, max_size: int = 20) -> st.SearchStrategy[dict[str, str]]:
    """Strategy for a ``doc_id -> text`` corpus with unique ids."""
    return st.lists(_TEXT, min_size=min_size, max_size=max_size).map(
        lambda texts: {f"doc_{i}": text for i, text in enumerate(texts)}
    )


def _assert_all_rows_done(store: FakeStore) -> None:
    """At BACKFILLED every present row is done from its current hash, exactly once (I1)."""
    for row in store.rows.values():
        assert row.stored_hash == content_hash(row.text)
        assert row.provenance == "embedded"
    assert all(count == 1 for count in store.satisfy_events.values())


# --- (a) exactly-once effect set under any crash schedule + dual-write arrivals --------


@given(_corpus(), _CRASHES, _BATCH, st.lists(st.tuples(st.integers(1, 60), _TEXT), max_size=4))
def test_crash_schedule_preserves_exactly_once_set(
    docs: dict[str, str],
    crash_steps: list[int],
    batch_size: int,
    inserts: list[tuple[int, str]],
) -> None:
    """(a) After resume the satisfied set is exactly pending-at-start plus dual-write
    arrivals, with no duplicate store effects."""
    scenario = harness.new_scenario(docs)
    mutations = [
        (step, ("insert", f"ins_{j}", text)) for j, (step, text) in enumerate(inserts)
    ]
    harness.run(scenario, batch_size=batch_size, crash_steps=crash_steps, mutations=mutations)
    store = scenario.store

    assert scenario.cp.get_state() == "BACKFILLED"
    assert store.pending_count() == 0
    _assert_all_rows_done(store)
    # Every original document survived and is satisfied from its current content.
    for doc_id, text in docs.items():
        assert store.rows[doc_id].stored_hash == content_hash(text)
    # Exactly the dual-write arrivals that actually fired are present as extra rows.
    inserted_present = [key for key in store.rows if key.startswith("ins_")]
    assert len(inserted_present) == scenario.schedule.mutations_fired


# --- (b) ledger monotone across any schedule; never double-counts a replayed batch -----


@given(_corpus(min_size=1, max_size=15), _CRASHES, _BATCH)
def test_ledger_monotone_and_never_double_counts(
    docs: dict[str, str], crash_steps: list[int], batch_size: int
) -> None:
    """(b) Ledger counters only ever increase across a crash/resume schedule, never
    exceed the no-crash baseline, and no key is counted twice."""
    session = Session()
    dead: set[tuple[str, str]] = set()
    store = FakeStore(docs, dead_letters=dead, session=session)
    cp = FakeControlPlane(session, dead_letters=dead)
    provider = FakeProvider(dim=8)
    session.controller = GlobalSchedule(crash_steps=crash_steps)

    trace = [cp.tokens_in]
    for _ in range(len(crash_steps) + len(docs) + 8):
        if cp.get_state() == "BACKFILLED":
            break
        budget = BudgetTracker(BudgetConfig(max_usd=1e9, max_tokens=10**12))
        budget.tokens_in = cp.tokens_in
        budget.usd_est = cp.usd_est
        with contextlib.suppress(CrashInjected):
            drive_backfill(
                store=store,
                provider=provider,
                control_plane=cp,
                budget=budget,
                target_model="fake",
                target_dims=8,
                batch_size=batch_size,
                usd_per_mtok=0.0,
                sleeper=lambda _s: None,
            )
        trace.append(cp.tokens_in)

    assert cp.get_state() == "BACKFILLED"
    # Monotone non-decreasing across the whole schedule (I5).
    assert all(later >= earlier for earlier, later in zip(trace, trace[1:], strict=False))
    # Never exceeds the additive baseline: each key counted at most once (I1).
    baseline = sum(provider.count_tokens([text], model="fake") for text in docs.values())
    assert cp.tokens_in <= baseline
    assert all(count == 1 for count in store.satisfy_events.values())


# --- (c) a delete injected at any point never resurrects (I3) --------------------------


@given(_corpus(min_size=2, max_size=15), _BATCH, st.data())
def test_delete_never_resurrects(
    docs: dict[str, str], batch_size: int, data: st.DataObject
) -> None:
    """(c) A document deleted mid-run is gone and is never satisfied, while every
    surviving row still completes."""
    victim = data.draw(st.sampled_from(sorted(docs)))
    step = data.draw(st.integers(min_value=1, max_value=2 * len(docs) + 3))
    scenario = harness.new_scenario(docs)
    harness.run(scenario, batch_size=batch_size, mutations=[(step, ("delete", victim))])
    assume(scenario.schedule.all_mutations_fired)

    store = scenario.store
    assert scenario.cp.get_state() == "BACKFILLED"
    assert victim not in store.rows
    assert all(key[0] != victim for key in store.satisfied_keys())
    _assert_all_rows_done(store)


# --- (d) an update injected at any point leaves the row pending until the NEW hash (I4) -


@given(_corpus(min_size=2, max_size=15), _TEXT, _BATCH, st.data())
def test_update_pending_until_embedded_from_new_hash(
    docs: dict[str, str], new_text: str, batch_size: int, data: st.DataObject
) -> None:
    """(d) A document updated mid-run is done only once embedded from its new content;
    a stale in-flight write can never satisfy the dirty row."""
    victim = data.draw(st.sampled_from(sorted(docs)))
    assume(content_hash(new_text) != content_hash(docs[victim]))
    step = data.draw(st.integers(min_value=1, max_value=2 * len(docs) + 3))
    scenario = harness.new_scenario(docs)
    harness.run(
        scenario,
        batch_size=batch_size,
        mutations=[(step, ("update", victim, new_text))],
    )
    assume(scenario.schedule.all_mutations_fired)

    store = scenario.store
    assert scenario.cp.get_state() == "BACKFILLED"
    row = store.rows[victim]
    assert row.stored_hash == content_hash(new_text)
    assert row.provenance == "embedded"
    assert store.satisfy_events[(victim, content_hash(new_text))] == 1
    _assert_all_rows_done(store)


# --- (e) budget exhaustion lands in PAUSED_BUDGET at a batch boundary, clean resume ----


@given(_corpus(min_size=3, max_size=15), _BATCH)
def test_budget_exhaustion_pauses_then_resumes(
    docs: dict[str, str], batch_size: int
) -> None:
    """(e) A token ceiling below the total stops cleanly in PAUSED_BUDGET with no partial
    commit and full resumability once the ceiling is raised."""
    provider = FakeProvider(dim=8)
    total = sum(provider.count_tokens([text], model="fake") for text in docs.values())
    cap = max(1, total // 2)

    scenario = harness.new_scenario(docs, provider=provider)
    harness.run(scenario, batch_size=batch_size, max_tokens=cap)
    cp, store = scenario.cp, scenario.store

    assert cp.get_state() == "PAUSED_BUDGET"
    assert cp.tokens_in <= cap  # ceiling enforced before spend (I5)
    assert store.pending_count() > 0  # clean stop, work remains
    assert all(count == 1 for count in store.satisfy_events.values())

    state = harness.drain(scenario, batch_size=batch_size, max_tokens=10**12)
    assert state == "BACKFILLED"
    _assert_all_rows_done(store)
    for doc_id, text in docs.items():
        assert store.rows[doc_id].stored_hash == content_hash(text)


# --- poison + transient handling (dead-letter idempotency, circuit breaker) ------------


def test_poison_is_dead_lettered_exactly_once() -> None:
    """A poison document is dead-lettered once, the run completes, and neighbours in its
    batch still finish."""
    docs = {f"doc_{i}": f"text-{i}" for i in range(10)}
    poison_text = docs["doc_3"]
    provider = FakeProvider(dim=8, poison_texts={poison_text}, poison_reason="token_limit")
    scenario = harness.new_scenario(docs, provider=provider)
    harness.run(scenario, batch_size=4)

    cp, store = scenario.cp, scenario.store
    key = ("doc_3", content_hash(poison_text))
    assert cp.get_state() == "BACKFILLED"
    assert key in cp.dead_letters
    assert cp.dead_letter_attempts[key] == 1
    assert cp.dead_letter_reason[key] == "token_limit"
    assert key not in store.satisfied_keys()
    for i in range(10):
        if i == 3:
            continue
        assert store.rows[f"doc_{i}"].stored_hash == content_hash(docs[f"doc_{i}"])


@given(_CRASHES)
def test_poison_not_duplicated_across_crashes(crash_steps: list[int]) -> None:
    """Replays after crashes never re-dead-letter an already-poisoned key."""
    docs = {f"doc_{i}": f"text-{i}" for i in range(8)}
    poison_text = docs["doc_2"]
    provider = FakeProvider(dim=8, poison_texts={poison_text}, poison_reason="empty")
    scenario = harness.new_scenario(docs, provider=provider)
    harness.run(scenario, batch_size=3, crash_steps=crash_steps)

    cp = scenario.cp
    key = ("doc_2", content_hash(poison_text))
    assert cp.get_state() == "BACKFILLED"
    assert cp.dead_letter_attempts[key] == 1


def test_transient_storm_pauses_error_then_resumes() -> None:
    """A run of transient provider failures trips the circuit breaker to PAUSED_ERROR and
    resumes cleanly with a healthy provider."""
    docs = {f"doc_{i}": f"text-{i}" for i in range(5)}
    provider = FakeProvider(dim=8, fail_first=5)
    scenario = harness.new_scenario(docs, provider=provider)
    harness.run(scenario, batch_size=4)

    assert scenario.cp.get_state() == "PAUSED_ERROR"

    state = harness.drain(scenario, batch_size=4, provider=FakeProvider(dim=8))
    assert state == "BACKFILLED"
    for doc_id, text in docs.items():
        assert scenario.store.rows[doc_id].stored_hash == content_hash(text)


def test_clean_run_backfills_everything() -> None:
    """Baseline: with no faults every row is embedded exactly once and state is BACKFILLED."""
    docs = {f"doc_{i}": f"body number {i}" for i in range(20)}
    scenario = harness.new_scenario(docs)
    harness.run(scenario, batch_size=6)
    assert scenario.cp.get_state() == "BACKFILLED"
    _assert_all_rows_done(scenario.store)
    assert scenario.store.pending_count() == 0
