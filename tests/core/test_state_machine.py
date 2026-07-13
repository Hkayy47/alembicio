"""Property tests for the state machine (INVARIANTS I6, DESIGN.md §2).

Property (f): the worker never drives a transition outside the legal table, and the
table itself matches the design.
"""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, precondition, rule

from alembicio.core.state import (
    LEGAL_TRANSITIONS,
    RESUMABLE_ENTRY,
    TERMINAL_STATES,
    MigrationState,
    TransitionError,
    assert_transition,
    is_legal,
)
from tests.core import harness

ALL_STATES: list[MigrationState] = list(LEGAL_TRANSITIONS.keys())


def test_table_covers_every_state() -> None:
    """Every declared state appears as a key with a (possibly identity-only) entry."""
    assert set(ALL_STATES) == {
        "CREATED",
        "PREPARED",
        "BACKFILLING",
        "BACKFILLED",
        "VERIFIED",
        "CANARY",
        "CUTOVER",
        "SOAKING",
        "DONE",
        "ROLLED_BACK",
        "PAUSED_BUDGET",
        "PAUSED_ERROR",
    }


def test_spot_check_design_transitions() -> None:
    """A few load-bearing transitions from DESIGN.md §2 / I6."""
    assert is_legal("PREPARED", "BACKFILLING")
    assert is_legal("BACKFILLING", "BACKFILLED")
    assert is_legal("BACKFILLING", "PAUSED_BUDGET")
    assert is_legal("PAUSED_BUDGET", "BACKFILLING")
    assert is_legal("BACKFILLED", "VERIFIED")
    assert is_legal("VERIFIED", "CANARY")
    assert is_legal("SOAKING", "DONE")
    # Rollback is legal exactly from CANARY, CUTOVER, SOAKING (I6).
    for state in ("CANARY", "CUTOVER", "SOAKING"):
        assert is_legal(state, "ROLLED_BACK")
    for state in ("BACKFILLED", "VERIFIED", "BACKFILLING"):
        assert not is_legal(state, "ROLLED_BACK")


def test_illegal_transitions_raise() -> None:
    """Skipping states or moving backward is rejected."""
    for frm, to in [
        ("CREATED", "DONE"),
        ("CREATED", "BACKFILLING"),
        ("PREPARED", "VERIFIED"),
        ("BACKFILLED", "CANARY"),
        ("DONE", "SOAKING"),
        ("ROLLED_BACK", "BACKFILLING"),
    ]:
        assert not is_legal(frm, to)
        with pytest.raises(TransitionError):
            assert_transition(frm, to)


def test_identity_transitions_are_legal() -> None:
    """Idempotent verb re-runs (I6) express as legal self-transitions."""
    for state in ALL_STATES:
        assert is_legal(state, state)
        assert_transition(state, state)


def test_resumable_entry_and_terminal_sets() -> None:
    """The worker's entry set and terminal set match the design."""
    assert {"PREPARED", "BACKFILLING", "PAUSED_BUDGET", "PAUSED_ERROR"} == RESUMABLE_ENTRY
    assert {"DONE", "ROLLED_BACK"} == TERMINAL_STATES
    for terminal in TERMINAL_STATES:
        assert LEGAL_TRANSITIONS[terminal] == {terminal}


@given(st.sampled_from(ALL_STATES), st.sampled_from(ALL_STATES))
def test_is_legal_and_assert_agree(frm: MigrationState, to: MigrationState) -> None:
    """``is_legal`` and ``assert_transition`` never disagree for any state pair."""
    if to in LEGAL_TRANSITIONS[frm]:
        assert is_legal(frm, to)
        assert_transition(frm, to)  # must not raise
    else:
        assert not is_legal(frm, to)
        with pytest.raises(TransitionError):
            assert_transition(frm, to)


@given(harness_crashes=st.lists(st.integers(1, 120), max_size=6), batch_size=st.integers(1, 6))
def test_worker_state_history_is_a_legal_path(
    harness_crashes: list[int], batch_size: int
) -> None:
    """(f) Every consecutive pair the worker writes is a legal transition, under crashes."""
    docs = {f"doc_{i}": f"text {i}" for i in range(12)}
    scenario = harness.new_scenario(docs)
    harness.run(scenario, batch_size=batch_size, crash_steps=harness_crashes)
    history = scenario.cp.state_history
    assert history[0] == "PREPARED"
    assert history[-1] == "BACKFILLED"
    for earlier, later in zip(history, history[1:], strict=False):
        assert is_legal(earlier, later), f"illegal {earlier} -> {later}"


class LegalWalk(RuleBasedStateMachine):
    """A random walk that only ever takes legal steps must never raise."""

    def __init__(self) -> None:
        super().__init__()
        self.state: MigrationState = "CREATED"

    @precondition(lambda self: True)
    @rule(data=st.data())
    def step(self, data: st.DataObject) -> None:
        targets = sorted(LEGAL_TRANSITIONS[self.state])
        target = data.draw(st.sampled_from(targets))
        assert_transition(self.state, target)  # must not raise
        self.state = target
        assert self.state in LEGAL_TRANSITIONS


TestLegalWalk = LegalWalk.TestCase
