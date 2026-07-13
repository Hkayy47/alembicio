"""Migration state machine: legal transitions and validation (DESIGN.md §2, I6).

This module is pure data plus validation. Persistence of the current state is a
control-plane concern (Postgres ``alembicio.migration`` row or the SQLite mirror);
this module only decides whether a proposed transition is one the law permits.
"""

from __future__ import annotations

from typing import Literal

MigrationState = Literal[
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
]

# The complete legal transition table from DESIGN.md §2, cross-checked against I6.
# Each state maps to the set of states it may move to. The identity transition
# (state -> itself) is legal everywhere so that idempotent verb re-runs (``prepare``
# twice, ``doctor`` on CREATED, a no-op ``backfill`` that finds nothing pending) are a
# no-op rather than an error, exactly as I6 requires.
LEGAL_TRANSITIONS: dict[MigrationState, frozenset[MigrationState]] = {
    "CREATED": frozenset({"CREATED", "PREPARED"}),
    "PREPARED": frozenset({"PREPARED", "BACKFILLING"}),
    "BACKFILLING": frozenset(
        {"BACKFILLING", "BACKFILLED", "PAUSED_BUDGET", "PAUSED_ERROR"}
    ),
    "PAUSED_BUDGET": frozenset({"PAUSED_BUDGET", "BACKFILLING"}),
    "PAUSED_ERROR": frozenset({"PAUSED_ERROR", "BACKFILLING"}),
    "BACKFILLED": frozenset({"BACKFILLED", "VERIFIED"}),
    "VERIFIED": frozenset({"VERIFIED", "CANARY", "CUTOVER"}),
    "CANARY": frozenset({"CANARY", "CUTOVER", "SOAKING", "ROLLED_BACK"}),
    "CUTOVER": frozenset({"CUTOVER", "SOAKING", "ROLLED_BACK"}),
    "SOAKING": frozenset({"SOAKING", "DONE", "ROLLED_BACK"}),
    "DONE": frozenset({"DONE"}),
    "ROLLED_BACK": frozenset({"ROLLED_BACK"}),
}

# States a backfill worker may legally start (or resume) from.
RESUMABLE_ENTRY: frozenset[MigrationState] = frozenset(
    {"PREPARED", "BACKFILLING", "PAUSED_BUDGET", "PAUSED_ERROR"}
)

# Terminal states from which no further transition (other than identity) is legal.
TERMINAL_STATES: frozenset[MigrationState] = frozenset({"DONE", "ROLLED_BACK"})


class TransitionError(ValueError):
    """Raised when a proposed state transition is not in DESIGN.md §2."""


def is_legal(frm: MigrationState, to: MigrationState) -> bool:
    """Return whether moving from ``frm`` to ``to`` is a legal transition.

    Args:
        frm: The current migration state.
        to: The proposed next state.

    Returns:
        ``True`` iff the transition (including the identity transition) is permitted.
    """
    return to in LEGAL_TRANSITIONS.get(frm, frozenset())


def assert_transition(frm: MigrationState, to: MigrationState) -> None:
    """Validate a transition, raising :class:`TransitionError` if it is illegal.

    Args:
        frm: The current migration state.
        to: The proposed next state.

    Returns:
        None.
    """
    if not is_legal(frm, to):
        legal = ", ".join(sorted(LEGAL_TRANSITIONS.get(frm, frozenset()))) or "(none)"
        msg = f"illegal transition {frm} -> {to}; legal targets from {frm}: {legal}"
        raise TransitionError(msg)
