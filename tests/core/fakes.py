"""In-memory FakeStore/FakeProvider/FakeControlPlane for worker property tests.

These fakes model the pgvector exactly-once machine faithfully enough to exercise
INVARIANTS I1-I6 under arbitrary crash/mutation schedules:

* The **companion hash** is the done-record (D10): a row is *done* iff its stored hash
  equals ``md5(current text)``. There is no separate done-set; replays are no-ops.
* **Deletes are structural** (I3): deleting a row removes it and its vector together, so
  a late write for that id finds no row and cannot resurrect it.
* **Updates null the done-record** (I4): changing a row's text changes ``md5(text)``, so
  a stale in-flight write (carrying the old hash) fails the upsert guard.
* **Durability is per-item**: a crash injected mid-upsert leaves the items applied so far
  durable and the rest pending -- exactly what resume must tolerate.

Crashes and concurrent mutations are driven by a shared :class:`CrashController` whose
``tick`` is called at every durability-relevant step, so a test can land a ``kill -9`` or
a delete/update/insert at any step index.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Callable, Sequence

import numpy as np
import numpy.typing as npt

from alembicio.adapters.base import DocRecord, ReconcileReport, VectorRecord
from alembicio.core.state import MigrationState, assert_transition
from alembicio.providers.base import (
    PoisonInputError,
    RateLimits,
    TransientProviderError,
)


def content_hash(text: str) -> str:
    """Return the canonical content hash for text (mirrors ``md5({text}::text)``)."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


class CrashInjected(Exception):
    """Simulated ``kill -9``: propagates uncaught out of the worker loop."""

    def __init__(self, tick: int) -> None:
        self.tick = tick
        super().__init__(f"crash injected at tick {tick}")


class GlobalSchedule:
    """A single global step timeline shared by the store and control plane.

    ``tick`` is called at each durability-relevant step across *every* resume round (the
    same schedule object is reused, so its counter is monotonic over the whole scenario).
    At each step it applies any due one-shot mutations (delete/update/insert interleaved
    at a precise moment) and then raises :class:`CrashInjected` when a crash step is
    reached. Mutations at the same step as a crash are applied *before* the crash.
    """

    def __init__(
        self,
        *,
        crash_steps: Sequence[int] | None = None,
        mutations: Sequence[tuple[int, Callable[[], None]]] | None = None,
    ) -> None:
        self.n = 0
        self._crash_steps = sorted(crash_steps or [])
        self._crash_idx = 0
        self._mutations = sorted(mutations or [], key=lambda item: item[0])
        self._mut_idx = 0
        self.mutations_fired = 0
        self.crashes_fired = 0

    def tick(self) -> None:
        """Advance one global step; apply due mutations, then maybe crash."""
        self.n += 1
        while (
            self._mut_idx < len(self._mutations)
            and self.n >= self._mutations[self._mut_idx][0]
        ):
            self._mutations[self._mut_idx][1]()
            self._mut_idx += 1
            self.mutations_fired += 1
        while (
            self._crash_idx < len(self._crash_steps)
            and self.n >= self._crash_steps[self._crash_idx]
        ):
            self._crash_idx += 1
            self.crashes_fired += 1
            raise CrashInjected(self.n)

    @property
    def all_mutations_fired(self) -> bool:
        """True once every scheduled mutation has been applied."""
        return self._mut_idx >= len(self._mutations)


class Session:
    """Holder for the active :class:`GlobalSchedule` (durable fakes outlive rounds)."""

    def __init__(self) -> None:
        self.controller = GlobalSchedule()


class _Row:
    """A single document row with its target companion-hash done-record."""

    __slots__ = ("doc_id", "text", "stored_hash", "vector", "provenance")

    def __init__(self, doc_id: str, text: str) -> None:
        self.doc_id = doc_id
        self.text = text
        self.stored_hash: str | None = None
        self.vector: npt.NDArray[np.float32] | None = None
        self.provenance: str | None = None


class FakeStore:
    """In-memory dual-column store modelling the pgvector pending predicate."""

    def __init__(
        self,
        docs: dict[str, str],
        *,
        dead_letters: set[tuple[str, str]],
        session: Session,
    ) -> None:
        self.rows: dict[str, _Row] = {
            doc_id: _Row(doc_id, text) for doc_id, text in docs.items()
        }
        self.dead_letters = dead_letters
        self.session = session
        # Count of genuine pending->done transitions per key: the exactly-once ledger of
        # store *effects*. A replayed batch produces zero increments (I1).
        self.satisfy_events: Counter[tuple[str, str]] = Counter()

    # --- mutation helpers a test drives via CrashController.on_tick -------------------

    def insert(self, doc_id: str, text: str) -> None:
        """Simulate a dual-write insert arriving mid-run."""
        self.rows[doc_id] = _Row(doc_id, text)

    def update(self, doc_id: str, text: str) -> None:
        """Simulate a document update: new text -> new hash -> row goes dirty (I4)."""
        row = self.rows.get(doc_id)
        if row is not None:
            row.text = text

    def delete(self, doc_id: str) -> None:
        """Simulate a delete: row and vector vanish together (I3, structural)."""
        self.rows.pop(doc_id, None)

    # --- pending predicate -----------------------------------------------------------

    def _is_pending(self, row: _Row) -> bool:
        current = content_hash(row.text)
        if (row.doc_id, current) in self.dead_letters:
            return False
        return row.stored_hash != current

    def satisfied_keys(self) -> set[tuple[str, str]]:
        """Return every ``(doc_id, stored_hash)`` currently marked done."""
        return {
            (row.doc_id, row.stored_hash)
            for row in self.rows.values()
            if row.stored_hash is not None
            and row.stored_hash == content_hash(row.text)
        }

    # --- StoreAdapter subset the worker uses -----------------------------------------

    def pending_count(self) -> int:
        """Return rows requiring a fresh embed, excluding dead-lettered keys."""
        return sum(1 for row in self.rows.values() if self._is_pending(row))

    def claim_batch(self, *, limit: int) -> list[DocRecord]:
        """Claim up to ``limit`` pending, non-dead-lettered rows (insertion order)."""
        self.session.controller.tick()
        claimed: list[DocRecord] = []
        for row in self.rows.values():
            if len(claimed) >= limit:
                break
            if self._is_pending(row):
                claimed.append(
                    DocRecord(
                        doc_id=row.doc_id,
                        text=row.text,
                        content_ref=None,
                        content_hash=content_hash(row.text),
                    )
                )
        return claimed

    def upsert_vectors(self, batch: list[VectorRecord]) -> None:
        """Idempotently persist vectors with the delete/dirty guards (I1/I3/I4)."""
        for record in batch:
            # tick() BEFORE applying, so a crash leaves prior records durable and this
            # one (and the rest) not.
            self.session.controller.tick()
            row = self.rows.get(record["doc_id"])
            if row is None:
                continue  # deleted -> cannot resurrect (I3)
            current = content_hash(row.text)
            if record["content_hash"] != current:
                continue  # stale write cannot satisfy a dirty row (I4)
            was_pending = row.stored_hash != current
            row.stored_hash = record["content_hash"]
            row.vector = record["vector"]
            row.provenance = record["provenance"]
            if was_pending:
                self.satisfy_events[(record["doc_id"], record["content_hash"])] += 1

    def reconcile(self) -> ReconcileReport:
        """Dual-column deletes are structural and dirty rows are already pending; no-op."""
        self.session.controller.tick()
        return ReconcileReport(
            tombstones_applied=0, dirty_requeued=0, orphans_removed=0
        )


class FakeControlPlane:
    """In-memory control plane: state machine, monotone ledger, dead-letter book."""

    def __init__(
        self,
        session: Session,
        *,
        dead_letters: set[tuple[str, str]],
        initial: MigrationState = "PREPARED",
    ) -> None:
        self.session = session
        self._state: MigrationState = initial
        self.tokens_in: int = 0
        self.usd_est: float = 0.0
        self.dead_letters = dead_letters
        self.dead_letter_attempts: Counter[tuple[str, str]] = Counter()
        self.dead_letter_reason: dict[tuple[str, str], str] = {}
        self.state_history: list[MigrationState] = [initial]

    def get_state(self) -> MigrationState:
        return self._state

    def set_state(self, state: MigrationState, /) -> None:
        assert_transition(self._state, state)  # I6: illegal moves raise loudly
        self._state = state
        self.state_history.append(state)

    def commit_spend(self, *, tokens: int, usd: float) -> None:
        # tick() models a crash after the durable upsert but before the ledger row is
        # committed: the increment below never runs, so the ledger under-counts by this
        # batch and never double-counts on resume.
        self.session.controller.tick()
        self.tokens_in += tokens
        self.usd_est += usd

    def record_dead_letter(
        self, *, doc_id: str, content_hash: str, reason: str, error: str
    ) -> None:
        key = (doc_id, content_hash)
        self.dead_letters.add(key)  # PK-deduped set: never a duplicate key
        self.dead_letter_attempts[key] += 1
        self.dead_letter_reason[key] = reason

    def spend(self) -> dict[str, float]:
        return {"tokens_in": self.tokens_in, "usd_est": self.usd_est}


class FakeProvider:
    """Deterministic embedding provider with poison/transient fault injection."""

    def __init__(
        self,
        *,
        dim: int,
        usd_per_mtok: float = 0.0,
        poison_texts: set[str] | None = None,
        poison_reason: str = "token_limit",
        fail_first: int = 0,
    ) -> None:
        self.dim = dim
        self.usd_per_mtok = usd_per_mtok
        self.poison_texts = poison_texts or set()
        self.poison_reason = poison_reason
        self._fail_remaining = fail_first
        self.embed_calls = 0

    def embed_batch(
        self,
        texts: Sequence[str],
        *,
        model: str,
        dims: int | None = None,
    ) -> npt.NDArray[np.float32]:
        if self._fail_remaining > 0:
            self._fail_remaining -= 1
            raise TransientProviderError("simulated 429/5xx")
        bad = [i for i, text in enumerate(texts) if text in self.poison_texts]
        if bad:
            raise PoisonInputError(bad, reason=self.poison_reason)
        self.embed_calls += 1
        width = dims if dims is not None else self.dim
        return np.zeros((len(texts), width), dtype=np.float32)

    def count_tokens(self, texts: Sequence[str], *, model: str) -> int:
        return sum(max(1, math.ceil(len(text) / 4)) for text in texts)

    def limits(self, *, model: str) -> RateLimits:
        return RateLimits(
            tpm=0, rpm=0, max_input_tokens=100_000, usd_per_mtok=self.usd_per_mtok
        )
