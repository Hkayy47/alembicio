"""Provider unit tests with respx HTTP fixtures (P5, D12)."""

from __future__ import annotations

import httpx
import numpy as np
import pytest
import respx

from alembicio.config import BudgetConfig
from alembicio.core.budget import BudgetExhaustedError, BudgetTracker
from alembicio.core.worker import drive_backfill
from alembicio.providers.base import ProviderPausedError
from alembicio.providers.http_util import RetryState
from alembicio.providers.openai import _OPENAI_EMBED_URL, OpenAIProvider
from tests.core.fakes import FakeControlPlane, FakeStore, Session


@respx.mock
def test_openai_embed_batch_success() -> None:
    route = respx.post(_OPENAI_EMBED_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [
                    {"index": 0, "embedding": [0.1, 0.2, 0.3]},
                    {"index": 1, "embedding": [0.4, 0.5, 0.6]},
                ]
            },
        )
    )
    provider = OpenAIProvider(api_key="test-key", client=httpx.Client())
    matrix = provider.embed_batch(["a", "b"], model="text-embedding-3-small")
    assert route.called
    assert matrix.shape == (2, 3)


@respx.mock
def test_openai_retries_on_429() -> None:
    route = respx.post(_OPENAI_EMBED_URL).mock(
        side_effect=[
            httpx.Response(429),
            httpx.Response(
                200,
                json={"data": [{"index": 0, "embedding": [1.0, 0.0, 0.0]}]},
            ),
        ]
    )
    provider = OpenAIProvider(api_key="test-key", client=httpx.Client())
    matrix = provider.embed_batch(["hello"], model="text-embedding-3-small")
    assert route.call_count == 2
    assert matrix.shape == (1, 3)


def test_circuit_breaker_trips_after_five_failures() -> None:
    state = RetryState()
    with pytest.raises(ProviderPausedError):
        for _ in range(5):
            state.record_failure()


class _CountingProvider:
    calls = 0

    def embed_batch(self, texts, *, model: str, dims: int | None = None):
        type(self).calls += 1
        return np.zeros((len(texts), dims or 3), dtype=np.float32)

    def count_tokens(self, texts, *, model: str) -> int:
        return 1000 * len(texts)

    def limits(self, *, model: str):
        return {
            "tpm": 0,
            "rpm": 0,
            "max_input_tokens": 8192,
            "usd_per_mtok": 0.0,
        }


def test_budget_pre_reservation_skips_embed_when_exhausted() -> None:
    _CountingProvider.calls = 0
    session = Session()
    store = FakeStore({"d1": "hello", "d2": "world"}, dead_letters=set(), session=session)
    cp = FakeControlPlane(session, dead_letters=set())
    budget = BudgetTracker(BudgetConfig(max_usd=1.0, max_tokens=500))
    result = drive_backfill(
        store=store,
        provider=_CountingProvider(),
        control_plane=cp,
        budget=budget,
        target_model="m",
        target_dims=3,
        batch_size=2,
        usd_per_mtok=0.0,
    )
    assert result == "PAUSED_BUDGET"
    assert _CountingProvider.calls == 0


def test_budget_reserve_raises_before_provider() -> None:
    budget = BudgetTracker(BudgetConfig(max_usd=0.0, max_tokens=10))
    with pytest.raises(BudgetExhaustedError):
        budget.reserve(tokens=11, usd=0.0)
