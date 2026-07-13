"""Runtime helpers for request-level canary routing."""

from __future__ import annotations


def choose_space(request_key: str, *, active: str, canary_pct: int, seed: int) -> str:
    """Return ``old`` or ``new`` for a request key given canary settings.

    Args:
        request_key: Stable identifier for the request (session id, user id, etc.).
        active: Current default space when canary is disabled.
        canary_pct: Percentage of requests routed to the non-default space.
        seed: Deterministic seed stored in read_state.

    Returns:
        ``"old"`` or ``"new"``.
    """
    raise NotImplementedError("choose_space")
