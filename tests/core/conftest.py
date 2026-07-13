"""Hypothesis profiles for the correctness-core property suite.

* ``ci``   -- 200 examples/property (INVARIANTS I13 floor); the default.
* ``dev``  -- 500 examples/property (the local acceptance bar in P2).
* ``fast`` -- 25 examples for a quick inner loop.

Select with ``HYPOTHESIS_PROFILE=dev`` (etc.). Deadlines are disabled because a few
generated schedules legitimately run many resume rounds.
"""

from __future__ import annotations

import os

from hypothesis import HealthCheck, settings

_COMMON = {
    "deadline": None,
    "suppress_health_check": [HealthCheck.too_slow, HealthCheck.data_too_large],
}

settings.register_profile("fast", max_examples=25, **_COMMON)
settings.register_profile("ci", max_examples=200, **_COMMON)
settings.register_profile("dev", max_examples=500, **_COMMON)

settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "ci"))
