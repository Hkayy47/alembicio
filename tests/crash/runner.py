"""Subprocess driver for ``alembicio`` verbs with fault injection and SIGKILL.

The crash harness runs the *real* CLI as a child process so that a ``kill -9`` exercises
true process death, not a caught exception. Fault timing is handed to the worker via the
env-var contract owned by :mod:`alembicio.core.worker`.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import dataclass

from alembicio.core.worker import (
    FAULT_AFTER_N_BATCHES_ENV,
    FAULT_INTRA_BATCH_SLEEP_MS_ENV,
)

_CLI = [sys.executable, "-m", "alembicio.cli"]


@dataclass(frozen=True)
class RunResult:
    """Outcome of a subprocess verb invocation."""

    returncode: int
    killed: bool
    stdout: str
    stderr: str


def _base_env(dsn: str) -> dict[str, str]:
    env = dict(os.environ)
    env["DATABASE_URL"] = dsn
    # Keep any accidental fault vars out of non-fault runs (e.g. the baseline).
    env.pop(FAULT_AFTER_N_BATCHES_ENV, None)
    env.pop(FAULT_INTRA_BATCH_SLEEP_MS_ENV, None)
    return env


def run_verb(
    verb: str,
    *,
    dsn: str,
    config_path: str,
    extra_args: tuple[str, ...] = (),
    timeout: float = 300.0,
) -> RunResult:
    """Run an ``alembicio`` verb to completion.

    Args:
        verb: The CLI verb (e.g. ``prepare``, ``backfill``).
        dsn: Postgres DSN exported as ``DATABASE_URL``.
        config_path: Path to the migration yaml.
        extra_args: Additional CLI arguments.
        timeout: Seconds before the call is aborted.

    Returns:
        The :class:`RunResult`.
    """
    proc = subprocess.run(
        [*_CLI, verb, "--config", config_path, *extra_args],
        env=_base_env(dsn),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    return RunResult(
        returncode=proc.returncode, killed=False, stdout=proc.stdout, stderr=proc.stderr
    )


def run_backfill_with_kill(
    *,
    dsn: str,
    config_path: str,
    fault_after_n_batches: int,
    intra_batch_sleep_ms: int,
    kill_after_seconds: float,
    timeout: float = 300.0,
) -> RunResult:
    """Start ``backfill``, then SIGKILL it mid-run at the injected fault window.

    The worker sleeps once at least ``fault_after_n_batches`` batches are durably
    upserted (post-upsert / pre-ledger-commit). We wait ``kill_after_seconds`` and then
    hard-kill the process so death lands in that window. If the process finishes before
    the kill (few batches, fast run), it simply exits normally and ``killed`` is False.

    Args:
        dsn: Postgres DSN.
        config_path: Path to the migration yaml.
        fault_after_n_batches: Batches to durably upsert before entering the fault window.
        intra_batch_sleep_ms: Milliseconds to hold in the fault window.
        kill_after_seconds: Wall-clock delay before SIGKILL.
        timeout: Seconds before the call is aborted.

    Returns:
        The :class:`RunResult` (``killed`` True if we delivered the kill).
    """
    env = _base_env(dsn)
    env[FAULT_AFTER_N_BATCHES_ENV] = str(fault_after_n_batches)
    env[FAULT_INTRA_BATCH_SLEEP_MS_ENV] = str(intra_batch_sleep_ms)

    proc = subprocess.Popen(
        [*_CLI, "backfill", "--config", config_path],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    killed = False
    deadline = time.monotonic() + kill_after_seconds
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            break
        time.sleep(0.02)
    if proc.poll() is None:
        proc.kill()  # SIGKILL on POSIX, TerminateProcess on Windows
        killed = True
    stdout, stderr = proc.communicate(timeout=timeout)
    return RunResult(
        returncode=proc.returncode, killed=killed, stdout=stdout, stderr=stderr
    )


def backfill_stack_ready(*, dsn: str, config_path: str) -> bool:
    """Probe whether the execution stack (CLI + adapter + provider) is wired.

    Runs ``alembicio prepare`` and treats a ``NotImplementedError``/``WorkerError`` exit
    as "not wired yet". This lets the 50-seed resume loop skip cleanly until the pgvector
    adapter, fastembed provider, and ``init``/``prepare``/``backfill`` verbs land, and
    activate automatically once they do.

    Args:
        dsn: Postgres DSN.
        config_path: Path to the migration yaml.

    Returns:
        True if ``prepare`` ran to a clean exit, False otherwise.
    """
    try:
        result = run_verb("prepare", dsn=dsn, config_path=config_path, timeout=60.0)
    except (subprocess.TimeoutExpired, OSError):
        return False
    if result.returncode == 0:
        return True
    blockers = ("NotImplementedError", "WorkerError", "not wired")
    return not any(marker in result.stderr for marker in blockers)
