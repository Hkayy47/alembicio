"""CLI smoke tests."""

from __future__ import annotations

import os
from pathlib import Path

from typer.testing import CliRunner

from alembicio.cli import app

runner = CliRunner()
EXAMPLE_CONFIG = Path("examples/embmigrate.example.yaml")


def test_help_lists_all_verbs() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for verb in (
        "init",
        "doctor",
        "prepare",
        "backfill",
        "verify",
        "cutover",
        "rollback",
        "decommission",
        "status",
    ):
        assert verb in result.output


def test_init_stub_raises_not_implemented() -> None:
    os.environ["DATABASE_URL"] = "postgresql://test:test@localhost/test"
    try:
        result = runner.invoke(app, ["init", "--config", str(EXAMPLE_CONFIG)])
        assert result.exit_code != 0
        assert isinstance(result.exception, NotImplementedError)
        assert str(result.exception) == "init"
    finally:
        os.environ.pop("DATABASE_URL", None)
