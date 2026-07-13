"""Typer CLI entry point for alembicio verbs."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from alembicio.config import EmbMigrateConfig, load_config

app = typer.Typer(
    name="alembicio",
    help="Zero-downtime embedding model migration orchestrator.",
    no_args_is_help=True,
)

ConfigPath = Annotated[
    Path,
    typer.Option(
        "--config",
        "-c",
        help="Path to embmigrate.yaml",
        exists=True,
        readable=True,
        resolve_path=True,
    ),
]


def _load_config(path: Path) -> EmbMigrateConfig:
    return load_config(path)


@app.command()
def init(
    config: ConfigPath = Path("embmigrate.yaml"),
) -> None:
    """Initialize migration state from config."""
    _load_config(config)
    raise NotImplementedError("init")


@app.command()
def doctor(
    config: ConfigPath = Path("embmigrate.yaml"),
) -> None:
    """Run preflight checks and cost estimates."""
    _load_config(config)
    raise NotImplementedError("doctor")


@app.command()
def prepare(
    config: ConfigPath = Path("embmigrate.yaml"),
) -> None:
    """Prepare target store schema and dual-write capture."""
    _load_config(config)
    raise NotImplementedError("prepare")


@app.command()
def backfill(
    config: ConfigPath = Path("embmigrate.yaml"),
    resume: Annotated[
        bool,
        typer.Option("--resume", help="Resume an interrupted backfill run"),
    ] = False,
) -> None:
    """Backfill target embeddings with checkpointed, resumable workers."""
    _load_config(config)
    if resume:
        typer.echo("resume requested")
    raise NotImplementedError("backfill")


@app.command()
def verify(
    config: ConfigPath = Path("embmigrate.yaml"),
) -> None:
    """Run golden-query verification and emit a report."""
    _load_config(config)
    raise NotImplementedError("verify")


@app.command()
def cutover(
    config: ConfigPath = Path("embmigrate.yaml"),
    canary: Annotated[
        int | None,
        typer.Option("--canary", min=1, max=99, help="Canary traffic percentage"),
    ] = None,
) -> None:
    """Flip read path to the new embedding space."""
    _load_config(config)
    if canary is not None:
        typer.echo(f"canary {canary}% requested")
    raise NotImplementedError("cutover")


@app.command()
def rollback(
    config: ConfigPath = Path("embmigrate.yaml"),
) -> None:
    """Flip read path back to the source embedding space."""
    _load_config(config)
    raise NotImplementedError("rollback")


@app.command()
def decommission(
    config: ConfigPath = Path("embmigrate.yaml"),
) -> None:
    """Remove migration artifacts after soak."""
    _load_config(config)
    raise NotImplementedError("decommission")


@app.command()
def status(
    config: ConfigPath = Path("embmigrate.yaml"),
) -> None:
    """Show migration state, progress, and spend."""
    _load_config(config)
    raise NotImplementedError("status")


def main() -> None:
    """Console script entry point."""
    app()


if __name__ == "__main__":
    main()
