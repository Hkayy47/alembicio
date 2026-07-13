"""Config loading and round-trip tests."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from alembicio.config import (
    ConfigError,
    EmbMigrateConfig,
    dump_config,
    dump_config_yaml,
    load_config,
    load_config_raw,
    resolve_env_value,
)

EXAMPLE_PATH = Path("examples/embmigrate.example.yaml")


def test_example_yaml_loads_without_env_resolution() -> None:
    config = load_config(EXAMPLE_PATH, resolve_env=False)
    assert config.migration == "demo-minilm-to-bge"
    assert config.source.provider == "fastembed"
    assert config.target.dim == 384
    assert config.store.kind == "pgvector"
    assert config.store.content_hash_expr == "md5(content::text)"
    assert config.mapping.kind == "none"
    assert config.verify.gates.min_recall_ratio == 1.0


def test_content_hash_expr_default() -> None:
    config = load_config(EXAMPLE_PATH, resolve_env=False)
    assert config.store.text_column == "content"
    assert config.store.content_hash_expr == "md5(content::text)"


def test_config_round_trip_structure() -> None:
    config = load_config(EXAMPLE_PATH, resolve_env=False)
    round_tripped = EmbMigrateConfig.model_validate(dump_config(config))
    assert round_tripped == config


def test_dump_config_yaml_parses_back() -> None:
    config = load_config(EXAMPLE_PATH, resolve_env=False)
    dumped = dump_config_yaml(config)
    reparsed = yaml.safe_load(dumped)
    assert EmbMigrateConfig.model_validate(reparsed) == config


def test_env_resolution() -> None:
    os.environ["DATABASE_URL"] = "postgresql://test:test@localhost/test"
    try:
        config = load_config(EXAMPLE_PATH, resolve_env=True)
        assert config.store.dsn == "postgresql://test:test@localhost/test"
    finally:
        os.environ.pop("DATABASE_URL", None)


def test_env_resolution_missing_raises() -> None:
    os.environ.pop("DATABASE_URL", None)
    with pytest.raises(ConfigError, match="DATABASE_URL"):
        load_config(EXAMPLE_PATH, resolve_env=True)


def test_resolve_env_value_literal() -> None:
    assert resolve_env_value("postgresql://local", field_name="dsn") == "postgresql://local"


def test_load_config_raw_is_dict() -> None:
    data = load_config_raw(EXAMPLE_PATH)
    assert data["migration"] == "demo-minilm-to-bge"
    assert "store" in data
