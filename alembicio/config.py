"""Pydantic models and loaders for embmigrate.yaml."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

_ENV_PREFIX = "env:"
_ENV_PATTERN = re.compile(r"^env:([A-Za-z_][A-Za-z0-9_]*)$")


class ConfigError(ValueError):
    """Raised when embmigrate.yaml is invalid or references missing environment variables."""


def resolve_env_value(value: str, *, field_name: str) -> str:
    """Resolve ``env:VAR`` indirection to the environment variable value."""
    match = _ENV_PATTERN.match(value.strip())
    if match is None:
        return value
    var_name = match.group(1)
    resolved = os.environ.get(var_name)
    if resolved is None:
        msg = f"{field_name}: environment variable {var_name!r} is not set"
        raise ConfigError(msg)
    return resolved


def _resolve_env_fields(data: Any, *, path: str = "") -> Any:
    """Recursively resolve env: strings in parsed YAML data."""
    if isinstance(data, dict):
        return {
            key: _resolve_env_fields(value, path=f"{path}.{key}" if path else key)
            for key, value in data.items()
        }
    if isinstance(data, list):
        return [
            _resolve_env_fields(item, path=f"{path}[{index}]")
            for index, item in enumerate(data)
        ]
    if isinstance(data, str) and data.startswith(_ENV_PREFIX):
        return resolve_env_value(data, field_name=path or "config")
    return data


class ModelSpec(BaseModel):
    """Source or target embedding model specification."""

    provider: Literal["openai", "gemini", "fastembed"]
    model: str
    dim: int = Field(gt=0)


class StoreConfig(BaseModel):
    """Vector store connection and table mapping."""

    kind: Literal["pgvector", "qdrant"]
    dsn: str
    table: str
    id_column: str
    text_column: str | None = None
    content_ref: str | None = None
    content_hash_expr: str | None = None
    old_embedding_column: str | None = None

    @model_validator(mode="after")
    def validate_text_source(self) -> Self:
        if self.text_column is None and self.content_ref is None:
            msg = "store: at least one of text_column or content_ref is required"
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def default_content_hash_expr(self) -> Self:
        if self.content_hash_expr is None:
            if self.text_column is None:
                msg = "store.content_hash_expr is required when text_column is absent"
                raise ValueError(msg)
            object.__setattr__(
                self,
                "content_hash_expr",
                f"md5({self.text_column}::text)",
            )
        return self


class BudgetConfig(BaseModel):
    """Token and USD spending limits for backfill."""

    max_usd: float = Field(ge=0)
    max_tokens: int = Field(ge=0)


class RateLimitConfig(BaseModel):
    """Provider rate limits; zero means unthrottled (local providers)."""

    tpm: int = Field(ge=0)
    rpm: int = Field(ge=0)


class BackfillConfig(BaseModel):
    """Backfill worker tuning."""

    batch_size: int = Field(default=256, gt=0)
    budget: BudgetConfig
    rate_limit: RateLimitConfig
    on_poison: Literal["dead_letter"] = "dead_letter"


class VerifyGates(BaseModel):
    """Golden-query verification thresholds."""

    min_recall_ratio: float = Field(default=1.0, gt=0)
    k: int = Field(default=10, gt=0)
    report: str = "report.md"


class VerifyConfig(BaseModel):
    """Verification harness configuration."""

    golden_queries: str
    gates: VerifyGates
    allow_unindexed: bool = False


class CutoverConfig(BaseModel):
    """Staged cutover and soak settings."""

    mode: Literal["staged"] = "staged"
    canary_pct: int = Field(default=5, ge=0, le=100)
    soak_hours: int = Field(default=72, gt=0)


class MappingConfig(BaseModel):
    """Optional Procrustes/ridge projection stopgap."""

    kind: Literal["procrustes", "ridge", "none"] = "none"
    mode: Literal["default", "mapping_only"] = "default"
    anchors: int = Field(default=4096, gt=0)
    min_recovery: float = Field(default=0.95, gt=0, le=1.0)


class PricingConfig(BaseModel):
    """Optional per-model USD/M-token overrides."""

    usd_per_mtok: dict[str, float] = Field(default_factory=dict)

    @field_validator("usd_per_mtok")
    @classmethod
    def validate_prices(cls, value: dict[str, float]) -> dict[str, float]:
        for price in value.values():
            if price < 0:
                msg = "pricing.usd_per_mtok values must be non-negative"
                raise ValueError(msg)
        return value


class EmbMigrateConfig(BaseModel):
    """Root configuration for a single migration."""

    migration: str
    source: ModelSpec
    target: ModelSpec
    store: StoreConfig
    backfill: BackfillConfig
    verify: VerifyConfig
    cutover: CutoverConfig
    mapping: MappingConfig = Field(default_factory=MappingConfig)
    pricing: PricingConfig | None = None


def load_config_raw(path: Path | str) -> dict[str, Any]:
    """Load YAML from disk without env resolution or validation."""
    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        msg = f"{config_path}: expected a YAML mapping at the top level"
        raise ConfigError(msg)
    return data


def load_config(path: Path | str, *, resolve_env: bool = True) -> EmbMigrateConfig:
    """Load and validate embmigrate.yaml, optionally resolving env: secrets."""
    data = load_config_raw(path)
    if resolve_env:
        data = _resolve_env_fields(data)
    return EmbMigrateConfig.model_validate(data)


def dump_config(config: EmbMigrateConfig) -> dict[str, Any]:
    """Serialize config to a plain dict suitable for YAML export."""
    return config.model_dump(mode="python")


def dump_config_yaml(config: EmbMigrateConfig) -> str:
    """Serialize config to YAML text."""
    return yaml.safe_dump(dump_config(config), sort_keys=False, default_flow_style=False)
