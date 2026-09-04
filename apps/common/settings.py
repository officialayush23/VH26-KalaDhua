"""Environment-driven configuration shared by the example applications.

Nothing here has a secret baked in. Every credential-shaped value comes from the
process environment, which is also how the compose deployment injects it.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import AliasChoices, BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Pricing(BaseModel):
    """Unit prices from `engine/config/default.toml` (contract section 8).

    The applications price their *measured* resource usage with exactly the same
    table the engine uses, so the USD numbers on both sides are comparable.
    """

    cpu_ms_usd: float = 0.0000000116
    gpu_ms_usd: float = 0.000000255
    db_ms_usd: float = 0.0000000320
    network_gb_usd: float = 0.09
    cache_gb_hour_usd: float = 0.0225
    sla_penalty_per_ms_over_slo_usd: float = 0.0000004
    slo_p95_ms: float = 150.0


class Settings(BaseSettings):
    """Process configuration.

    Read with the `AURA_APPS_` prefix, e.g. `AURA_APPS_AURA_BASE_URL`.
    A few keys carry deployment-standard names as well (`SUPABASE_*`,
    `LOG_LEVEL`) and are accepted unprefixed.
    """

    model_config = SettingsConfigDict(
        env_prefix="AURA_APPS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    aura_base_url: str = "http://localhost:8080"
    aura_timeout_s: float = 2.0
    aura_connect_timeout_s: float = 0.5
    aura_max_connections: int = 64

    # Circuit breaker guarding every call to aura-server.
    breaker_failure_threshold: int = 5
    breaker_reset_s: float = 10.0

    # Analytics backend. When the Postgres URL is absent the app falls back to a
    # local SQLite fixture carrying the same schema, so it always boots.
    supabase_direct_connection_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "AURA_APPS_SUPABASE_DIRECT_CONNECTION_URL",
            "SUPABASE_DIRECT_CONNECTION_URL",
        ),
    )
    db_statement_timeout_ms: int = 15_000
    db_pool_min_size: int = 1
    db_pool_max_size: int = 8
    sqlite_path: str = "/tmp/aura_analytics.sqlite3"
    sqlite_orders: int = 300_000

    log_level: str = Field(
        default="INFO",
        validation_alias=AliasChoices("AURA_APPS_LOG_LEVEL", "LOG_LEVEL"),
    )
    seed: int = 42

    # Expensive-tail defaults. ~5% of keys, 20-50x the regeneration cost of the
    # rest, at identical access frequency.
    tail_fraction: float = 0.05
    tail_multiplier: float = 30.0

    pricing: Pricing = Field(default_factory=Pricing)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton."""
    return Settings()
