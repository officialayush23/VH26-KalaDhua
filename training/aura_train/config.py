"""Configuration objects for the AURA training pipeline.

Everything that a run needs to be reproducible lives in :class:`TrainingConfig`.
Values can be overridden from the environment with ``AURA_TRAIN_<FIELD>`` so the
same code runs unchanged in Colab, in CI and on a workstation.

The pricing block is a mirror of ``[pricing]`` in ``engine/config/default.toml``
(contract section 8). It must stay numerically identical to the engine, because
``regen_cost_usd`` is a model feature and a drifting price table silently
shifts the feature distribution.
"""

from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

LOG = logging.getLogger(__name__)

# Contract section 8, [pricing]. USD.
CPU_MS_USD = 0.0000000116
GPU_MS_USD = 0.000000255
DB_MS_USD = 0.0000000320
NETWORK_GB_USD = 0.09
CACHE_GB_HOUR_USD = 0.0225
SLA_PENALTY_PER_MS_OVER_SLO_USD = 0.0000004
SLO_P95_MS = 150.0

BYTES_PER_GB = 1_000_000_000.0


@dataclass(frozen=True)
class Pricing:
    """USD price table used to turn a raw cost vector into a scalar."""

    cpu_ms_usd: float = CPU_MS_USD
    gpu_ms_usd: float = GPU_MS_USD
    db_ms_usd: float = DB_MS_USD
    network_gb_usd: float = NETWORK_GB_USD
    cache_gb_hour_usd: float = CACHE_GB_HOUR_USD
    sla_penalty_per_ms_over_slo_usd: float = SLA_PENALTY_PER_MS_OVER_SLO_USD
    slo_p95_ms: float = SLO_P95_MS

    def regen_cost_usd(
        self,
        cpu_ms: float,
        gpu_ms: float,
        db_ms: float,
        network_bytes: float,
        api_cost_usd: float,
    ) -> float:
        """Price one regeneration. Mirrors ``CostVector::usd`` on the Rust side."""
        return (
            cpu_ms * self.cpu_ms_usd
            + gpu_ms * self.gpu_ms_usd
            + db_ms * self.db_ms_usd
            + (network_bytes / BYTES_PER_GB) * self.network_gb_usd
            + api_cost_usd
        )


@dataclass(frozen=True)
class FeatureConfig:
    """Constants baked into the feature builder.

    Any change here is a breaking change: the Rust feature builder in
    ``engine/aura-core`` hard-codes the same numbers, and
    ``tests/golden/feature_vectors.json`` pins the resulting outputs.
    """

    inter_arrival_alpha: float = 0.3
    freq_windows_ms: tuple[float, ...] = (60_000.0, 300_000.0, 3_600_000.0)
    half_life_fast_s: float = 5.0
    half_life_slow_s: float = 60.0
    trend_eps: float = 1e-6
    quantile_lr: float = 0.02
    # Capacity of the offline cache replica used to derive ``cache_pressure``.
    sim_capacity_bytes: int = 536_870_912


@dataclass(frozen=True)
class SplitConfig:
    """Regime-stratified split. This is the scientific claim of the project.

    We never split randomly: a random split leaks the future of a key into the
    training set and inflates AUC by a wide margin. Train and validate on
    ordinary regimes, test on regimes the model has never seen.
    """

    train_regimes: tuple[str, ...] = ("steady", "zipf_shift_moderate", "analytics_stable")
    test_regimes: tuple[str, ...] = ("flash_crowd", "scan", "expensive_tail", "cost_spike")
    # Trailing fraction of each training regime, by time, held out for validation.
    val_tail_fraction: float = 0.2


@dataclass(frozen=True)
class GbdtParams:
    """LightGBM hyper-parameters (also mapped onto the scikit-learn fallback)."""

    num_leaves: int = 63
    learning_rate: float = 0.06
    n_estimators: int = 400
    min_child_samples: int = 50
    subsample: float = 0.85
    subsample_freq: int = 1
    colsample_bytree: float = 0.9
    reg_lambda: float = 1.0
    early_stopping_rounds: int = 40
    max_bin: int = 255
    seed: int = 42


@dataclass
class TrainingConfig:
    """Top-level run configuration."""

    trace_dir: Path = Path("data/traces")
    dataset_dir: Path = Path("data/dataset")
    model_dir: Path = Path("models")
    report_dir: Path = Path("reports")

    horizons_ms: tuple[int, ...] = (10_000, 60_000, 600_000)
    primary_horizon_ms: int = 60_000

    shard_rows: int = 250_000
    max_rows_per_trace: int | None = None

    features: FeatureConfig = field(default_factory=FeatureConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    gbdt: GbdtParams = field(default_factory=GbdtParams)
    pricing: Pricing = field(default_factory=Pricing)

    # Supabase
    storage_bucket: str = "aura-models"
    trace_bucket: str = "aura-traces"

    seed: int = 42

    def horizon_label(self, horizon_ms: int) -> str:
        """``60000 -> 'h60s'``. Used in column names and bundle names."""
        seconds = horizon_ms // 1000
        return f"h{seconds}s"

    def label_column(self, horizon_ms: int) -> str:
        return f"label_{self.horizon_label(horizon_ms)}"

    def censored_column(self, horizon_ms: int) -> str:
        return f"censored_{self.horizon_label(horizon_ms)}"

    def ensure_dirs(self) -> None:
        for path in (self.trace_dir, self.dataset_dir, self.model_dir, self.report_dir):
            path.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict[str, object]:
        raw = asdict(self)
        for key in ("trace_dir", "dataset_dir", "model_dir", "report_dir"):
            raw[key] = str(raw[key])
        return raw


_PATH_FIELDS = {"trace_dir", "dataset_dir", "model_dir", "report_dir"}
_INT_FIELDS = {"shard_rows", "primary_horizon_ms", "seed"}


def load_config(**overrides: object) -> TrainingConfig:
    """Build a config from defaults, then environment, then explicit overrides.

    Environment variables are named ``AURA_TRAIN_<UPPERCASE_FIELD>``, e.g.
    ``AURA_TRAIN_TRACE_DIR=/data/traces AURA_TRAIN_SHARD_ROWS=100000``.
    """
    cfg = TrainingConfig()
    for name in list(_PATH_FIELDS) + list(_INT_FIELDS) + ["max_rows_per_trace", "storage_bucket"]:
        env_key = f"AURA_TRAIN_{name.upper()}"
        raw = os.environ.get(env_key)
        if raw is None:
            continue
        if name in _PATH_FIELDS:
            setattr(cfg, name, Path(raw))
        elif name in _INT_FIELDS or name == "max_rows_per_trace":
            setattr(cfg, name, int(raw))
        else:
            setattr(cfg, name, raw)
        LOG.debug("config override from env: %s=%s", name, raw)

    for name, value in overrides.items():
        if value is None:
            continue
        if not hasattr(cfg, name):
            raise ValueError(f"unknown config field: {name}")
        if name in _PATH_FIELDS:
            value = Path(str(value))
        setattr(cfg, name, value)
    return cfg
