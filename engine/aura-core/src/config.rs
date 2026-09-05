//! Engine configuration.
//!
//! Loaded from `engine/config/default.toml`, overridable per key by environment
//! variables of the form `AURA__SECTION__KEY` (see [`Config::apply_env`]).

use crate::types::CostVector;
use serde::{Deserialize, Serialize};

/// Bytes in a gigabyte, decimal — cloud providers bill in decimal gigabytes and so do we.
pub const BYTES_PER_GB: f64 = 1_000_000_000.0;

/// USD price table. Turns a physical cost vector into money.
///
/// These numbers are mirrored in `training/aura_train/config.py`. `regen_cost_usd` is a
/// model feature, so a price table that drifts between the two silently shifts the feature
/// distribution the model was trained on. There is a test that pins them.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Pricing {
    pub cpu_ms_usd: f64,
    pub gpu_ms_usd: f64,
    pub db_ms_usd: f64,
    pub network_gb_usd: f64,
    pub cache_gb_hour_usd: f64,
    pub sla_penalty_per_ms_over_slo_usd: f64,
    pub slo_p95_ms: f64,
}

impl Default for Pricing {
    fn default() -> Self {
        Self {
            cpu_ms_usd: 0.000_000_011_6,
            gpu_ms_usd: 0.000_000_255,
            db_ms_usd: 0.000_000_032_0,
            network_gb_usd: 0.09,
            cache_gb_hour_usd: 0.0225,
            sla_penalty_per_ms_over_slo_usd: 0.000_000_4,
            slo_p95_ms: 150.0,
        }
    }
}

impl Pricing {
    /// Price one regeneration. Mirrors `Pricing.regen_cost_usd` in the training pipeline.
    pub fn regen_cost_usd(&self, c: &CostVector) -> f64 {
        c.cpu_ms * self.cpu_ms_usd
            + c.gpu_ms * self.gpu_ms_usd
            + c.db_ms * self.db_ms_usd
            + (c.network_bytes / BYTES_PER_GB) * self.network_gb_usd
            + c.api_cost_usd
    }

    /// What it costs to hold `bytes` for `ms` milliseconds.
    pub fn holding_cost_usd(&self, bytes: f64, ms: f64) -> f64 {
        (bytes / BYTES_PER_GB) * self.cache_gb_hour_usd * (ms / 3_600_000.0)
    }

    /// Penalty for a request that breached the latency objective.
    pub fn sla_penalty_usd(&self, latency_ms: f64, weight: f64) -> f64 {
        let over = (latency_ms - self.slo_p95_ms).max(0.0);
        over * self.sla_penalty_per_ms_over_slo_usd * weight
    }
}

/// Constants baked into the feature builder. Changing any of these invalidates every
/// trained model, because the golden fixture in `training/tests/golden/` pins the outputs.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct FeatureConfig {
    pub inter_arrival_alpha: f64,
    pub freq_windows_ms: [f64; 3],
    pub half_life_fast_s: f64,
    pub half_life_slow_s: f64,
    pub trend_eps: f64,
    pub quantile_lr: f64,
}

impl Default for FeatureConfig {
    fn default() -> Self {
        Self {
            inter_arrival_alpha: 0.3,
            freq_windows_ms: [60_000.0, 300_000.0, 3_600_000.0],
            half_life_fast_s: 5.0,
            half_life_slow_s: 60.0,
            trend_eps: 1e-6,
            quantile_lr: 0.02,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct L1Config {
    pub capacity_bytes: u64,
}

impl Default for L1Config {
    fn default() -> Self {
        Self { capacity_bytes: 32 * 1024 * 1024 }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct CacheConfig {
    pub capacity_bytes: u64,
    pub shards: usize,
    pub l1: L1Config,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self { capacity_bytes: 512 * 1024 * 1024, shards: 16, l1: L1Config::default() }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct EngineConfig {
    /// How many victims to sample when the cache is under pressure. The model only ever
    /// scores this many objects, which is what keeps inference off the hot path.
    pub candidate_sample: usize,
    pub controller_tick_ms: f64,
    /// An object must clear the current eviction threshold by this factor to be admitted.
    pub admission_margin: f64,
    /// Refresh is considered once less than this fraction of the TTL remains.
    pub refresh_ttl_threshold: f64,
    /// Below this confidence the learned preference is ignored entirely and the engine
    /// runs on heuristics. Cold start depends on this.
    pub ml_confidence_floor: f64,
    /// Weight given to each reuse horizon when computing expected value.
    pub horizon_weights: [f64; 3],
    /// Extra weight on the tail of the regeneration-latency distribution. Objects whose
    /// misses are *occasionally* catastrophic are worth more than their median suggests.
    pub tail_risk_lambda: f64,
    /// Fraction of the TTL at which an object stops being fresh and becomes serve-once
    /// stale. 0.8 means the last fifth of the lifetime is the refresh-ahead window: readers
    /// still get an answer immediately while the rebuild happens behind them, which is what
    /// stops a popular key turning into a thundering herd the instant it expires.
    pub soft_ttl_fraction: f64,
    /// How long one caller may hold the rebuild lease for a missing key before another is
    /// allowed to take it. Long enough to cover an honest rebuild, short enough that a
    /// crashed caller cannot wedge the key.
    pub rebuild_lease_ms: f64,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            candidate_sample: 32,
            controller_tick_ms: 100.0,
            admission_margin: 1.10,
            refresh_ttl_threshold: 0.15,
            ml_confidence_floor: 0.20,
            horizon_weights: [0.5, 0.35, 0.15],
            tail_risk_lambda: 0.35,
            soft_ttl_fraction: 0.8,
            rebuild_lease_ms: 5_000.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct CapacityConfig {
    pub auto: bool,
    pub step_bytes: u64,
    pub min_bytes: u64,
    pub max_bytes: u64,
    /// Memory the host is willing to give the cache in total, across all nodes.
    pub host_budget_bytes: u64,
    /// Marginal benefit must exceed marginal cost by this factor before scaling.
    pub roi_threshold: f64,
    /// Minimum seconds between capacity actions, so the controller cannot oscillate.
    pub cooldown_s: f64,
}

impl Default for CapacityConfig {
    fn default() -> Self {
        Self {
            auto: true,
            step_bytes: 256 * 1024 * 1024,
            min_bytes: 128 * 1024 * 1024,
            max_bytes: 4 * 1024 * 1024 * 1024,
            host_budget_bytes: 2 * 1024 * 1024 * 1024,
            roi_threshold: 1.25,
            cooldown_s: 5.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PredictorConfig {
    /// `heuristic` | `linear` | `gbdt`
    pub kind: String,
    pub bundle_path: Option<String>,
    pub supabase_autoload: bool,
    /// Learning rate for the always-on online logistic model.
    pub online_lr: f64,
}

impl Default for PredictorConfig {
    fn default() -> Self {
        Self {
            kind: "heuristic".to_string(),
            bundle_path: None,
            supabase_autoload: false,
            online_lr: 0.05,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct BanditConfig {
    /// `thompson` | `epsilon_greedy` | `exp3`
    pub kind: String,
    pub exploration: f64,
    /// How often the bandit re-picks an arm, in engine milliseconds.
    pub decision_interval_ms: f64,
}

impl Default for BanditConfig {
    fn default() -> Self {
        Self {
            kind: "thompson".to_string(),
            exploration: 0.08,
            decision_interval_ms: 2_000.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct GhostConfig {
    /// Fraction of keys sampled into the reuse-distance histogram. SHARDS-style spatial
    /// sampling: a 1% sample tracks the miss-ratio curve within a couple of percent at a
    /// hundredth of the memory.
    pub sample_rate: f64,
    pub buckets: usize,
    /// Largest capacity, as a multiple of the current one, that the curve extrapolates to.
    pub max_probe_multiple: f64,
}

impl Default for GhostConfig {
    fn default() -> Self {
        Self { sample_rate: 0.01, buckets: 64, max_probe_multiple: 4.0 }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct Config {
    pub cache: CacheConfig,
    pub pricing: Pricing,
    pub features: FeatureConfig,
    pub engine: EngineConfig,
    pub capacity: CapacityConfig,
    pub predictor: PredictorConfig,
    pub bandit: BanditConfig,
    pub ghost: GhostConfig,
}

impl Config {
    pub fn from_toml_str(text: &str) -> anyhow::Result<Self> {
        Ok(toml::from_str(text)?)
    }

    pub fn from_path(path: &std::path::Path) -> anyhow::Result<Self> {
        let text = std::fs::read_to_string(path)?;
        Self::from_toml_str(&text)
    }

    /// Overlay `AURA__SECTION__KEY` environment variables onto the loaded config.
    ///
    /// Only the knobs a deployment actually needs to move are wired up; everything else
    /// stays in the file, where it is reviewable.
    pub fn apply_env(&mut self) {
        fn u64_var(name: &str) -> Option<u64> {
            std::env::var(name).ok()?.parse().ok()
        }
        fn f64_var(name: &str) -> Option<f64> {
            std::env::var(name).ok()?.parse().ok()
        }
        fn bool_var(name: &str) -> Option<bool> {
            std::env::var(name).ok()?.parse().ok()
        }

        if let Some(v) = u64_var("AURA__CACHE__CAPACITY_BYTES") {
            self.cache.capacity_bytes = v;
        }
        if let Some(v) = u64_var("AURA__CACHE__L1__CAPACITY_BYTES") {
            self.cache.l1.capacity_bytes = v;
        }
        if let Some(v) = bool_var("AURA__CDN__ENABLED") {
        }
        if let Some(v) = bool_var("AURA__CAPACITY__AUTO") {
            self.capacity.auto = v;
        }
        if let Some(v) = u64_var("AURA__CAPACITY__HOST_BUDGET_BYTES") {
            self.capacity.host_budget_bytes = v;
        }
        if let Some(v) = f64_var("AURA__CAPACITY__ROI_THRESHOLD") {
            self.capacity.roi_threshold = v;
        }
        if let Ok(v) = std::env::var("AURA__PREDICTOR__KIND") {
            self.predictor.kind = v;
        }
        if let Ok(v) = std::env::var("AURA__PREDICTOR__BUNDLE_PATH") {
            self.predictor.bundle_path = Some(v);
        }
        if let Some(v) = bool_var("AURA__PREDICTOR__SUPABASE_AUTOLOAD") {
            self.predictor.supabase_autoload = v;
        }
        if let Ok(v) = std::env::var("AURA__BANDIT__KIND") {
            self.bandit.kind = v;
        }
    }

    /// Load `path` if it exists, fall back to defaults if it does not, then apply the
    /// environment. A missing config file is not an error: the defaults are a working
    /// configuration, which is what makes `docker run` with no volume work.
    pub fn load(path: Option<&std::path::Path>) -> Self {
        let mut cfg = match path {
            Some(p) if p.exists() => Self::from_path(p).unwrap_or_else(|err| {
                tracing::warn!(?p, %err, "config file could not be parsed, using defaults");
                Config::default()
            }),
            _ => Config::default(),
        };
        cfg.apply_env();
        cfg
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pricing_matches_the_training_pipeline() {
        let p = Pricing::default();
        let c = CostVector {
            cpu_ms: 320.0,
            gpu_ms: 80.0,
            db_ms: 140.0,
            network_bytes: 1_854_200.0,
            api_calls: 1.0,
            api_cost_usd: 0.002,
            latency_ms: 412.0,
        };
        // Computed with the Python price table; any drift is a bug on one of the two sides.
        let expected = 320.0 * 0.000_000_011_6
            + 80.0 * 0.000_000_255
            + 140.0 * 0.000_000_032_0
            + (1_854_200.0 / 1e9) * 0.09
            + 0.002;
        assert!((p.regen_cost_usd(&c) - expected).abs() < 1e-15);
    }

    #[test]
    fn defaults_round_trip_through_toml() {
        let text = toml::to_string(&Config::default()).expect("serialize");
        let parsed = Config::from_toml_str(&text).expect("parse");
        assert_eq!(parsed, Config::default());
    }
}
