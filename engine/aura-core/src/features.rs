//! The feature builder.
//!
//! This is the twin of `training/aura_train/features.py`. The two implementations are
//! asserted against a shared golden fixture (`training/tests/golden/feature_vectors.json`)
//! by [`tests::golden_vectors_match_the_training_pipeline`], because a feature that means
//! one thing at training time and another at serving time is worse than no feature at all.
//!
//! Two rules govern everything below:
//!
//! 1. **No leakage.** Every value emitted for the access at time `t` derives only from
//!    accesses at or before `t`. Counters are decayed and *read*, then the current access
//!    is folded in. Regeneration quantiles are read before the current observation is
//!    folded in, because at decision time the engine has not yet paid for this miss.
//! 2. **Streaming.** One pass, O(unique keys) memory, no lookahead.

use crate::config::{FeatureConfig, Pricing};
use crate::sketch::{DecayCounter, QuantilePair};
use crate::types::{app_id, CostVector, KeyId};
use ahash::AHashMap;

/// Number of features in the vector the models consume.
/// Sixteen base features plus the eight extra signals from [`crate::signals`].
///
/// The base sixteen are computed here and pinned by the golden fixture shared with the
/// training pipeline. The extra eight are computed by `SignalBuilder` and written into
/// slots 16..24 by the engine, because they need per-application state this builder does
/// not carry.
///
/// A trained model does not have to use all of them. The bundle names the columns it wants
/// and the loader projects this vector onto them, so the engine and the trainer can evolve
/// their feature sets independently.
pub const N_FEATURES: usize = 24;

/// Where the extra signals begin.
pub const EXTRA_OFFSET: usize = 16;

/// Feature names, in contract order. Serialised into every model bundle and checked on
/// load, so a bundle trained against a different vector is rejected rather than silently
/// mis-scored.
pub const FEATURE_NAMES: [&str; N_FEATURES] = [
    "log_age_ms",
    "log_inter_arrival_ms",
    "freq_1m",
    "freq_5m",
    "freq_1h",
    "ewma_fast",
    "ewma_slow",
    "trend",
    "acceleration",
    "log_size_bytes",
    "log_regen_p50_ms",
    "cost_variance_ratio",
    "regen_cost_usd",
    "ttl_remaining_frac",
    "cache_pressure",
    "app_id",
    // --- from signals.rs, written by the engine into slots 16..24 ---
    "size_percentile",
    "cost_percentile",
    "cost_variance_ratio_app",
    "log_reuse_distance",
    "burstiness",
    "novelty_rate",
    "hour_sin",
    "hour_cos",
];

/// A dense feature vector. Fixed size, `Copy`, never heap-allocated on the hot path.
pub type Features = [f64; N_FEATURES];

/// Indices, so downstream code can talk about features by name without string lookups.
pub mod idx {
    pub const LOG_AGE_MS: usize = 0;
    pub const LOG_INTER_ARRIVAL_MS: usize = 1;
    pub const FREQ_1M: usize = 2;
    pub const FREQ_5M: usize = 3;
    pub const FREQ_1H: usize = 4;
    pub const EWMA_FAST: usize = 5;
    pub const EWMA_SLOW: usize = 6;
    pub const TREND: usize = 7;
    pub const ACCELERATION: usize = 8;
    pub const LOG_SIZE_BYTES: usize = 9;
    pub const LOG_REGEN_P50_MS: usize = 10;
    pub const COST_VARIANCE_RATIO: usize = 11;
    pub const REGEN_COST_USD: usize = 12;
    pub const TTL_REMAINING_FRAC: usize = 13;
    pub const CACHE_PRESSURE: usize = 14;
    pub const APP_ID: usize = 15;
}

/// Per-key state carried between accesses. 88 bytes; there is one of these per resident
/// *and* recently-seen key, so it is deliberately small.
#[derive(Debug, Clone, Default)]
pub struct KeyState {
    pub last_ts_ms: f64,
    pub ewma_gap_ms: f64,
    pub seen_gap: bool,
    pub freq: [DecayCounter; 3],
    pub ewma_fast: DecayCounter,
    pub ewma_slow: DecayCounter,
    pub prev_trend: f64,
    pub seen_trend: bool,
    pub accesses: u64,
    pub first_ts_ms: f64,
}

/// One access, as the feature builder sees it. The live engine fills this from a request;
/// the simulator and the trace reader fill it from a row.
#[derive(Debug, Clone)]
pub struct AccessEvent {
    pub ts_ms: f64,
    pub key_id: KeyId,
    pub application: String,
    pub object_type: String,
    pub size_bytes: u64,
    pub ttl_ms: f64,
    pub regen: CostVector,
    /// Measured regeneration latency for *this* access, if it was a miss. Folded into the
    /// group quantiles only after the features have been read.
    pub regen_latency_ms: f64,
}

/// Ambient state the builder cannot know on its own.
#[derive(Debug, Clone, Copy, Default)]
pub struct AmbientState {
    /// `used_bytes / capacity_bytes`, read before this access changes occupancy.
    pub cache_pressure: f64,
    /// Freshness of the resident copy, in `[0, 1]`. Zero when nothing is resident.
    pub ttl_remaining_frac: f64,
}

/// Running regeneration-cost shape for one `(application, object_type)` group.
///
/// Grouping at this level is what keeps the cost model small: the engine does not fit a
/// regression per key, it learns the shape of each operation class and lets the per-key
/// features carry the rest.
#[derive(Debug, Clone, Default)]
pub struct CostGroup {
    pub latency: QuantilePair,
    pub cost: CostVector,
    pub observations: u64,
}

#[derive(Debug)]
pub struct FeatureBuilder {
    cfg: FeatureConfig,
    pricing: Pricing,
    keys: AHashMap<KeyId, KeyState>,
    groups: AHashMap<(String, String), CostGroup>,
    last_ts_ms: f64,
}

impl FeatureBuilder {
    pub fn new(cfg: FeatureConfig, pricing: Pricing) -> Self {
        Self {
            cfg,
            pricing,
            keys: AHashMap::new(),
            groups: AHashMap::new(),
            last_ts_ms: f64::NEG_INFINITY,
        }
    }

    pub fn pricing(&self) -> &Pricing {
        &self.pricing
    }

    pub fn tracked_keys(&self) -> usize {
        self.keys.len()
    }

    pub fn key_state(&self, key: KeyId) -> Option<&KeyState> {
        self.keys.get(&key)
    }

    /// Regeneration-cost shape for a group, if the engine has ever paid for one.
    pub fn cost_group(&self, application: &str, object_type: &str) -> Option<&CostGroup> {
        self.groups.get(&(application.to_string(), object_type.to_string()))
    }

    /// Drop per-key state for keys not seen since `cutoff_ms`. Called by the controller
    /// tick; without it the builder's memory grows with the total key space rather than
    /// the working set.
    pub fn evict_stale(&mut self, cutoff_ms: f64) -> usize {
        let before = self.keys.len();
        self.keys.retain(|_, s| s.last_ts_ms >= cutoff_ms);
        before - self.keys.len()
    }

    /// Build the feature vector for `event` and advance the builder's state.
    ///
    /// Not idempotent: every call advances the counters exactly once, which mirrors the
    /// engine, where an access happens once.
    pub fn transform(&mut self, event: &AccessEvent, ambient: AmbientState) -> Features {
        debug_assert!(
            event.ts_ms >= self.last_ts_ms,
            "feature builder requires non-decreasing timestamps"
        );
        self.last_ts_ms = event.ts_ms;

        let entry = self.keys.entry(event.key_id);
        let first_access = matches!(entry, std::collections::hash_map::Entry::Vacant(_));
        let state = match entry {
            std::collections::hash_map::Entry::Occupied(o) => o.into_mut(),
            std::collections::hash_map::Entry::Vacant(v) => v.insert(KeyState {
                last_ts_ms: event.ts_ms,
                first_ts_ms: event.ts_ms,
                ..Default::default()
            }),
        };

        let dt_ms = if first_access { 0.0 } else { (event.ts_ms - state.last_ts_ms).max(0.0) };

        // 0, 1 — recency. A key's first access has no measurable gap and reports zero
        // rather than a fabricated one.
        let log_age_ms = dt_ms.ln_1p();
        if !first_access {
            if state.seen_gap {
                let a = self.cfg.inter_arrival_alpha;
                state.ewma_gap_ms = a * dt_ms + (1.0 - a) * state.ewma_gap_ms;
            } else {
                state.ewma_gap_ms = dt_ms;
                state.seen_gap = true;
            }
        }
        let log_inter_arrival_ms = state.ewma_gap_ms.ln_1p();

        // 2..4 — decayed frequency. Decay, read, then fold this access in.
        let mut freqs = [0.0f64; 3];
        for i in 0..3 {
            let decayed = state.freq[i].decayed_tau(dt_ms, self.cfg.freq_windows_ms[i]);
            freqs[i] = decayed;
            state.freq[i].set(decayed + 1.0);
        }

        // 5..8 — trend. Two half-lives; their ratio says whether interest in this key is
        // building or dying, which is precisely what frequency-only policies cannot see.
        let fast = state.ewma_fast.decayed_half_life(dt_ms, self.cfg.half_life_fast_s);
        let slow = state.ewma_slow.decayed_half_life(dt_ms, self.cfg.half_life_slow_s);
        state.ewma_fast.set(fast + 1.0);
        state.ewma_slow.set(slow + 1.0);
        let eps = self.cfg.trend_eps;
        let trend = ((fast + eps) / (slow + eps)).ln();
        let acceleration = if state.seen_trend { trend - state.prev_trend } else { 0.0 };
        state.prev_trend = trend;
        state.seen_trend = true;

        state.last_ts_ms = event.ts_ms;
        state.accesses += 1;

        // 9 — footprint.
        let log_size_bytes = (event.size_bytes as f64).max(0.0).ln_1p();

        // 10, 11 — regeneration-cost shape, read before this observation is folded in.
        let group = self
            .groups
            .entry((event.application.clone(), event.object_type.clone()))
            .or_default();
        let regen_p50 = group.latency.p50;
        let regen_p95 = group.latency.p95;
        let log_regen_p50_ms = regen_p50.max(0.0).ln_1p();
        let cost_variance_ratio = regen_p95 / regen_p50.max(1.0);
        if event.regen_latency_ms > 0.0 {
            group.latency.observe(event.regen_latency_ms, self.cfg.quantile_lr);
        }
        if !event.regen.is_empty() {
            group.cost = if group.observations == 0 {
                event.regen
            } else {
                group.cost.blend(&event.regen, 0.1)
            };
            group.observations += 1;
        }

        // 12 — the priced cost vector.
        let regen_cost_usd = self.pricing.regen_cost_usd(&event.regen);

        let base: [f64; EXTRA_OFFSET] = [
            log_age_ms,
            log_inter_arrival_ms,
            freqs[0],
            freqs[1],
            freqs[2],
            fast,
            slow,
            trend,
            acceleration,
            log_size_bytes,
            log_regen_p50_ms,
            cost_variance_ratio,
            regen_cost_usd,
            ambient.ttl_remaining_frac,
            ambient.cache_pressure,
            app_id(&event.application) as f64,
        ];
        // Slots 16..24 stay zero here. The engine fills them from `SignalBuilder`, which
        // keeps per-application distributions this builder has no business owning.
        let mut out = [0.0f64; N_FEATURES];
        out[..EXTRA_OFFSET].copy_from_slice(&base);
        out
    }

    /// Rebuild a feature vector for a resident object *without* recording an access.
    ///
    /// Eviction scoring needs current features for objects nobody just asked for. Reading
    /// the counters at `now_ms` without folding anything in is the difference between
    /// scoring an object and pretending it was used.
    pub fn peek(
        &self,
        key: KeyId,
        now_ms: f64,
        application: &str,
        object_type: &str,
        size_bytes: u64,
        regen: &CostVector,
        ambient: AmbientState,
    ) -> Features {
        let mut out = [0.0f64; N_FEATURES];
        let (regen_p50, regen_p95) = match self.groups.get(&(application.to_string(), object_type.to_string())) {
            Some(g) => (g.latency.p50, g.latency.p95),
            None => (0.0, 0.0),
        };
        out[idx::LOG_SIZE_BYTES] = (size_bytes as f64).max(0.0).ln_1p();
        out[idx::LOG_REGEN_P50_MS] = regen_p50.max(0.0).ln_1p();
        out[idx::COST_VARIANCE_RATIO] = regen_p95 / regen_p50.max(1.0);
        out[idx::REGEN_COST_USD] = self.pricing.regen_cost_usd(regen);
        out[idx::TTL_REMAINING_FRAC] = ambient.ttl_remaining_frac;
        out[idx::CACHE_PRESSURE] = ambient.cache_pressure;
        out[idx::APP_ID] = app_id(application) as f64;

        if let Some(state) = self.keys.get(&key) {
            let dt_ms = (now_ms - state.last_ts_ms).max(0.0);
            out[idx::LOG_AGE_MS] = dt_ms.ln_1p();
            out[idx::LOG_INTER_ARRIVAL_MS] = state.ewma_gap_ms.ln_1p();
            for i in 0..3 {
                out[idx::FREQ_1M + i] = state.freq[i].decayed_tau(dt_ms, self.cfg.freq_windows_ms[i]);
            }
            let fast = state.ewma_fast.decayed_half_life(dt_ms, self.cfg.half_life_fast_s);
            let slow = state.ewma_slow.decayed_half_life(dt_ms, self.cfg.half_life_slow_s);
            out[idx::EWMA_FAST] = fast;
            out[idx::EWMA_SLOW] = slow;
            let eps = self.cfg.trend_eps;
            out[idx::TREND] = ((fast + eps) / (slow + eps)).ln();
            out[idx::ACCELERATION] = out[idx::TREND] - state.prev_trend;
        }
        out
    }
}

/// Name a raw vector. Used by the explain endpoint; never on the hot path.
pub fn named(features: &Features) -> Vec<(&'static str, f64)> {
    FEATURE_NAMES.iter().copied().zip(features.iter().copied()).collect()
}

/// The LRU replica that reproduces `cache_pressure` and `ttl_remaining_frac` offline.
///
/// The training pipeline has no access to the engine's occupancy series, so it
/// reconstructs a deterministic one from a plain LRU of the configured size. This type
/// exists so the Rust side can reproduce the golden fixture exactly; the live engine reads
/// its real occupancy instead.
#[derive(Debug)]
pub struct PressureReplica {
    capacity_bytes: u64,
    used_bytes: u64,
    order: Vec<KeyId>,
    entries: AHashMap<KeyId, (u64, f64, f64)>,
}

impl PressureReplica {
    pub fn new(capacity_bytes: u64) -> Self {
        Self {
            capacity_bytes: capacity_bytes.max(1),
            used_bytes: 0,
            order: Vec::new(),
            entries: AHashMap::new(),
        }
    }

    pub fn pressure(&self) -> f64 {
        (self.used_bytes as f64 / self.capacity_bytes as f64).min(1.0)
    }

    fn resident(&self, key: KeyId, now_ms: f64) -> Option<(u64, f64, f64)> {
        let e = *self.entries.get(&key)?;
        if e.2 > 0.0 && now_ms - e.1 >= e.2 {
            return None;
        }
        Some(e)
    }

    pub fn ttl_remaining_frac(&self, key: KeyId, now_ms: f64) -> f64 {
        match self.resident(key, now_ms) {
            Some((_, fill, ttl)) if ttl > 0.0 => (1.0 - (now_ms - fill) / ttl).clamp(0.0, 1.0),
            _ => 0.0,
        }
    }

    pub fn touch(&mut self, key: KeyId, now_ms: f64, size_bytes: u64, ttl_ms: f64) -> bool {
        if self.resident(key, now_ms).is_some() {
            self.order.retain(|k| *k != key);
            self.order.push(key);
            return true;
        }
        self.fill(key, now_ms, size_bytes, ttl_ms);
        false
    }

    fn fill(&mut self, key: KeyId, now_ms: f64, size_bytes: u64, ttl_ms: f64) {
        if let Some((old, _, _)) = self.entries.remove(&key) {
            self.used_bytes -= old;
            self.order.retain(|k| *k != key);
        }
        if size_bytes > self.capacity_bytes {
            return;
        }
        while self.used_bytes + size_bytes > self.capacity_bytes && !self.order.is_empty() {
            let victim = self.order.remove(0);
            if let Some((sz, _, _)) = self.entries.remove(&victim) {
                self.used_bytes -= sz;
            }
        }
        self.entries.insert(key, (size_bytes, now_ms, ttl_ms));
        self.order.push(key);
        self.used_bytes += size_bytes;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    fn golden_path() -> std::path::PathBuf {
        // engine/aura-core -> repository root -> training/tests/golden
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../training/tests/golden/feature_vectors.json")
    }

    /// The contract between this file and the training pipeline. If this fails, one side
    /// has drifted and every trained model in the registry is scoring against a different
    /// feature space than it was fitted on.
    #[test]
    fn golden_vectors_match_the_training_pipeline() {
        let path = golden_path();
        let Ok(text) = std::fs::read_to_string(&path) else {
            eprintln!("golden fixture not present at {}, skipping parity check", path.display());
            return;
        };
        let doc: Value = serde_json::from_str(&text).expect("golden fixture is valid json");

        let cfg_json = &doc["config"];
        let cfg = FeatureConfig {
            inter_arrival_alpha: cfg_json["inter_arrival_alpha"].as_f64().unwrap(),
            freq_windows_ms: [
                cfg_json["freq_windows_ms"][0].as_f64().unwrap(),
                cfg_json["freq_windows_ms"][1].as_f64().unwrap(),
                cfg_json["freq_windows_ms"][2].as_f64().unwrap(),
            ],
            half_life_fast_s: cfg_json["half_life_fast_s"].as_f64().unwrap(),
            half_life_slow_s: cfg_json["half_life_slow_s"].as_f64().unwrap(),
            trend_eps: cfg_json["trend_eps"].as_f64().unwrap(),
            quantile_lr: cfg_json["quantile_lr"].as_f64().unwrap(),
        };
        let capacity = cfg_json["sim_capacity_bytes"].as_u64().unwrap();

        // The fixture pins the base sixteen. The extra eight are computed elsewhere and
        // have their own tests, so the fixture must match this vector's *prefix*, not its
        // whole length.
        let names = doc["feature_names"].as_array().expect("feature_names");
        assert_eq!(names.len(), EXTRA_OFFSET, "the golden fixture covers the base features");
        for (i, n) in names.iter().enumerate() {
            assert_eq!(n.as_str().unwrap(), FEATURE_NAMES[i], "feature order drifted at {i}");
        }

        let mut builder = FeatureBuilder::new(cfg, Pricing::default());
        let mut replica = PressureReplica::new(capacity);

        for (case_no, case) in doc["cases"].as_array().unwrap().iter().enumerate() {
            let e = &case["event"];
            let event = AccessEvent {
                ts_ms: e["ts_ms"].as_f64().unwrap(),
                key_id: e["key_id"].as_u64().unwrap(),
                application: e["application"].as_str().unwrap().to_string(),
                object_type: e["object_type"].as_str().unwrap().to_string(),
                size_bytes: e["size_bytes"].as_u64().unwrap(),
                ttl_ms: e["ttl_ms"].as_f64().unwrap(),
                regen: CostVector {
                    cpu_ms: e["cpu_ms"].as_f64().unwrap_or(0.0),
                    gpu_ms: e["gpu_ms"].as_f64().unwrap_or(0.0),
                    db_ms: e["db_ms"].as_f64().unwrap_or(0.0),
                    network_bytes: e["network_bytes"].as_f64().unwrap_or(0.0),
                    api_calls: e["api_calls"].as_f64().unwrap_or(0.0),
                    api_cost_usd: e["api_cost_usd"].as_f64().unwrap_or(0.0),
                    latency_ms: e["regen_latency_ms"].as_f64().unwrap_or(0.0),
                },
                regen_latency_ms: e["regen_latency_ms"].as_f64().unwrap_or(0.0),
            };

            let ambient = AmbientState {
                cache_pressure: replica.pressure(),
                ttl_remaining_frac: replica.ttl_remaining_frac(event.key_id, event.ts_ms),
            };
            let got = builder.transform(&event, ambient);
            replica.touch(event.key_id, event.ts_ms, event.size_bytes, event.ttl_ms);

            let want = case["features"].as_array().unwrap();
            for i in 0..EXTRA_OFFSET {
                let w = want[i].as_f64().unwrap();
                let g = got[i];
                assert!(
                    (w - g).abs() <= 1e-9 * w.abs().max(1.0),
                    "case {case_no} feature {} ({}): python {w} != rust {g}",
                    i,
                    FEATURE_NAMES[i]
                );
            }
        }
    }

    #[test]
    fn counters_report_strictly_prior_accesses() {
        let mut b = FeatureBuilder::new(FeatureConfig::default(), Pricing::default());
        let mk = |ts: f64| AccessEvent {
            ts_ms: ts,
            key_id: 1,
            application: "analytics".into(),
            object_type: "q".into(),
            size_bytes: 1000,
            ttl_ms: 0.0,
            regen: CostVector::default(),
            regen_latency_ms: 0.0,
        };
        let f0 = b.transform(&mk(0.0), AmbientState::default());
        assert_eq!(f0[idx::FREQ_1M], 0.0, "the first access must not count itself");
        let f1 = b.transform(&mk(1.0), AmbientState::default());
        assert!(f1[idx::FREQ_1M] > 0.99 && f1[idx::FREQ_1M] <= 1.0);
    }

    #[test]
    fn trend_rises_for_an_emerging_key_and_falls_for_a_dying_one() {
        let mut b = FeatureBuilder::new(FeatureConfig::default(), Pricing::default());
        let mk = |ts: f64, key: u64| AccessEvent {
            ts_ms: ts,
            key_id: key,
            application: "content".into(),
            object_type: "blob".into(),
            size_bytes: 4096,
            ttl_ms: 0.0,
            regen: CostVector::default(),
            regen_latency_ms: 0.0,
        };
        // Key 1 accelerates: gaps shrink. Key 2 decays: gaps grow.
        let mut t = 0.0;
        let mut last_hot = 0.0;
        for gap in [4000.0, 2000.0, 1000.0, 500.0, 250.0, 120.0] {
            t += gap;
            last_hot = b.transform(&mk(t, 1), AmbientState::default())[idx::TREND];
        }
        let mut t2 = 0.0;
        let mut last_cold = 0.0;
        for gap in [120.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0] {
            t2 += gap;
            last_cold = b.transform(&mk(t2 + 100_000.0, 2), AmbientState::default())[idx::TREND];
        }
        assert!(
            last_hot > last_cold,
            "emerging key trend {last_hot} should exceed dying key trend {last_cold}"
        );
    }

    #[test]
    fn peek_does_not_advance_state() {
        let mut b = FeatureBuilder::new(FeatureConfig::default(), Pricing::default());
        let e = AccessEvent {
            ts_ms: 0.0,
            key_id: 5,
            application: "analytics".into(),
            object_type: "q".into(),
            size_bytes: 2048,
            ttl_ms: 1000.0,
            regen: CostVector::default(),
            regen_latency_ms: 12.0,
        };
        b.transform(&e, AmbientState::default());
        let before = b.key_state(5).unwrap().accesses;
        let _ = b.peek(5, 500.0, "analytics", "q", 2048, &CostVector::default(), AmbientState::default());
        assert_eq!(b.key_state(5).unwrap().accesses, before);
    }
}
