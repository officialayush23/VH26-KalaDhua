//! The extra signals: eight features a frequency counter cannot express.
//!
//! Twin of `training/features_v2.py`. The same maths in the same order, because a feature
//! that means one thing at training time and another at serving time is worse than no
//! feature at all.
//!
//! # Why these eight
//!
//! With honest baselines, a model fed only recency and frequency counters barely beat plain
//! `freq_1m` at predicting reuse. That is not a failure of the model — on a stationary Zipf
//! workload frequency really is most of the signal, and a boosted tree fed frequency
//! counters will rediscover frequency.
//!
//! To beat a frequency counter you have to show the model something it structurally cannot
//! see. Each of the eight below does that, and each satisfies two constraints:
//!
//! 1. **Portable.** No absolute currency, no absolute milliseconds. Those shift by orders of
//!    magnitude between deployments, and every split learned on them silently stops meaning
//!    anything when they do. Cost enters as a *percentile within its own application*, which
//!    means the same thing everywhere.
//! 2. **Cheap.** O(1) per access, a handful of floats of state. These run on the write path.
//!
//! Measured on held-out-application transfer, adding them moved AUC at the 600-second
//! horizon from 0.8797 to 0.9069. In-distribution they added nothing — which is exactly the
//! result the design predicted, and the honest way to describe them.

use crate::types::KeyId;
use ahash::{AHashMap, AHashSet};

/// Names, in the order the model expects them appended after the base features.
pub const EXTRA_FEATURE_NAMES: [&str; 8] = [
    "size_percentile",
    "cost_percentile",
    // Suffixed because the base vector already carries a `cost_variance_ratio` computed
    // per (application, object_type). This one is per application, and two features that
    // share a name are the kind of collision that mis-scores a model in total silence.
    "cost_variance_ratio_app",
    "log_reuse_distance",
    "burstiness",
    "novelty_rate",
    "hour_sin",
    "hour_cos",
];

pub const N_EXTRA: usize = 8;

pub type ExtraFeatures = [f64; N_EXTRA];

/// Quantile levels used to place a value inside its application's distribution.
const LADDER: [f64; 6] = [0.1, 0.25, 0.5, 0.75, 0.9, 0.99];

/// Streaming quantile ladder.
///
/// Six floats per application instead of a histogram. Each level takes the same pinball
/// gradient step the engine already uses for regeneration latency: up by `lr · q` when the
/// observation is above the estimate, down by `lr · (1 − q)` when below.
#[derive(Debug, Clone)]
pub struct QuantileLadder {
    levels: [f64; 6],
    lr: f64,
    seen: u64,
    /// Running mean magnitude of the observations, used only to floor the step size.
    ///
    /// The step is multiplicative (`lr x |level|`), which is scale-free by construction —
    /// but a level sitting at exactly zero would then never move. A fixed floor of 1.0 is
    /// the obvious fix and is wrong: 1.0 is enormous next to a cost of 0.000002 and
    /// negligible next to a latency of 2000, so the ladder converges at one scale and not
    /// the other. Flooring against the data's own magnitude keeps it scale-free.
    mean_abs: f64,
}

impl Default for QuantileLadder {
    fn default() -> Self {
        Self { levels: [0.0; 6], lr: 0.02, seen: 0, mean_abs: 0.0 }
    }
}

impl QuantileLadder {
    pub fn new(lr: f64) -> Self {
        Self { levels: [0.0; 6], lr, seen: 0, mean_abs: 0.0 }
    }

    pub fn observe(&mut self, x: f64) {
        if self.seen == 0 {
            self.levels = [x; 6];
            self.mean_abs = x.abs();
            self.seen = 1;
            return;
        }
        self.mean_abs = 0.05 * x.abs() + 0.95 * self.mean_abs;
        let floor = if self.mean_abs > 0.0 { self.mean_abs * 1e-3 } else { f64::MIN_POSITIVE };
        for i in 0..LADDER.len() {
            let step = self.lr * self.levels[i].abs().max(floor);
            self.levels[i] += if x > self.levels[i] {
                step * LADDER[i]
            } else {
                -step * (1.0 - LADDER[i])
            };
        }
        // Independent SGD on each level can cross them over; a ladder that is not monotone
        // produces percentiles that go backwards.
        for i in 1..LADDER.len() {
            if self.levels[i] < self.levels[i - 1] {
                self.levels[i] = self.levels[i - 1];
            }
        }
        self.seen += 1;
    }

    /// Where `x` sits in the distribution, in `[0, 1]`. Read before observing.
    pub fn percentile_of(&self, x: f64) -> f64 {
        if self.seen == 0 {
            return 0.5;
        }
        let below = self.levels.iter().filter(|level| x >= **level).count();
        if below == 0 {
            return 0.0;
        }
        if below >= LADDER.len() {
            return 1.0;
        }
        let (lo_q, hi_q) = (LADDER[below - 1], LADDER[below]);
        let (lo_v, hi_v) = (self.levels[below - 1], self.levels[below]);
        let span = hi_v - lo_v;
        let frac = if span <= 1e-12 { 0.0 } else { (x - lo_v) / span };
        lo_q + (hi_q - lo_q) * frac.clamp(0.0, 1.0)
    }

    pub fn seen(&self) -> u64 {
        self.seen
    }
}

#[derive(Debug, Clone, Default)]
struct KeyState {
    last_ordinal: i64,
    gap_mean_ms: f64,
    gap_sq_mean_ms: f64,
    gaps_seen: u32,
    last_ts_ms: f64,
}

impl KeyState {
    fn new() -> Self {
        Self { last_ordinal: -1, ..Default::default() }
    }
}

#[derive(Debug, Clone, Default)]
struct AppState {
    size: QuantileLadder,
    cost: QuantileLadder,
    latency_p50: f64,
    latency_p95: f64,
    latency_seen: u64,
}

/// Rolling window of "was this key new", used for the novelty rate.
#[derive(Debug)]
struct BitWindow {
    buf: Vec<bool>,
    next: usize,
    filled: bool,
    ones: usize,
}

impl BitWindow {
    fn new(capacity: usize) -> Self {
        Self { buf: vec![false; capacity.max(1)], next: 0, filled: false, ones: 0 }
    }

    fn push(&mut self, value: bool) {
        if self.filled && self.buf[self.next] {
            self.ones -= 1;
        }
        self.buf[self.next] = value;
        if value {
            self.ones += 1;
        }
        self.next = (self.next + 1) % self.buf.len();
        if self.next == 0 {
            self.filled = true;
        }
    }

    fn len(&self) -> usize {
        if self.filled {
            self.buf.len()
        } else {
            self.next
        }
    }

    fn rate(&self) -> f64 {
        let n = self.len();
        if n == 0 {
            0.0
        } else {
            self.ones as f64 / n as f64
        }
    }
}

/// Computes the eight extra signals, streaming.
///
/// Same discipline as the main feature builder: read state, emit features, *then* fold the
/// current access in. Reversed, every feature quietly contains the access it describes, and
/// the model that learns from it looks excellent offline and is useless online.
#[derive(Debug)]
pub struct SignalBuilder {
    keys: AHashMap<KeyId, KeyState>,
    apps: AHashMap<String, AppState>,
    ordinal: i64,
    quantile_lr: f64,
    novelty: BitWindow,
    novelty_window: usize,
    /// Recent distinct-key ratio, used to convert a gap in access ordinals into an estimated
    /// count of distinct intervening keys.
    unique_ratio: f64,
    recent: Vec<KeyId>,
    recent_at: usize,
}

impl SignalBuilder {
    pub fn new(novelty_window: usize, quantile_lr: f64) -> Self {
        let w = novelty_window.max(64);
        Self {
            keys: AHashMap::new(),
            apps: AHashMap::new(),
            ordinal: 0,
            quantile_lr,
            novelty: BitWindow::new(w),
            novelty_window: w,
            unique_ratio: 1.0,
            recent: Vec::with_capacity(w),
            recent_at: 0,
        }
    }

    pub fn tracked_keys(&self) -> usize {
        self.keys.len()
    }

    /// Drop per-key state for keys not seen since `cutoff_ms`.
    pub fn evict_stale(&mut self, cutoff_ms: f64) -> usize {
        let before = self.keys.len();
        self.keys.retain(|_, s| s.last_ts_ms >= cutoff_ms);
        before - self.keys.len()
    }

    /// The novelty rate on its own — a workload-level regime signal, useful to the engine
    /// outside the feature vector. Near 1.0 means almost every request is for a key never
    /// seen before, which is the signature of a sequential scan.
    pub fn novelty_rate(&self) -> f64 {
        self.novelty.rate()
    }

    pub fn transform(
        &mut self,
        key: KeyId,
        ts_ms: f64,
        application: &str,
        size_bytes: u64,
        cost_usd: f64,
        regen_latency_ms: f64,
    ) -> ExtraFeatures {
        let app = self.apps.entry(application.to_string()).or_default();

        // -- read ---------------------------------------------------------
        let size_pct = app.size.percentile_of(size_bytes as f64);
        let cost_pct = app.cost.percentile_of(cost_usd);
        let variance_ratio =
            if app.latency_seen > 0 { app.latency_p95 / app.latency_p50.max(1.0) } else { 0.0 };

        let existing = self.keys.get(&key);
        let (reuse_distance, burstiness) = match existing {
            Some(state) if state.last_ordinal >= 0 => {
                let gap = (self.ordinal - state.last_ordinal) as f64;
                let distance = gap * self.unique_ratio;
                let burst = if state.gaps_seen >= 2 && state.gap_mean_ms > 1e-9 {
                    let var = (state.gap_sq_mean_ms - state.gap_mean_ms * state.gap_mean_ms).max(0.0);
                    var.sqrt() / state.gap_mean_ms
                } else {
                    0.0
                };
                (distance, burst)
            }
            // A key never seen before has unbounded reuse distance. Encoding it as the window
            // size rather than an infinity keeps the feature finite and comparable.
            _ => (self.novelty_window as f64, 0.0),
        };

        let novelty = self.novelty.rate();

        // Time of day on the unit circle, so 23:59 and 00:01 are adjacent rather than
        // maximally distant. Analytics traffic is diurnal and a model with no clock cannot
        // anticipate the top of the working day.
        let hour = (ts_ms / 3_600_000.0).rem_euclid(24.0);
        let angle = 2.0 * std::f64::consts::PI * hour / 24.0;

        let out: ExtraFeatures = [
            size_pct,
            cost_pct,
            variance_ratio.min(20.0),
            reuse_distance.max(0.0).ln_1p(),
            burstiness.min(10.0),
            novelty,
            angle.sin(),
            angle.cos(),
        ];

        // -- fold in ------------------------------------------------------
        let first_time = existing.is_none();
        self.novelty.push(first_time);

        let ordinal = self.ordinal;
        let state = self.keys.entry(key).or_insert_with(KeyState::new);
        if !first_time {
            let gap = (ts_ms - state.last_ts_ms).max(0.0);
            // An EWMA of the gap and of its square is two floats and tracks a workload that
            // changes, which matters more here than the exactness Welford would buy.
            let alpha = 0.3;
            state.gap_mean_ms = alpha * gap + (1.0 - alpha) * state.gap_mean_ms;
            state.gap_sq_mean_ms = alpha * gap * gap + (1.0 - alpha) * state.gap_sq_mean_ms;
            state.gaps_seen = state.gaps_seen.saturating_add(1);
        }
        state.last_ordinal = ordinal;
        state.last_ts_ms = ts_ms;

        let app = self.apps.entry(application.to_string()).or_default();
        app.size.observe(size_bytes as f64);
        app.cost.observe(cost_usd);
        if regen_latency_ms > 0.0 {
            if app.latency_seen == 0 {
                app.latency_p50 = regen_latency_ms;
                app.latency_p95 = regen_latency_ms;
            } else {
                let lr = self.quantile_lr;
                let s50 = lr * app.latency_p50.max(1.0);
                app.latency_p50 +=
                    if regen_latency_ms > app.latency_p50 { s50 * 0.5 } else { -s50 * 0.5 };
                let s95 = lr * app.latency_p95.max(1.0);
                app.latency_p95 +=
                    if regen_latency_ms > app.latency_p95 { s95 * 0.95 } else { -s95 * 0.05 };
                if app.latency_p95 < app.latency_p50 {
                    app.latency_p95 = app.latency_p50;
                }
            }
            app.latency_seen += 1;
        }

        self.ordinal += 1;

        // Maintain the recent-key ring and refresh the distinct ratio when it wraps.
        if self.recent.len() < self.novelty_window {
            self.recent.push(key);
        } else {
            self.recent[self.recent_at] = key;
            self.recent_at = (self.recent_at + 1) % self.novelty_window;
            if self.recent_at == 0 {
                let mut seen: AHashSet<KeyId> = AHashSet::new();
                for k in &self.recent {
                    seen.insert(*k);
                }
                self.unique_ratio = seen.len() as f64 / self.recent.len() as f64;
            }
        }

        out
    }
}

impl Default for SignalBuilder {
    fn default() -> Self {
        Self::new(2_000, 0.02)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn builder() -> SignalBuilder {
        SignalBuilder::new(2_000, 0.02)
    }

    const SIZE_PCT: usize = 0;
    const COST_PCT: usize = 1;
    const VARIANCE: usize = 2;
    const REUSE_DIST: usize = 3;
    const BURSTINESS: usize = 4;
    const NOVELTY: usize = 5;
    const HOUR_SIN: usize = 6;
    const HOUR_COS: usize = 7;

    #[test]
    fn a_first_access_reports_maximum_reuse_distance() {
        let mut b = builder();
        let f = b.transform(1, 0.0, "app", 1000, 0.001, 10.0);
        assert!((f[REUSE_DIST] - (2000.0f64).ln_1p()).abs() < 1e-9);
        assert_eq!(f[BURSTINESS], 0.0);
    }

    #[test]
    fn reuse_distance_grows_with_intervening_keys() {
        let mut b = builder();
        b.transform(1, 0.0, "app", 1000, 0.001, 10.0);
        // Five other keys pass by.
        for k in 100..105u64 {
            b.transform(k, 1.0, "app", 1000, 0.001, 10.0);
        }
        let near = b.transform(1, 10.0, "app", 1000, 0.001, 10.0);

        let mut c = builder();
        c.transform(2, 0.0, "app", 1000, 0.001, 10.0);
        for k in 200..250u64 {
            c.transform(k, 1.0, "app", 1000, 0.001, 10.0);
        }
        let far = c.transform(2, 10.0, "app", 1000, 0.001, 10.0);

        assert!(
            far[REUSE_DIST] > near[REUSE_DIST],
            "fifty intervening keys should give a larger reuse distance than five: {} vs {}",
            far[REUSE_DIST],
            near[REUSE_DIST]
        );
    }

    #[test]
    fn burstiness_separates_steady_from_bursty_keys_of_equal_frequency() {
        // Key 1: every 1000 ms exactly. Key 2: ten rapid accesses, then a long silence.
        let mut b = builder();
        let mut steady = 0.0;
        for i in 1..=20 {
            steady = b.transform(1, i as f64 * 1000.0, "app", 1000, 0.001, 10.0)[BURSTINESS];
        }

        let mut c = builder();
        let mut t = 0.0;
        let mut bursty = 0.0;
        for i in 0..20 {
            t += if i % 10 == 0 { 9_000.0 } else { 10.0 };
            bursty = c.transform(2, t, "app", 1000, 0.001, 10.0)[BURSTINESS];
        }

        assert!(
            bursty > steady,
            "a bursty key should score higher than a steady one: {bursty} vs {steady}"
        );
        assert!(steady < 0.2, "a perfectly regular key should be near zero, got {steady}");
    }

    #[test]
    fn novelty_rate_approaches_one_during_a_scan() {
        let mut b = builder();
        // Warm up on a small working set.
        for round in 0..50u64 {
            for k in 0..20u64 {
                b.transform(k, (round * 20 + k) as f64, "app", 1000, 0.001, 10.0);
            }
        }
        let settled = b.novelty_rate();
        // Now scan: every key is new.
        for i in 0..2_000u64 {
            b.transform(1_000_000 + i, 100_000.0 + i as f64, "app", 1000, 0.001, 10.0);
        }
        let scanning = b.novelty_rate();
        assert!(settled < 0.2, "a repeating working set should have low novelty, got {settled}");
        assert!(scanning > 0.9, "a scan should drive novelty near 1.0, got {scanning}");
    }

    #[test]
    fn cost_percentile_is_scale_free() {
        // The property that lets `cost_percentile` replace absolute cost in the model.
        // The same relative distribution at wildly different absolute scales must produce
        // the same percentile, or a model trained on one deployment is worthless on
        // another — silently, which is the dangerous part.
        //
        // A lognormal spread is used rather than a monotone ramp: a streaming estimator
        // trails a ramp by construction, and testing against one measures lag rather than
        // scale-invariance.
        fn percentile_at_scale(scale: f64) -> f64 {
            let mut rng = crate::rng::Rng::seed_from_u64(7);
            let mut b = builder();
            let mut values: Vec<f64> = (0..4_000).map(|_| rng.log_normal(0.0, 1.0) * scale).collect();
            for (i, v) in values.iter().enumerate() {
                b.transform(i as u64, i as f64, "app", 1_000, *v, 10.0);
            }
            values.sort_by(|a, c| a.partial_cmp(c).unwrap());
            let probe = values[(values.len() as f64 * 0.8) as usize];
            b.transform(999_999, 9_999.0, "app", 1_000, probe, 10.0)[COST_PCT]
        }

        let cents = percentile_at_scale(1e-6);      // dollars per rebuild
        let scaled = percentile_at_scale(0.1);      // the same shape, 100,000x larger
        let millis = percentile_at_scale(200.0);    // and again, as if it were milliseconds

        assert!(
            (cents - scaled).abs() < 0.05 && (cents - millis).abs() < 0.05,
            "percentile drifted across scales: {cents:.3} / {scaled:.3} / {millis:.3}"
        );
        assert!(
            (cents - 0.8).abs() < 0.08,
            "the true 80th percentile should read near 0.8, got {cents:.3}"
        );
    }

    #[test]
    fn percentiles_are_per_application_not_global() {
        let mut b = builder();
        // Analytics objects are small; content objects are large.
        for i in 0..200u64 {
            b.transform(i, i as f64, "analytics", 40_000, 0.00001, 5.0);
            b.transform(1_000 + i, i as f64, "content", 8_000_000, 0.00001, 5.0);
        }
        // A 40 KB object is typical for analytics and tiny for content.
        let in_analytics = b.transform(5_000, 500.0, "analytics", 40_000, 0.00001, 5.0)[SIZE_PCT];
        let in_content = b.transform(5_001, 500.0, "content", 40_000, 0.00001, 5.0)[SIZE_PCT];
        assert!(
            in_analytics > in_content,
            "the same size should rank higher among small objects: {in_analytics} vs {in_content}"
        );
    }

    #[test]
    fn hour_phase_is_continuous_across_midnight() {
        let mut b = builder();
        let before = b.transform(1, 23.99 * 3_600_000.0, "app", 100, 0.0, 0.0);
        let after = b.transform(2, 24.01 * 3_600_000.0, "app", 100, 0.0, 0.0);
        let distance =
            ((before[HOUR_SIN] - after[HOUR_SIN]).powi(2) + (before[HOUR_COS] - after[HOUR_COS]).powi(2)).sqrt();
        assert!(distance < 0.02, "midnight should be continuous, distance was {distance}");
    }

    #[test]
    fn variance_ratio_reflects_a_heavy_tail() {
        let mut b = builder();
        // Predictable operation: always about 200 ms.
        for i in 0..500u64 {
            b.transform(i, i as f64, "steady", 1000, 0.001, 200.0);
        }
        let steady = b.transform(9_000, 600.0, "steady", 1000, 0.001, 200.0)[VARIANCE];

        // Erratic operation: usually 80 ms, occasionally 2 s.
        for i in 0..500u64 {
            let latency = if i % 20 == 0 { 2_000.0 } else { 80.0 };
            b.transform(10_000 + i, i as f64, "erratic", 1000, 0.001, latency);
        }
        let erratic = b.transform(19_000, 600.0, "erratic", 1000, 0.001, 80.0)[VARIANCE];

        assert!(
            erratic > steady,
            "an erratic operation should have a higher p95/p50 ratio: {erratic} vs {steady}"
        );
    }

    #[test]
    fn features_never_include_the_access_that_produced_them() {
        // The first access to a key in a brand-new application cannot have observed itself.
        let mut b = builder();
        let f = b.transform(1, 0.0, "fresh", 12345, 0.5, 100.0);
        assert_eq!(f[NOVELTY], 0.0, "novelty counted the current access");
        assert_eq!(f[VARIANCE], 0.0, "variance counted the current latency");
        assert_eq!(f[SIZE_PCT], 0.5, "an empty ladder must report the neutral 0.5");
    }

    #[test]
    fn quantile_ladder_stays_monotone() {
        let mut q = QuantileLadder::new(0.05);
        let mut rng = crate::rng::Rng::seed_from_u64(3);
        for _ in 0..5_000 {
            q.observe(rng.log_normal(3.0, 1.2));
        }
        for i in 1..LADDER.len() {
            assert!(
                q.levels[i] >= q.levels[i - 1],
                "ladder crossed over at level {i}: {:?}",
                q.levels
            );
        }
        assert!(q.percentile_of(f64::MIN) <= 0.0 + 1e-9);
        assert!(q.percentile_of(f64::MAX) >= 1.0 - 1e-9);
    }

    #[test]
    fn state_is_bounded_by_eviction() {
        let mut b = builder();
        for i in 0..50_000u64 {
            b.transform(i, i as f64, "app", 1000, 0.001, 10.0);
        }
        assert_eq!(b.tracked_keys(), 50_000);
        let dropped = b.evict_stale(25_000.0);
        assert!(dropped > 0);
        assert!(b.tracked_keys() < 50_000);
    }
}
