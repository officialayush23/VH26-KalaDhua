use aura_core::features::{idx, Features};
use aura_core::rng::Rng;
use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Policy {
    Lru,
    Lfu,
    Gdsf,
    TinyLfu,
    CostAware,
    TrendAware,
}

impl Policy {
    pub const ALL: [Policy; 6] = [
        Policy::Lru,
        Policy::Lfu,
        Policy::Gdsf,
        Policy::TinyLfu,
        Policy::CostAware,
        Policy::TrendAware,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Policy::Lru => "lru",
            Policy::Lfu => "lfu",
            Policy::Gdsf => "gdsf",
            Policy::TinyLfu => "tiny_lfu",
            Policy::CostAware => "cost_aware",
            Policy::TrendAware => "trend_aware",
        }
    }

    /// Utility of keeping an object, in the units each classical policy actually optimizes.
    /// They are deliberately not normalised against each other: the mixture below learns
    /// the scale it needs.
    pub fn utility(self, f: &Features, age_ms: f64) -> f64 {
        let size = f[idx::LOG_SIZE_BYTES].max(1.0);
        let cost = f[idx::REGEN_COST_USD].max(1e-9);
        match self {
            Policy::Lru => 1.0 / (1.0 + age_ms / 1000.0),
            Policy::Lfu => f[idx::FREQ_1H],
            Policy::Gdsf => (f[idx::FREQ_5M] * cost) / size,
            Policy::TinyLfu => f[idx::FREQ_1M] * 0.7 + f[idx::FREQ_5M] * 0.3,
            Policy::CostAware => cost / size,
            Policy::TrendAware => {
                f[idx::EWMA_FAST] * (1.0 + f[idx::TREND].max(0.0)) + f[idx::ACCELERATION].max(0.0)
            }
        }
    }
}

/// Thompson sampling over the six experts. Reward is realised cost avoided per unit of
/// occupancy, so an arm that keeps cheap objects resident loses even when its hit rate
/// looks fine.
#[derive(Debug)]
pub struct Bandit {
    alpha: [f64; 6],
    beta: [f64; 6],
    pending: [f64; 6],
    exploration: f64,
    rng: Rng,
    pub pulls: u64,
    pub regret: f64,
}

impl Bandit {
    pub fn new(exploration: f64, seed: u64) -> Self {
        Self {
            alpha: [1.0; 6],
            beta: [1.0; 6],
            pending: [0.0; 6],
            exploration,
            rng: Rng::seed_from_u64(seed),
            pulls: 0,
            regret: 0.0,
        }
    }

    /// Posterior mean weight per expert, renormalised. This is the mixture the engine uses
    /// and the dashboard shows.
    pub fn mixture(&self) -> [f64; 6] {
        let mut w = [0.0f64; 6];
        let mut sum = 0.0;
        for i in 0..6 {
            let m = self.alpha[i] / (self.alpha[i] + self.beta[i]);
            let m = m + self.exploration;
            w[i] = m;
            sum += m;
        }
        if sum > 0.0 {
            for v in w.iter_mut() {
                *v /= sum;
            }
        }
        w
    }

    /// One Thompson draw, used when the engine wants a single expert rather than a blend.
    pub fn sample_arm(&mut self) -> Policy {
        self.pulls += 1;
        let mut best = 0usize;
        let mut best_v = f64::MIN;
        for i in 0..6 {
            let m = self.alpha[i] / (self.alpha[i] + self.beta[i]);
            let spread = 1.0 / (self.alpha[i] + self.beta[i]).sqrt();
            let draw = m + self.rng.normal() * spread * 0.5;
            if draw > best_v {
                best_v = draw;
                best = i;
            }
        }
        Policy::ALL[best]
    }

    pub fn credit(&mut self, policy: Policy, reward: f64) {
        let i = Policy::ALL.iter().position(|p| *p == policy).unwrap_or(0);
        let r = reward.clamp(0.0, 1.0);
        self.alpha[i] += r;
        self.beta[i] += 1.0 - r;
        self.pending[i] += r;
        let best = self
            .alpha
            .iter()
            .zip(self.beta.iter())
            .map(|(a, b)| a / (a + b))
            .fold(0.0f64, f64::max);
        let chosen = self.alpha[i] / (self.alpha[i] + self.beta[i]);
        self.regret = self.regret * 0.999 + (best - chosen).max(0.0) * 0.001;
    }

    /// Decay keeps the posterior from freezing after a regime change: an expert that was
    /// right for an hour should not outvote fresh evidence forever.
    pub fn decay(&mut self, factor: f64) {
        for i in 0..6 {
            self.alpha[i] = 1.0 + (self.alpha[i] - 1.0) * factor;
            self.beta[i] = 1.0 + (self.beta[i] - 1.0) * factor;
        }
    }

    pub fn mixture_map(&self) -> serde_json::Value {
        let w = self.mixture();
        let mut m = serde_json::Map::new();
        for (i, p) in Policy::ALL.iter().enumerate() {
            m.insert(p.as_str().to_string(), serde_json::json!(round4(w[i])));
        }
        serde_json::Value::Object(m)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum Regime {
    Steady,
    FlashCrowd,
    Scan,
    Shifting,
    Expensive,
    Growing,
}

impl Regime {
    pub fn as_str(self) -> &'static str {
        match self {
            Regime::Steady => "Steady",
            Regime::FlashCrowd => "FlashCrowd",
            Regime::Scan => "Scan",
            Regime::Shifting => "Shifting",
            Regime::Expensive => "Expensive",
            Regime::Growing => "Growing",
        }
    }
}

/// Regime detection from six cheap streaming statistics. No model, on purpose: this has to
/// run on every request without showing up in the latency budget.
#[derive(Debug, Default, Clone, Copy)]
pub struct WorkloadFeatures {
    pub burstiness: f64,
    pub entropy: f64,
    pub working_set_growth: f64,
    pub reuse_distance_p50: f64,
    pub popularity_shift: f64,
    pub scan_score: f64,
}

impl WorkloadFeatures {
    pub fn classify(&self) -> (Regime, f64) {
        let mut scores = [
            (Regime::Steady, 0.35),
            (Regime::FlashCrowd, self.burstiness / 4.0 + (8.0 - self.entropy).max(0.0) / 8.0),
            (Regime::Scan, self.scan_score * 1.6),
            (Regime::Shifting, self.popularity_shift * 2.2),
            (Regime::Expensive, 0.0),
            (Regime::Growing, self.working_set_growth * 2.0),
        ];
        for s in scores.iter_mut() {
            s.1 = s.1.max(0.0);
        }
        let total: f64 = scores.iter().map(|s| s.1).sum();
        let best = scores
            .iter()
            .copied()
            .fold((Regime::Steady, 0.0), |a, b| if b.1 > a.1 { b } else { a });
        let confidence = if total > 0.0 { best.1 / total } else { 0.0 };
        (best.0, confidence.clamp(0.0, 1.0))
    }
}

pub fn round4(v: f64) -> f64 {
    (v * 10_000.0).round() / 10_000.0
}
