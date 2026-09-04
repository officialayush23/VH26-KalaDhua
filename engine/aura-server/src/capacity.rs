use aura_core::config::Config;
use serde::Serialize;

use crate::engine::{human_bytes, Engine};

#[derive(Debug, Clone, Serialize)]
pub struct MarginalStep {
    pub from_bytes: u64,
    pub to_bytes: u64,
    pub delta_hit_rate: f64,
    pub backend_savings_usd_hr: f64,
    pub cache_cost_usd_hr: f64,
    pub net_usd_hr: f64,
    pub verdict: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct CapacityReport {
    pub mode: String,
    pub logical_bytes: u64,
    pub recommended_bytes: u64,
    pub host_budget_bytes: u64,
    pub host_available_bytes: u64,
    pub decision: String,
    pub marginal: Vec<MarginalStep>,
    pub reason: String,
    pub nodes: u32,
    pub provider: String,
    pub pressure: f64,
    pub used_bytes: u64,
    pub mrc: Vec<MrcPoint>,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct MrcPoint {
    pub bytes: u64,
    pub hit_rate: f64,
}

/// Capacity control asks one question: does the next block of memory pay for itself?
/// The miss-ratio curve gives the hit-rate gain, the cost model prices it, and the answer
/// is a number in dollars per hour rather than a utilisation threshold.
#[derive(Debug)]
pub struct CapacityController {
    pub manual: bool,
    pub last_change_ms: f64,
    pub last_decision: String,
}

impl CapacityController {
    pub fn new(cfg: &Config) -> Self {
        Self {
            manual: !cfg.capacity.auto,
            last_change_ms: -1e12,
            last_decision: "Hold".to_string(),
        }
    }

    /// Concave saturating fit anchored on the measured hit rate. It is an estimate and is
    /// labelled as one, but it is anchored to live data at the current size rather than
    /// invented wholesale.
    pub fn mrc(&self, engine: &Engine) -> Vec<MrcPoint> {
        let cur = engine.store.capacity_bytes().max(1) as f64;
        let hr = engine.store.l2_stats.hit_rate().clamp(0.01, 0.99);
        let k = cur / (1.0 / hr - 1.0).max(1e-6);
        let mut pts = Vec::new();
        for mult in [0.25f64, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0] {
            let b = (cur * mult) as u64;
            let h = (b as f64 / (b as f64 + k)).clamp(0.0, 0.995);
            pts.push(MrcPoint {
                bytes: b,
                hit_rate: (h * 10_000.0).round() / 10_000.0,
            });
        }
        pts
    }

    pub fn report(&self, engine: &Engine, cfg: &Config) -> CapacityReport {
        let cur = engine.store.capacity_bytes();
        let step = cfg.capacity.step_bytes;
        let next = (cur + step).min(cfg.capacity.max_bytes);
        let mrc = self.mrc(engine);

        let hr_now = interp(&mrc, cur);
        let hr_next = interp(&mrc, next);
        let delta = (hr_next - hr_now).max(0.0);

        // Requests per hour times the average price of a miss gives what the extra hit
        // rate is worth. That is the only figure that justifies buying memory.
        let elapsed_s = (engine.now_ms / 1000.0).max(1.0);
        let rph = engine.requests as f64 / elapsed_s * 3600.0;
        let avg_miss_usd = if engine.store.l2_stats.misses > 0 {
            engine.ledger.backend_usd / engine.store.l2_stats.misses as f64
        } else {
            0.0
        };
        let savings = delta * rph * avg_miss_usd;
        let extra_gb = (next - cur) as f64 / 1e9;
        let cache_cost = extra_gb * cfg.pricing.cache_gb_hour_usd;
        let net = savings - cache_cost;
        let roi = if cache_cost > 0.0 { savings / cache_cost } else { 0.0 };

        let pressure = engine.store.pressure();
        let host_available = cfg
            .capacity
            .host_budget_bytes
            .saturating_sub(engine.store.used_bytes());

        let (decision, reason) = if self.manual {
            ("Hold".to_string(), "capacity is under manual control".to_string())
        } else if roi >= cfg.capacity.roi_threshold && next > cur && host_available > step {
            (
                "ScaleUp".to_string(),
                format!(
                    "marginal ROI {roi:.2}x above the {:.2}x bar; {} free on host",
                    cfg.capacity.roi_threshold,
                    human_bytes(host_available)
                ),
            )
        } else if roi >= cfg.capacity.roi_threshold && host_available <= step {
            (
                "ScaleOut".to_string(),
                "more memory pays for itself but this host has none left".to_string(),
            )
        } else if pressure < 0.55 && cur > cfg.capacity.min_bytes {
            (
                "ScaleDown".to_string(),
                format!("only {:.0}% of the pool is in use; the rest is rent for nothing", pressure * 100.0),
            )
        } else {
            (
                "Hold".to_string(),
                format!("marginal ROI {roi:.2}x is below the {:.2}x bar", cfg.capacity.roi_threshold),
            )
        };

        let recommended = match decision.as_str() {
            "ScaleUp" => next,
            "ScaleDown" => cur.saturating_sub(step).max(cfg.capacity.min_bytes),
            _ => cur,
        };

        CapacityReport {
            mode: if self.manual { "manual".into() } else { "auto".into() },
            logical_bytes: cur,
            recommended_bytes: recommended,
            host_budget_bytes: cfg.capacity.host_budget_bytes,
            host_available_bytes: host_available,
            decision,
            marginal: vec![MarginalStep {
                from_bytes: cur,
                to_bytes: next,
                delta_hit_rate: (delta * 10_000.0).round() / 10_000.0,
                backend_savings_usd_hr: round2(savings),
                cache_cost_usd_hr: round2(cache_cost),
                net_usd_hr: round2(net),
                verdict: if net > 0.0 { "profitable".into() } else { "not worth it".into() },
            }],
            reason,
            nodes: 1,
            provider: "local".into(),
            pressure: (pressure * 10_000.0).round() / 10_000.0,
            used_bytes: engine.store.used_bytes(),
            mrc,
        }
    }

    /// Applies the recommendation, subject to a cooldown so the controller cannot
    /// oscillate through a transient burst.
    pub fn maybe_apply(&mut self, engine: &mut Engine, cfg: &Config) -> Option<CapacityReport> {
        let report = self.report(engine, cfg);
        if self.manual {
            return Some(report);
        }
        let cooldown_ms = cfg.capacity.cooldown_s * 1000.0;
        if engine.now_ms - self.last_change_ms < cooldown_ms {
            return Some(report);
        }
        if report.recommended_bytes != report.logical_bytes {
            engine.set_capacity(report.recommended_bytes);
            self.last_change_ms = engine.now_ms;
            self.last_decision = report.decision.clone();
        }
        Some(report)
    }
}

fn interp(mrc: &[MrcPoint], bytes: u64) -> f64 {
    if mrc.is_empty() {
        return 0.0;
    }
    let b = bytes as f64;
    for w in mrc.windows(2) {
        let (a, c) = (w[0], w[1]);
        if b >= a.bytes as f64 && b <= c.bytes as f64 {
            let span = (c.bytes - a.bytes).max(1) as f64;
            let t = (b - a.bytes as f64) / span;
            return a.hit_rate + (c.hit_rate - a.hit_rate) * t;
        }
    }
    mrc.last().map(|p| p.hit_rate).unwrap_or(0.0)
}

fn round2(v: f64) -> f64 {
    (v * 100.0).round() / 100.0
}
