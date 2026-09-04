use aura_core::rng::{Rng, Zipf};
use aura_core::types::{CostVector, KeyId, ObjectContext, SlaClass};

use crate::attack::{ActiveAttack, Attack};
use crate::scenario::{Scenario, ScenarioSpec};

/// One synthetic request.
#[derive(Debug, Clone)]
pub struct Request {
    pub ts_ms: f64,
    pub key_id: KeyId,
    pub context: ObjectContext,
    pub regen_latency_ms: f64,
}

const APPS: [&str; 3] = ["recommendation", "analytics", "content"];
const OBJECT_TYPES: [&str; 3] = ["ranking_result", "aggregate_query", "media_variant"];

#[derive(Debug)]
pub struct Generator {
    spec: ScenarioSpec,
    scenario: Scenario,
    rng: Rng,
    zipf: Zipf,
    now_ms: f64,
    hot_offset: usize,
    attacks: Vec<ActiveAttack>,
    scan_cursor: u64,
    emitted: u64,
}

impl Generator {
    pub fn new(scenario: Scenario, seed: u64) -> Self {
        let spec = scenario.spec();
        let zipf = Zipf::new(spec.unique_keys, spec.zipf_alpha);
        Self {
            spec,
            scenario,
            rng: Rng::seed_from_u64(seed),
            zipf,
            now_ms: 0.0,
            hot_offset: 0,
            attacks: Vec::new(),
            scan_cursor: 0,
            emitted: 0,
        }
    }

    pub fn scenario(&self) -> Scenario {
        self.scenario
    }

    pub fn spec(&self) -> &ScenarioSpec {
        &self.spec
    }

    pub fn now_ms(&self) -> f64 {
        self.now_ms
    }

    pub fn emitted(&self) -> u64 {
        self.emitted
    }

    pub fn inject(&mut self, attack: Attack, duration_s: f64) {
        self.attacks.push(ActiveAttack {
            attack,
            started_s: self.now_ms / 1000.0,
            duration_s,
        });
    }

    pub fn live_attacks(&self) -> Vec<Attack> {
        let now_s = self.now_ms / 1000.0;
        self.attacks
            .iter()
            .filter(|a| a.is_live(now_s))
            .map(|a| a.attack)
            .collect()
    }

    /// Requests per second right now, after any live disturbance.
    pub fn rps(&self) -> f64 {
        let mut rps = self.spec.base_rps;
        let now_s = self.now_ms / 1000.0;
        for a in self.attacks.iter().filter(|a| a.is_live(now_s)) {
            match a.attack {
                Attack::FlashCrowd => rps *= 1.0 + 2.5 * bump(a.progress(now_s)),
                Attack::MixedChaos => rps *= 1.0 + 1.2 * bump(a.progress(now_s)),
                Attack::WorkingSetExplosion => rps *= 1.25,
                _ => {}
            }
        }
        rps
    }

    /// Advance virtual time and return the requests that fall in the window.
    pub fn step(&mut self, dt_ms: f64) -> Vec<Request> {
        let count = ((self.rps() * dt_ms) / 1000.0).round() as usize;
        let mut out = Vec::with_capacity(count);
        let per = if count > 0 { dt_ms / count as f64 } else { 0.0 };
        for i in 0..count {
            let ts = self.now_ms + per * i as f64;
            out.push(self.one(ts));
        }
        self.now_ms += dt_ms;
        self.retire_attacks();
        out
    }

    /// Pull a fixed number of requests, used by the offline benchmark where wall clock
    /// does not matter but the arrival spacing still has to be believable.
    pub fn take(&mut self, n: usize) -> Vec<Request> {
        let mut out = Vec::with_capacity(n);
        let gap = 1000.0 / self.spec.base_rps;
        for _ in 0..n {
            let ts = self.now_ms;
            out.push(self.one(ts));
            self.now_ms += gap;
            if self.emitted % 20_000 == 0 {
                self.rotate_scripted_attacks();
            }
        }
        out
    }

    fn retire_attacks(&mut self) {
        let now_s = self.now_ms / 1000.0;
        self.attacks.retain(|a| now_s < a.started_s + a.duration_s + 5.0);
    }

    /// The offline benchmark replays each scenario's own disturbances so that all policies
    /// meet the same sequence of events.
    fn rotate_scripted_attacks(&mut self) {
        if self.spec.attacks.is_empty() {
            return;
        }
        let idx = (self.emitted / 20_000) as usize % self.spec.attacks.len();
        let attack = self.spec.attacks[idx];
        self.inject(attack, 8.0);
        if attack == Attack::PopularityShift {
            self.hot_offset = (self.hot_offset + self.spec.unique_keys / 7) % self.spec.unique_keys;
        }
    }

    fn one(&mut self, ts_ms: f64) -> Request {
        self.emitted += 1;
        let now_s = ts_ms / 1000.0;
        let live: Vec<ActiveAttack> = self
            .attacks
            .iter()
            .copied()
            .filter(|a| a.is_live(now_s))
            .collect();

        let key_id = self.pick_key(&live);
        let app_idx = self.pick_app(key_id);
        let (mut size_bytes, mut regen) = self.cost_shape(app_idx, key_id);

        for a in &live {
            match a.attack {
                Attack::CostSpike => {
                    regen.cpu_ms *= 6.0;
                    regen.db_ms *= 8.0;
                    regen.api_cost_usd *= 5.0;
                }
                Attack::ExpensiveTail => {
                    // Rarity and cost are deliberately correlated here: the long tail is
                    // where the money is, which is exactly what hit-rate-only policies miss.
                    let rarity = (key_id as f64 / self.spec.unique_keys as f64).clamp(0.0, 1.0);
                    let mult = 1.0 + 14.0 * rarity.powf(1.6);
                    regen.db_ms *= mult;
                    regen.gpu_ms *= mult;
                }
                Attack::WorkingSetExplosion => size_bytes = (size_bytes as f64 * 1.8) as u64,
                _ => {}
            }
        }

        let regen_latency_ms = regen.cpu_ms + regen.gpu_ms + regen.db_ms
            + self.rng.range(0.0, 12.0)
            + if regen.api_calls > 0.0 { 45.0 } else { 0.0 };

        let sla = match app_idx {
            0 => SlaClass::High,
            1 => SlaClass::Normal,
            _ => SlaClass::Low,
        };
        let ttl_ms = match app_idx {
            0 => 300_000.0,
            1 => 900_000.0,
            _ => 3_600_000.0,
        };

        let context = ObjectContext::new(APPS[app_idx], OBJECT_TYPES[app_idx], size_bytes)
            .with_regen(regen)
            .with_ttl(ttl_ms)
            .with_sla(sla);

        Request {
            ts_ms,
            key_id,
            context,
            regen_latency_ms,
        }
    }

    fn pick_key(&mut self, live: &[ActiveAttack]) -> KeyId {
        let n = self.spec.unique_keys;
        for a in live {
            match a.attack {
                Attack::Scan => {
                    if self.rng.bool_with(0.55) {
                        // Every scan key is new, which is the point: admitting any of them
                        // is pure loss.
                        self.scan_cursor = self.scan_cursor.wrapping_add(1);
                        return n as u64 + self.scan_cursor;
                    }
                }
                Attack::FlashCrowd => {
                    if self.rng.bool_with(0.75) {
                        return (self.hot_offset as u64 + self.rng.below(24)) % n as u64;
                    }
                }
                Attack::HotKeyEmergence => {
                    if self.rng.bool_with(0.45) {
                        return (n - 1) as u64;
                    }
                }
                Attack::MixedChaos => {
                    if self.rng.bool_with(0.25) {
                        self.scan_cursor = self.scan_cursor.wrapping_add(1);
                        return n as u64 + self.scan_cursor;
                    }
                    if self.rng.bool_with(0.35) {
                        return (self.hot_offset as u64 + self.rng.below(64)) % n as u64;
                    }
                }
                Attack::WorkingSetExplosion => {
                    if self.rng.bool_with(0.4) {
                        return self.rng.below(n as u64 * 3);
                    }
                }
                _ => {}
            }
        }
        let rank = self.zipf.sample(&mut self.rng);
        ((rank + self.hot_offset) % n) as u64
    }

    fn pick_app(&mut self, key_id: KeyId) -> usize {
        match self.scenario {
            Scenario::MixedProduction => (key_id % 3) as usize,
            Scenario::ExpensiveTail => {
                if key_id % 4 == 0 {
                    1
                } else {
                    (key_id % 3) as usize
                }
            }
            _ => (key_id % 3) as usize,
        }
    }

    /// Each application has a different cost signature.
    fn cost_shape(&mut self, app_idx: usize, key_id: KeyId) -> (u64, CostVector) {
        let jitter = self.rng.range(0.8, 1.25);
        match app_idx {
            0 => {
                let size = (self.rng.log_normal(11.6, 0.5) as u64).clamp(4_096, 6_000_000);
                (
                    size,
                    CostVector {
                        cpu_ms: 180.0 * jitter,
                        gpu_ms: 70.0 * jitter,
                        db_ms: 60.0 * jitter,
                        network_bytes: size as f64,
                        api_calls: 0.0,
                        api_cost_usd: 0.0,
                        latency_ms: 0.0,
                    },
                )
            }
            1 => {
                let size = (self.rng.log_normal(10.2, 0.7) as u64).clamp(2_048, 2_000_000);
                let scan_factor = 1.0 + (key_id % 11) as f64 * 0.35;
                (
                    size,
                    CostVector {
                        cpu_ms: 40.0 * jitter,
                        gpu_ms: 0.0,
                        db_ms: 220.0 * jitter * scan_factor,
                        network_bytes: size as f64,
                        api_calls: 0.0,
                        api_cost_usd: 0.0,
                        latency_ms: 0.0,
                    },
                )
            }
            _ => {
                let size = (self.rng.log_normal(13.4, 0.6) as u64).clamp(32_768, 24_000_000);
                (
                    size,
                    CostVector {
                        cpu_ms: 60.0 * jitter,
                        gpu_ms: 210.0 * jitter,
                        db_ms: 10.0,
                        network_bytes: size as f64,
                        api_calls: 1.0,
                        api_cost_usd: 0.0021,
                        latency_ms: 0.0,
                    },
                )
            }
        }
    }
}

/// Ramp up, hold, ramp down. Disturbances that switch on instantly are easy to detect and
/// prove nothing about adaptation.
fn bump(progress: f64) -> f64 {
    if progress < 0.2 {
        progress / 0.2
    } else if progress > 0.8 {
        (1.0 - progress) / 0.2
    } else {
        1.0
    }
}
