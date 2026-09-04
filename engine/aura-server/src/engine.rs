use std::collections::VecDeque;

use ahash::AHashMap;
use aura_core::config::Config;
use aura_core::features::{idx, AccessEvent, AmbientState, FeatureBuilder, Features};
use aura_core::rng::Rng;
use aura_core::types::{Action, CostVector, Decision, KeyId, ObjectContext};
use serde::Serialize;

use crate::audit::{self, AuditKind, AuditLog, Fact, Severity};
use crate::feedback::{Journal, JournalStats, TrainingRow};
use crate::policy::{round4, Bandit, Policy, Regime, WorkloadFeatures};
use crate::predictor::Predictor;
use crate::store::{Entry, Store};

#[derive(Debug, Clone, Serialize)]
pub struct ExplainRecord {
    pub t: f64,
    pub key: String,
    pub application: String,
    pub action: String,
    pub reason_code: String,
    pub reuse_probability: ReuseProbability,
    pub economic_value_usd: f64,
    pub value_density: f64,
    pub eviction_threshold: f64,
    pub contributions: Vec<Contribution>,
    pub reasons: Vec<String>,
    pub predictor: String,
    pub predictor_confidence: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ReuseProbability {
    pub h10s: f64,
    pub h60s: f64,
    pub h600s: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct Contribution {
    pub feature: String,
    pub weight: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct EngineEvent {
    pub t: f64,
    pub kind: String,
    pub detail: String,
}

#[derive(Debug, Default, Clone)]
pub struct AppStats {
    pub requests: u64,
    pub hits: u64,
    pub bytes_total: u64,
    pub regen_ms_total: f64,
    pub regen_count: u64,
    pub regen_samples: Vec<f64>,
    pub reuse_gaps: Vec<f64>,
    pub cost_usd: f64,
    pub allocated_bytes: u64,
    pub policy_credit: [f64; 6],
}

#[derive(Debug, Default, Clone, Copy)]
pub struct CostLedger {
    pub backend_usd: f64,
    pub cache_usd: f64,
    pub sla_penalty_usd: f64,
    pub no_cache_usd: f64,
}

impl CostLedger {
    pub fn total(&self) -> f64 {
        self.backend_usd + self.cache_usd + self.sla_penalty_usd
    }
}

/// A shadow cache that replays the same request stream under one classical policy.
#[derive(Debug)]
pub struct Shadow {
    pub policy: Policy,
    resident: AHashMap<KeyId, (u64, f64, f64, f64)>,
    used_bytes: u64,
    capacity_bytes: u64,
    pub hits: u64,
    pub misses: u64,
    pub cost_usd: f64,
    pub penalty_usd: f64,
    pub holding_usd: f64,
}

impl Shadow {
    pub fn new(policy: Policy, capacity_bytes: u64) -> Self {
        Self {
            policy,
            resident: AHashMap::new(),
            used_bytes: 0,
            capacity_bytes,
            hits: 0,
            misses: 0,
            cost_usd: 0.0,
            penalty_usd: 0.0,
            holding_usd: 0.0,
        }
    }

    pub fn total_usd(&self) -> f64 {
        self.cost_usd + self.penalty_usd + self.holding_usd
    }

    pub fn set_capacity(&mut self, bytes: u64) {
        self.capacity_bytes = bytes.max(1);
    }

    pub fn hit_rate(&self) -> f64 {
        let t = self.hits + self.misses;
        if t == 0 {
            0.0
        } else {
            self.hits as f64 / t as f64
        }
    }

    pub fn access(
        &mut self,
        key: KeyId,
        size: u64,
        cost_usd: f64,
        now_ms: f64,
        freq_hint: f64,
        penalty_usd: f64,
        holding_usd: f64,
    ) {
        match self.resident.get_mut(&key) {
            Some(slot) => {
                self.hits += 1;
                slot.1 = now_ms;
                slot.3 += 1.0;
                return;
            }
            None => {
                self.misses += 1;
                self.cost_usd += cost_usd;
                self.penalty_usd += penalty_usd;
                self.holding_usd += holding_usd;
            }
        }
        if size > self.capacity_bytes {
            return;
        }
        while self.used_bytes + size > self.capacity_bytes {
            let victim = self
                .resident
                .iter()
                .map(|(k, v)| (*k, self.rank(v, now_ms)))
                .fold(None::<(KeyId, f64)>, |acc, cur| match acc {
                    Some(a) if a.1 <= cur.1 => Some(a),
                    _ => Some(cur),
                });
            match victim {
                Some((k, _)) => {
                    if let Some(v) = self.resident.remove(&k) {
                        self.used_bytes = self.used_bytes.saturating_sub(v.0);
                    }
                }
                None => break,
            }
        }
        self.resident.insert(key, (size, now_ms, cost_usd, freq_hint.max(1.0)));
        self.used_bytes += size;
    }

    fn rank(&self, v: &(u64, f64, f64, f64), now_ms: f64) -> f64 {
        let (size, last_ms, cost, freq) = *v;
        match self.policy {
            Policy::Lru => last_ms,
            Policy::Lfu => freq,
            Policy::Gdsf => (freq * cost.max(1e-9)) / (size as f64).max(1.0) * 1e9,
            Policy::TinyLfu => freq * 0.8 + last_ms / 1e6,
            Policy::CostAware => cost.max(1e-9) / (size as f64).max(1.0) * 1e9,
            Policy::TrendAware => freq / (1.0 + (now_ms - last_ms) / 1000.0),
        }
    }
}

#[derive(Debug)]
pub struct Engine {
    pub cfg: Config,
    pub store: Store,
    pub features: FeatureBuilder,
    pub predictor: Predictor,
    pub bandit: Bandit,
    /// Decisions awaiting the future that will judge them. See `feedback.rs`.
    pub journal: Journal,
    /// What the cache did, in words. See `audit.rs`.
    pub audit: AuditLog,
    pub ledger: CostLedger,
    pub shadows: Vec<Shadow>,
    pub apps: AHashMap<String, AppStats>,
    pub explains: VecDeque<ExplainRecord>,
    pub events: VecDeque<EngineEvent>,
    pub latencies: VecDeque<f64>,
    pub workload: WorkloadFeatures,
    pub regime: Regime,
    pub regime_confidence: f64,
    pub override_policy: Option<Policy>,
    pub eviction_threshold: f64,
    pub requests: u64,
    pub refreshes: u64,
    pub rejections: u64,
    pub decision_overhead_us: VecDeque<f64>,
    rng: Rng,
    recent_keys: VecDeque<KeyId>,
    seen_keys: AHashMap<KeyId, f64>,
    last_hot: Vec<KeyId>,
    unique_last_window: usize,
    single_touch: u64,
    novel_in_window: u64,
    window_requests: u64,
    inter_arrivals: VecDeque<f64>,
    last_request_ms: f64,
    pub now_ms: f64,
}

impl Engine {
    pub fn new(cfg: Config, seed: u64) -> Self {
        let capacity = cfg.cache.capacity_bytes;
        let l1_window = ((capacity / 32_768).max(1_024) as usize).min(200_000);
        let store = Store::new(capacity, l1_window);
        let features = FeatureBuilder::new(cfg.features, cfg.pricing);
        let predictor = Predictor::heuristic(cfg.predictor.online_lr);
        let bandit = Bandit::new(cfg.bandit.exploration, seed ^ 0x9E37);
        let shadows = [Policy::Lru, Policy::Lfu, Policy::Gdsf]
            .iter()
            .map(|p| Shadow::new(*p, capacity))
            .collect();
        Self {
            cfg,
            store,
            features,
            predictor,
            bandit,
            journal: Journal::new(),
            audit: AuditLog::default(),
            ledger: CostLedger::default(),
            shadows,
            apps: AHashMap::new(),
            explains: VecDeque::with_capacity(256),
            events: VecDeque::with_capacity(128),
            latencies: VecDeque::with_capacity(4_096),
            workload: WorkloadFeatures::default(),
            regime: Regime::Steady,
            regime_confidence: 0.0,
            override_policy: None,
            eviction_threshold: 0.0,
            requests: 0,
            refreshes: 0,
            rejections: 0,
            decision_overhead_us: VecDeque::with_capacity(2_048),
            rng: Rng::seed_from_u64(seed),
            recent_keys: VecDeque::with_capacity(8_192),
            seen_keys: AHashMap::new(),
            last_hot: Vec::new(),
            unique_last_window: 0,
            single_touch: 0,
            novel_in_window: 0,
            window_requests: 0,
            inter_arrivals: VecDeque::with_capacity(1_024),
            last_request_ms: 0.0,
            now_ms: 0.0,
        }
    }

    pub fn set_capacity(&mut self, bytes: u64) {
        let before = self.store.capacity_bytes();
        self.store.set_capacity(bytes);
        for s in self.shadows.iter_mut() {
            s.set_capacity(bytes);
        }
        self.push_event(
            "ScaleUp",
            &format!("{} -> {}", human_bytes(before), human_bytes(bytes)),
        );
        self.enforce_capacity();
    }

    /// The read path.
    pub fn get(&mut self, key: KeyId, application: &str, now_ms: f64) -> Option<Entry> {
        self.now_ms = now_ms.max(self.now_ms);
        self.requests += 1;
        self.window_requests += 1;
        self.track_arrival(key, now_ms);
        // The future arriving for decisions taken earlier. A miss counts as reuse just as
        // much as a hit: the label answers "was this wanted again", and counting only hits
        // would teach the model that whatever the cache kept was the right thing to keep.
        self.journal.observe_access(key, now_ms);

        let l1_seen = self.store.touch_l1(key, now_ms);
        self.store.l1_stats.record(l1_seen, 0);

        let expired = self
            .store
            .get(key)
            .map(|e| e.expired(now_ms))
            .unwrap_or(false);
        if expired {
            self.store.remove(key);
        }

        let entry = match self.store.entry_mut(key) {
            Some(e) => {
                e.hits += 1;
                e.last_hit_ms = now_ms;
                e.clone()
            }
            None => {
                let app = application.to_string();
                let st = self.apps.entry(app).or_default();
                st.requests += 1;
                self.store.l2_stats.record(false, 0);
                return None;
            }
        };

        let bytes = entry.size_bytes;
        self.store.l2_stats.record(true, bytes);
        let saved = self.cfg.pricing.regen_cost_usd(&entry.regen);
        self.ledger.no_cache_usd += saved;

        let st = self.apps.entry(entry.application.clone()).or_default();
        st.requests += 1;
        st.hits += 1;
        st.bytes_total += bytes;

        let freq_hint = self
            .features
            .key_state(key)
            .map(|k| k.accesses as f64)
            .unwrap_or(1.0);
        let hold = self
            .cfg
            .pricing
            .holding_cost_usd(bytes as f64, entry.ttl_ms.max(60_000.0).min(600_000.0));
        let latency = entry.regen.cpu_ms + entry.regen.gpu_ms + entry.regen.db_ms;
        let penalty = if latency > self.cfg.pricing.slo_p95_ms {
            self.cfg.pricing.sla_penalty_usd(
                latency - self.cfg.pricing.slo_p95_ms,
                aura_core::types::SlaClass::Normal.penalty_weight(),
            )
        } else {
            0.0
        };
        for s in self.shadows.iter_mut() {
            s.access(key, bytes, saved, now_ms, freq_hint, penalty, hold);
        }

        self.observe_reuse(key, now_ms, true);
        Some(entry)
    }

    /// The write path, and the only place a decision is made. `measured` is the cost the
    /// application actually paid to rebuild the object, not an estimate.
    pub fn put(
        &mut self,
        key: KeyId,
        value: serde_json::Value,
        ctx: &ObjectContext,
        measured: CostVector,
        now_ms: f64,
    ) -> (Decision, Vec<KeyId>) {
        let t0 = std::time::Instant::now();
        self.now_ms = now_ms.max(self.now_ms);

        let regen = if measured.is_empty() { ctx.regen } else { measured };
        let cost_usd = self.cfg.pricing.regen_cost_usd(&regen);
        let latency = if regen.latency_ms > 0.0 {
            regen.latency_ms
        } else {
            regen.cpu_ms + regen.gpu_ms + regen.db_ms
        };

        self.store.l2_stats.miss_bytes += ctx.size_bytes;
        self.ledger.backend_usd += cost_usd;
        self.ledger.no_cache_usd += cost_usd;
        if latency > self.cfg.pricing.slo_p95_ms {
            self.ledger.sla_penalty_usd += self
                .cfg
                .pricing
                .sla_penalty_usd(latency - self.cfg.pricing.slo_p95_ms, ctx.sla_class.penalty_weight());
        }
        self.push_latency(latency);

        {
            let st = self.apps.entry(ctx.application.clone()).or_default();
            st.regen_ms_total += latency;
            st.regen_count += 1;
            st.cost_usd += cost_usd;
            st.bytes_total += ctx.size_bytes;
            if st.regen_samples.len() < 4_096 {
                st.regen_samples.push(latency);
            }
        }

        let freq_hint = self
            .features
            .key_state(key)
            .map(|k| k.accesses as f64)
            .unwrap_or(1.0);
        let shadow_penalty = if latency > self.cfg.pricing.slo_p95_ms {
            self.cfg.pricing.sla_penalty_usd(
                latency - self.cfg.pricing.slo_p95_ms,
                ctx.sla_class.penalty_weight(),
            )
        } else {
            0.0
        };
        let shadow_holding = self
            .cfg
            .pricing
            .holding_cost_usd(ctx.size_bytes as f64, ctx.ttl_ms.unwrap_or(60_000.0).min(600_000.0));
        for s in self.shadows.iter_mut() {
            s.access(
                key,
                ctx.size_bytes,
                cost_usd,
                now_ms,
                freq_hint,
                shadow_penalty,
                shadow_holding,
            );
        }

        let ambient = AmbientState {
            cache_pressure: self.store.pressure(),
            ttl_remaining_frac: self
                .store
                .get(key)
                .map(|e| e.ttl_remaining_frac(now_ms))
                .unwrap_or(0.0),
        };
        let event = AccessEvent {
            ts_ms: now_ms,
            key_id: key,
            application: ctx.application.clone(),
            object_type: ctx.object_type.clone(),
            size_bytes: ctx.size_bytes,
            ttl_ms: ctx.ttl_ms.unwrap_or(0.0),
            regen,
            regen_latency_ms: latency,
        };
        let f = self.features.transform(&event, ambient);

        let reuse = self.predictor.reuse(&f);
        let value_density = self.value_density(&f, reuse, ctx.size_bytes, cost_usd);
        // The bar for admission is the object that would have to be evicted to make room,
        // not a running average of what has been arriving. Comparing an arrival against
        // the mean of other arrivals rejects the bottom half of the stream no matter how
        // valuable it is, and leaves capacity unused.
        let victim_bar = self.victim_bar(ctx.size_bytes, now_ms);
        let threshold = victim_bar * self.cfg.engine.admission_margin;
        self.eviction_threshold = victim_bar;

        let too_big = ctx.size_bytes > self.store.capacity_bytes() / 4;
        let scan_suspect = self.regime == Regime::Scan
            && reuse[0] < 0.15
            && !self.store.touch_l1(key, now_ms);

        let (admit, reason_code) = if too_big {
            (false, "object_exceeds_quarter_capacity")
        } else if scan_suspect {
            (false, "single_touch_scan_signature")
        } else if self.store.is_empty() || value_density >= threshold {
            (true, "density_above_threshold")
        } else if reuse[1] > 0.75 {
            (true, "high_near_term_reuse")
        } else {
            (false, "density_below_threshold")
        };

        let mut evicted = Vec::new();
        let decision = if admit {
            evicted = self.make_room(ctx.size_bytes, value_density, now_ms);
            if self.store.fits(ctx.size_bytes) {
                self.store.insert(Entry {
                    key,
                    value,
                    size_bytes: ctx.size_bytes,
                    application: ctx.application.clone(),
                    object_type: ctx.object_type.clone(),
                    ttl_ms: ctx.ttl_ms.unwrap_or(0.0),
                    regen,
                    inserted_ms: now_ms,
                    last_hit_ms: now_ms,
                    hits: 0,
                    score: value_density,
                });
                self.ledger.cache_usd += self.cfg.pricing.holding_cost_usd(
                    ctx.size_bytes as f64,
                    ctx.ttl_ms.unwrap_or(60_000.0).min(600_000.0),
                );
                Decision::new(Action::Admit, "density_above_threshold")
            } else {
                self.rejections += 1;
                self.store.rejections += 1;
                Decision::new(Action::Reject, "could_not_free_enough")
            }
        } else {
            self.rejections += 1;
            self.store.rejections += 1;
            Decision::new(Action::Reject, reason_code)
        };

        self.write_audit(key, ctx, &decision, reuse, cost_usd, value_density, threshold, &regen, evicted.len());
        self.record_explain(key, ctx, &decision, &f, reuse, cost_usd, value_density);
        self.observe_reuse(key, now_ms, false);

        // The decision is written down and left alone. Crediting the bandit here, from the
        // score that produced the decision, would be the system grading its own homework:
        // it would learn that its own confidence was justified, whatever actually happened.
        // `settle_feedback` credits it sixty seconds later, from the outcome.
        let credited = self.override_policy.unwrap_or_else(|| self.bandit.sample_arm());
        let holding_usd = self
            .cfg
            .pricing
            .holding_cost_usd(ctx.size_bytes as f64, crate::feedback::HORIZONS_MS[1]);
        self.journal.record(
            key,
            &ctx.application,
            f,
            reuse,
            credited,
            decision.action == Action::Admit,
            value_density,
            threshold,
            cost_usd,
            holding_usd,
            ctx.size_bytes,
            now_ms,
        );

        let us = t0.elapsed().as_secs_f64() * 1e6;
        if self.decision_overhead_us.len() >= 2_048 {
            self.decision_overhead_us.pop_front();
        }
        self.decision_overhead_us.push_back(us);

        (decision, evicted)
    }

    /// Economic value per byte per second of occupancy. This is the single number every
    /// admission and eviction is decided on, and the only place cost, reuse and size meet.
    pub fn value_density(&self, f: &Features, reuse: [f64; 3], size_bytes: u64, cost_usd: f64) -> f64 {
        let w = self.cfg.engine.horizon_weights;
        let expected_reuses = reuse[0] * w[0] * 6.0 + reuse[1] * w[1] * 2.0 + reuse[2] * w[2];
        let sla_weight = 1.0 + f[idx::COST_VARIANCE_RATIO].min(4.0) * self.cfg.engine.tail_risk_lambda;
        let value = cost_usd * expected_reuses * sla_weight;
        let hold = self
            .cfg
            .pricing
            .holding_cost_usd(size_bytes as f64, 60_000.0)
            .max(1e-12);
        value / hold
    }

    /// Sampled eviction. Scanning every resident object to find the true minimum is not
    /// affordable at request rate, and sampling 32 gets within a few percent of it.
    fn make_room(&mut self, needed: u64, incoming_density: f64, now_ms: f64) -> Vec<KeyId> {
        let mut evicted = Vec::new();
        if self.store.fits(needed) {
            return evicted;
        }
        self.store.sweep_expired(now_ms);
        let sample_n = self.cfg.engine.candidate_sample.max(4);
        let mut guard = 0;
        while !self.store.fits(needed) && guard < 4_096 {
            guard += 1;
            let keys = self.store.keys();
            if keys.is_empty() {
                break;
            }
            let mut worst: Option<(KeyId, f64)> = None;
            for _ in 0..sample_n.min(keys.len()) {
                let k = keys[self.rng.below(keys.len() as u64) as usize];
                let d = self.resident_density(k, now_ms);
                if worst.map(|w| d < w.1).unwrap_or(true) {
                    worst = Some((k, d));
                }
            }
            match worst {
                Some((k, d)) if d <= incoming_density * 1.05 => {
                    self.store.evict(k);
                    evicted.push(k);
                }
                _ => break,
            }
        }
        evicted
    }

    /// Value density of an object already resident, on the *same scale* as
    /// [`Engine::value_density`].
    ///
    /// These two were previously computed by different formulas in different units, which
    /// silently disabled admission control: an arrival scored in the thousands was always
    /// going to clear a bar scored below one, so nothing was ever refused. Both sides now
    /// end in the same expression, and the only thing the expert blend and the model do is
    /// supply the reuse estimate that expression consumes.
    fn resident_density(&self, key: KeyId, now_ms: f64) -> f64 {
        let e = match self.store.get(key) {
            Some(e) => e,
            None => return 0.0,
        };
        let ambient = AmbientState {
            cache_pressure: self.store.pressure(),
            ttl_remaining_frac: e.ttl_remaining_frac(now_ms),
        };
        let f = self.features.peek(
            key,
            now_ms,
            &e.application,
            &e.object_type,
            e.size_bytes,
            &e.regen,
            ambient,
        );
        let age = e.age_ms(now_ms);

        // An operator forcing a single policy wants that policy's ranking verbatim, not a
        // blend, so this path returns before any of the economics.
        if let Some(p) = self.override_policy {
            return p.utility(&f, age);
        }

        let cost = self.cfg.pricing.regen_cost_usd(&e.regen);
        let mixture = self.bandit.mixture();
        let expert: f64 = Policy::ALL
            .iter()
            .enumerate()
            .map(|(i, p)| mixture[i] * p.utility(&f, age))
            .sum();
        // The experts rank objects but do not speak probabilities. Squashing turns the
        // blended score into something on the same [0, 1) footing as the model output so
        // the two can be mixed at all.
        let expert_reuse = expert / (1.0 + expert.max(0.0));
        let model_reuse = self.predictor.reuse_peek(&f);
        let share = self
            .cfg
            .engine
            .ml_confidence_floor
            .max(self.predictor.confidence())
            .clamp(0.0, 1.0);

        let blended = |model: f64| expert_reuse * (1.0 - share) + model * share;
        let reuse = [blended(model_reuse[0]), blended(model_reuse[1]), blended(model_reuse[2])];

        self.value_density(&f, reuse, e.size_bytes, cost) * e.ttl_remaining_frac(now_ms).max(0.05)
    }

    /// Density of the weakest resident object a sample can find.
    fn victim_bar(&mut self, incoming_bytes: u64, now_ms: f64) -> f64 {
        if self.store.fits(incoming_bytes) {
            return 0.0;
        }
        let keys = self.store.keys();
        if keys.is_empty() {
            return 0.0;
        }
        let n = self.cfg.engine.candidate_sample.max(4).min(keys.len());
        let picks: Vec<KeyId> = (0..n)
            .map(|_| keys[self.rng.below(keys.len() as u64) as usize])
            .collect();
        picks
            .into_iter()
            .map(|k| self.resident_density(k, now_ms))
            .fold(f64::INFINITY, f64::min)
    }

    pub fn enforce_capacity(&mut self) {
        let now = self.now_ms;
        let mut guard = 0;
        while self.store.used_bytes() > self.store.capacity_bytes() && guard < 100_000 {
            guard += 1;
            let keys = self.store.keys();
            if keys.is_empty() {
                break;
            }
            let mut worst: Option<(KeyId, f64)> = None;
            for _ in 0..16.min(keys.len()) {
                let k = keys[self.rng.below(keys.len() as u64) as usize];
                let d = self.resident_density(k, now);
                if worst.map(|w| d < w.1).unwrap_or(true) {
                    worst = Some((k, d));
                }
            }
            match worst {
                Some((k, _)) => {
                    self.store.evict(k);
                }
                None => break,
            }
        }
    }

    /// Objects close to expiry that are still valuable are rebuilt before anyone asks, so
    /// the expiry is never paid for on a user request.
    pub fn refresh_candidates(&mut self, limit: usize) -> Vec<KeyId> {
        let now = self.now_ms;
        let thr = self.cfg.engine.refresh_ttl_threshold;
        let mut out: Vec<(KeyId, f64)> = self
            .store
            .keys()
            .iter()
            .copied()
            .filter(|k| {
                self.store
                    .get(*k)
                    .map(|e| e.ttl_ms > 0.0 && e.ttl_remaining_frac(now) < thr)
                    .unwrap_or(false)
            })
            .map(|k| (k, self.resident_density(k, now)))
            .collect();
        out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        out.truncate(limit);
        self.refreshes += out.len() as u64;
        out.into_iter().map(|(k, _)| k).collect()
    }

    fn record_explain(
        &mut self,
        key: KeyId,
        ctx: &ObjectContext,
        decision: &Decision,
        f: &Features,
        reuse: [f64; 3],
        cost_usd: f64,
        density: f64,
    ) {
        let mut reasons = Vec::new();
        if reuse[1] > 0.6 {
            reasons.push("high predicted near-term reuse".to_string());
        } else if reuse[1] < 0.2 {
            reasons.push("little evidence this key returns".to_string());
        }
        if cost_usd > 1e-5 {
            reasons.push(format!("regeneration priced at ${cost_usd:.6}"));
        }
        if self.eviction_threshold > 0.0 {
            let ratio = density / self.eviction_threshold.max(1e-9);
            reasons.push(format!("value density {ratio:.2}x the current eviction bar"));
        } else {
            reasons.push("cache below contention, admission is free".to_string());
        }
        if self.regime == Regime::Scan {
            reasons.push("workload currently classified as a scan".to_string());
        }

        let rec = ExplainRecord {
            t: self.now_ms,
            key: format!("{}:{}", ctx.application, key),
            application: ctx.application.clone(),
            action: decision.action.as_str().to_string(),
            reason_code: decision.reason_code.to_string(),
            reuse_probability: ReuseProbability {
                h10s: round4(reuse[0]),
                h60s: round4(reuse[1]),
                h600s: round4(reuse[2]),
            },
            economic_value_usd: cost_usd,
            value_density: round4(density),
            eviction_threshold: round4(self.eviction_threshold),
            contributions: self
                .predictor
                .contributions(f)
                .into_iter()
                .map(|(feature, weight)| Contribution {
                    feature,
                    weight: round4(weight),
                })
                .collect(),
            reasons,
            predictor: self.predictor.kind().as_str().to_string(),
            predictor_confidence: round4(self.predictor.confidence()),
        };
        if self.explains.len() >= 200 {
            self.explains.pop_front();
        }
        self.explains.push_back(rec);
    }

    /// Close the loop: judge decisions whose sixty-second verdict is now in.
    ///
    /// This is the only place the bandit is rewarded and the only place the online model is
    /// corrected. Both learn from what actually happened rather than from what was
    /// predicted, which is the difference between a system that adapts and one that
    /// confirms its own priors.
    pub fn settle_feedback(&mut self, limit: usize) -> usize {
        let settled = self.journal.settle(self.now_ms, limit);
        for s in &settled {
            self.bandit.credit(s.policy, s.reward);
            // The 60-second label is the one that is both meaningful and available quickly
            // enough to steer a live cache.
            self.predictor.observe(&s.features, s.reused[1]);
        }
        // Retire fully matured records into training rows for the next model.
        self.journal.retire(self.now_ms, limit * 4);
        settled.len()
    }

    pub fn journal_stats(&self) -> JournalStats {
        self.journal.stats()
    }

    /// Drain finished training rows. `GET /v1/training/rows` serves these, and the trainer
    /// reads them: the cache produces its own training data as a by-product of running.
    pub fn drain_training_rows(&mut self, n: usize) -> Vec<TrainingRow> {
        self.journal.drain_completed(n)
    }

    /// Turn one decision into a sentence, with the numbers that produced it.
    ///
    /// Kept out of `put` so the hot path reads as a decision procedure rather than a
    /// logging routine, and so the sampling policy lives in one place.
    #[allow(clippy::too_many_arguments)]
    fn write_audit(
        &mut self,
        key: KeyId,
        ctx: &ObjectContext,
        decision: &Decision,
        reuse: [f64; 3],
        cost_usd: f64,
        density: f64,
        threshold: f64,
        regen: &CostVector,
        evicted: usize,
    ) {
        let name = format!("{}:{}", ctx.application, key);
        let facts = vec![
            Fact::new("size", audit::bytes(ctx.size_bytes)),
            Fact::new("reuse_60s", audit::percent(reuse[1])),
            Fact::new("rebuild_cost", audit::usd(cost_usd)),
            Fact::new("value_density", format!("{density:.2}")),
            Fact::new("bar", format!("{threshold:.2}")),
        ];

        match decision.action {
            Action::Admit => {
                // An expensive object is worth a line of its own even when routine
                // admissions are being sampled away.
                let severity = if cost_usd > 0.001 { Severity::Notice } else { Severity::Info };
                let message = audit::say::admitted(
                    &name, reuse[1], cost_usd, regen.cpu_ms, regen.gpu_ms, regen.db_ms,
                    regen.api_cost_usd, density, threshold, evicted,
                );
                self.audit.record(
                    self.now_ms, AuditKind::Admit, severity, name, &ctx.application,
                    message, facts, cost_usd,
                );
            }
            Action::Reject => {
                // Refusing something that turns out to be expensive is the decision most
                // worth reviewing, so it is never sampled away.
                let severity = if cost_usd > 0.001 { Severity::Notice } else { Severity::Info };
                let message = audit::say::rejected(
                    &name, decision.reason_code, ctx.size_bytes, reuse[1], density, threshold,
                );
                self.audit.record(
                    self.now_ms, AuditKind::Reject, severity, name, &ctx.application,
                    message, facts, -cost_usd,
                );
            }
            _ => {}
        }
    }

    /// Record a capacity decision in words, including the decision *not* to scale — which
    /// is the one a reader is most likely to doubt.
    pub fn audit_capacity(
        &mut self,
        from: u64,
        to: u64,
        delta_hit: f64,
        savings_hr: f64,
        rent_hr: f64,
        roi: f64,
        threshold: f64,
    ) {
        let (kind, message) = if to == from {
            (AuditKind::ScaleHold, audit::say::held(from, to, roi, threshold))
        } else if to > from {
            (AuditKind::ScaleUp, audit::say::scaled(from, to, delta_hit, savings_hr, rent_hr))
        } else {
            (AuditKind::ScaleDown, audit::say::scaled(from, to, delta_hit, savings_hr, rent_hr))
        };
        let facts = vec![
            Fact::new("from", audit::bytes(from)),
            Fact::new("to", audit::bytes(to)),
            Fact::new("delta_hit_rate", format!("{:.1} points", delta_hit * 100.0)),
            Fact::new("savings", audit::usd_per_hour(savings_hr)),
            Fact::new("rent", audit::usd_per_hour(rent_hr)),
            Fact::new("roi", format!("{roi:.2}x")),
        ];
        let severity = if kind == AuditKind::ScaleHold { Severity::Info } else { Severity::Notice };
        self.audit.record(
            self.now_ms, kind, severity, "capacity", "engine", message, facts,
            savings_hr - rent_hr,
        );
    }

    pub fn audit_regime(&mut self, from: &str, to: &str, confidence: f64) {
        let message = audit::say::regime_changed(from, to, confidence);
        self.audit.record(
            self.now_ms, AuditKind::RegimeChange, Severity::Notice, to, "engine", message,
            vec![Fact::new("confidence", audit::percent(confidence))], 0.0,
        );
    }

    pub fn audit_model(&mut self, name: &str, features: usize, source: &str, auc: Option<f64>) {
        let message = audit::say::model_loaded(name, features, source, auc);
        self.audit.record(
            self.now_ms, AuditKind::ModelLoad, Severity::Notice, name, "engine", message,
            vec![Fact::new("features", features.to_string()), Fact::new("source", source)], 0.0,
        );
    }

    pub fn explain_key(&self, key: KeyId) -> Option<&ExplainRecord> {
        self.explains.iter().rev().find(|r| r.key.ends_with(&format!(":{key}")))
    }

    pub fn push_event(&mut self, kind: &str, detail: &str) {
        if self.events.len() >= 100 {
            self.events.pop_front();
        }
        self.events.push_back(EngineEvent {
            t: self.now_ms,
            kind: kind.to_string(),
            detail: detail.to_string(),
        });
    }

    fn push_latency(&mut self, ms: f64) {
        if self.latencies.len() >= 4_096 {
            self.latencies.pop_front();
        }
        self.latencies.push_back(ms);
    }

    pub fn latency_quantile(&self, q: f64) -> f64 {
        if self.latencies.is_empty() {
            return 0.0;
        }
        let mut v: Vec<f64> = self.latencies.iter().copied().collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let i = ((v.len() - 1) as f64 * q).round() as usize;
        v[i]
    }

    pub fn overhead_p50(&self) -> f64 {
        if self.decision_overhead_us.is_empty() {
            return 0.0;
        }
        let mut v: Vec<f64> = self.decision_overhead_us.iter().copied().collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        v[v.len() / 2]
    }

    fn observe_reuse(&mut self, key: KeyId, now_ms: f64, hit: bool) {
        let prev = self.seen_keys.insert(key, now_ms);
        if let Some(t) = prev {
            let gap = now_ms - t;
            if self.inter_arrivals.len() >= 1_024 {
                self.inter_arrivals.pop_front();
            }
            self.inter_arrivals.push_back(gap);
        } else if !hit {
            self.single_touch += 1;
            self.novel_in_window += 1;
        }
        if self.seen_keys.len() > 400_000 {
            let cutoff = now_ms - 600_000.0;
            self.seen_keys.retain(|_, v| *v > cutoff);
        }
    }

    fn track_arrival(&mut self, key: KeyId, now_ms: f64) {
        if self.recent_keys.len() >= 8_192 {
            self.recent_keys.pop_front();
        }
        self.recent_keys.push_back(key);
        self.last_request_ms = now_ms;
    }

    /// Recomputed on the controller tick rather than per request. Every statistic here is
    /// derived from the last window only, so a regime change shows up within one window.
    pub fn recompute_workload(&mut self) {
        let n = self.recent_keys.len();
        if n < 64 {
            return;
        }
        let mut counts: AHashMap<KeyId, u64> = AHashMap::new();
        for k in self.recent_keys.iter() {
            *counts.entry(*k).or_insert(0) += 1;
        }
        let unique = counts.len();
        let total = n as f64;
        let entropy: f64 = counts
            .values()
            .map(|c| {
                let p = *c as f64 / total;
                -p * p.log2()
            })
            .sum();

        let mut ranked: Vec<(KeyId, u64)> = counts.into_iter().collect();
        ranked.sort_by(|a, b| b.1.cmp(&a.1));
        let hot: Vec<KeyId> = ranked.iter().take(32).map(|(k, _)| *k).collect();
        let overlap = hot
            .iter()
            .filter(|k| self.last_hot.contains(k))
            .count() as f64;
        let popularity_shift = if self.last_hot.is_empty() {
            0.0
        } else {
            1.0 - overlap / hot.len().max(1) as f64
        };
        self.last_hot = hot;

        let top_share = ranked
            .iter()
            .take(16)
            .map(|(_, c)| *c as f64)
            .sum::<f64>()
            / total;
        let novel_rate = if self.window_requests > 0 {
            self.novel_in_window as f64 / self.window_requests as f64
        } else {
            0.0
        };
        let repeat_rate = 1.0 - (unique as f64 / total);
        let scan_score = (novel_rate * (1.0 - repeat_rate)).clamp(0.0, 1.0);

        let growth = if self.unique_last_window > 0 {
            (unique as f64 - self.unique_last_window as f64) / self.unique_last_window as f64
        } else {
            0.0
        };
        self.unique_last_window = unique;

        let mut gaps: Vec<f64> = self.inter_arrivals.iter().copied().collect();
        gaps.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let p50 = if gaps.is_empty() { 0.0 } else { gaps[gaps.len() / 2] };

        self.workload = WorkloadFeatures {
            burstiness: (top_share * 8.0).clamp(0.0, 8.0),
            entropy,
            working_set_growth: growth.clamp(-1.0, 4.0),
            reuse_distance_p50: p50,
            popularity_shift: popularity_shift.clamp(0.0, 1.0),
            scan_score: scan_score.clamp(0.0, 1.0),
        };
        let (regime, conf) = self.workload.classify();
        if regime != self.regime {
            self.push_event("PolicyShift", &format!("{} -> {}", self.regime.as_str(), regime.as_str()));
        }
        self.regime = regime;
        self.regime_confidence = conf;
        self.bandit.decay(0.995);
        self.window_requests = 0;
        self.novel_in_window = 0;
    }
}

pub fn human_bytes(b: u64) -> String {
    const U: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];
    let mut v = b as f64;
    let mut i = 0;
    while v >= 1024.0 && i < 4 {
        v /= 1024.0;
        i += 1;
    }
    format!("{v:.0} {}", U[i])
}
