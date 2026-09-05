use std::collections::VecDeque;

use ahash::AHashMap;
use aura_core::config::Config;
use aura_core::features::{idx, AccessEvent, AmbientState, FeatureBuilder, Features, EXTRA_OFFSET};
use aura_core::signals::SignalBuilder;
use aura_core::rng::Rng;
use aura_core::policies::CachePolicy;
use aura_core::types::{Action, CostVector, Decision, KeyId, ObjectContext};
use serde::Serialize;

use crate::audit::{self, AuditKind, AuditLog, Fact, Severity};
use crate::consistency::{
    Consistency, ConsistencyStats, Freshness, InvalidationMode, InvalidationResult, Removal,
};
use crate::feedback::{Journal, JournalStats, TrainingRow};
use crate::policy::{round4, Bandit, Policy, Regime, WorkloadFeatures};
use crate::predictor::Predictor;
use crate::profiles::ProfileStore;
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

/// Predicted against realised, for one class of decision.
///
/// A single calibration number over every decision hides the asymmetry that matters. Being
/// over-confident about objects the cache *kept* wastes memory; being over-confident about
/// objects it *refused* forces rebuilds it could have avoided. Those are different failures
/// with different costs, and averaging them together can look healthy while both are bad in
/// opposite directions.
#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct Calibration {
    pub predicted: f64,
    pub realised: f64,
    pub n: u64,
}

impl Calibration {
    fn observe(&mut self, predicted: f64, realised: bool) {
        self.predicted += predicted;
        self.realised += if realised { 1.0 } else { 0.0 };
        self.n += 1;
    }

    pub fn mean_predicted(&self) -> f64 {
        if self.n == 0 { 0.0 } else { self.predicted / self.n as f64 }
    }

    pub fn mean_realised(&self) -> f64 {
        if self.n == 0 { 0.0 } else { self.realised / self.n as f64 }
    }

    /// Positive means over-optimistic.
    pub fn error(&self) -> f64 {
        self.mean_predicted() - self.mean_realised()
    }
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
    pub cost_usd: f64,
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

/// A baseline running beside the engine on the same request stream.
///
/// This used to be a scoring function: a hash map plus a `match` that approximated each
/// policy in one line. That is fine for a sketch and wrong for a claim. W-TinyLFU, S3-FIFO
/// and SIEVE are *defined* by their structure rather than by a score, and even LFU differs
/// from "evict the lowest counter" once admission and size-awareness are involved. So the
/// live baselines now run the same implementations the offline benchmark runs, from
/// `aura_core::policies`, and the two can no longer disagree about what LRU is.
///
/// Cost accounting stays here rather than in the policy: the policies answer "what happens
/// to this object", and pricing a miss is the engine's question, using the engine's price
/// table so that every column in the comparison is priced identically.
pub struct Shadow {
    pub name: &'static str,
    policy: Box<dyn CachePolicy>,
    pub hits: u64,
    pub misses: u64,
    /// Bytes served from the baseline's own cache, and bytes it had to fetch. Object hit
    /// rate alone cannot separate a policy that keeps a thousand small cheap objects from
    /// one that keeps a hundred expensive ones, and that distinction is the whole argument.
    pub hit_bytes: u64,
    pub miss_bytes: u64,
    pub cost_usd: f64,
    pub penalty_usd: f64,
    pub holding_usd: f64,
}

impl std::fmt::Debug for Shadow {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Shadow")
            .field("name", &self.name)
            .field("hits", &self.hits)
            .field("misses", &self.misses)
            .finish()
    }
}

impl Shadow {
    /// Build a baseline by name. Returns `None` for a name the policy roster does not know,
    /// so a typo is a missing column rather than a silently different policy.
    pub fn new(name: &'static str, capacity_bytes: u64) -> Option<Self> {
        let policy = aura_core::policies::build(name, capacity_bytes)?;
        Some(Self {
            name,
            policy,
            hits: 0,
            misses: 0,
            hit_bytes: 0,
            miss_bytes: 0,
            cost_usd: 0.0,
            penalty_usd: 0.0,
            holding_usd: 0.0,
        })
    }

    pub fn total_usd(&self) -> f64 {
        self.cost_usd + self.penalty_usd + self.holding_usd
    }

    pub fn set_capacity(&mut self, bytes: u64) {
        self.policy.set_capacity_bytes(bytes.max(1));
    }

    pub fn used_bytes(&self) -> u64 {
        self.policy.used_bytes()
    }

    pub fn hit_rate(&self) -> f64 {
        let t = self.hits + self.misses;
        if t == 0 {
            0.0
        } else {
            self.hits as f64 / t as f64
        }
    }

    pub fn byte_hit_rate(&self) -> f64 {
        let t = self.hit_bytes + self.miss_bytes;
        if t == 0 {
            0.0
        } else {
            self.hit_bytes as f64 / t as f64
        }
    }

    /// One request, exactly as the engine saw it. A baseline that never learns the object's
    /// size, TTL or rebuild cost cannot be said to have declined to use them.
    #[allow(clippy::too_many_arguments)]
    pub fn access(
        &mut self,
        key: KeyId,
        size: u64,
        cost_usd: f64,
        now_ms: f64,
        ttl_ms: f64,
        regen: CostVector,
        penalty_usd: f64,
        holding_usd: f64,
    ) {
        let req = aura_core::policies::Request {
            ts_ms: now_ms,
            key,
            size_bytes: size,
            ttl_ms,
            regen,
            application: "shadow",
            object_type: "object",
            sla: aura_core::types::SlaClass::Normal,
        };
        let result = self.policy.access(&req);
        if result.hit {
            self.hits += 1;
            self.hit_bytes += size;
        } else {
            self.misses += 1;
            self.miss_bytes += size;
            self.cost_usd += cost_usd;
            self.penalty_usd += penalty_usd;
            self.holding_usd += holding_usd;
        }
    }
}

#[derive(Debug)]
pub struct Engine {
    pub cfg: Config,
    pub store: Store,
    /// What each application asked the cache to optimise for. Read on every write; the
    /// lookup falls back to the global defaults, so an unconfigured application costs
    /// nothing extra.
    pub profiles: ProfileStore,
    /// Bytes each application currently holds. Maintained at the two places residency
    /// changes rather than counted on demand, because counting it means walking the pool.
    pub resident_bytes: AHashMap<String, u64>,
    pub features: FeatureBuilder,
    /// The eight extra signals. Separate from `features` because they carry
    /// per-application distributions rather than per-key counters.
    pub signals: SignalBuilder,
    pub predictor: Predictor,
    pub bandit: Bandit,
    /// Decisions awaiting the future that will judge them. See `feedback.rs`.
    pub journal: Journal,
    /// What the cache did, in words. See `audit.rs`.
    pub audit: AuditLog,
    /// Dependency tags, namespace versions and the freshness rule. See `consistency.rs`.
    /// Checked *before* value on every read: being fast is an optimisation, being right
    /// is not.
    pub consistency: Consistency,
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
    /// Reads answered from a value that was past its soft threshold. Each one is a request
    /// that did *not* wait for a rebuild, and each one is also a small admission of
    /// staleness, so it is counted rather than hidden inside the hit rate.
    pub stale_serves: u64,
    /// Objects whose rebuild is owed but not yet done. Draining this is what makes refresh
    /// real rather than a clock reset.
    pub refresh_queue: VecDeque<KeyId>,
    pub decision_overhead_us: VecDeque<f64>,
    /// Calibration for objects the cache kept, and for objects it refused.
    pub calib_kept: Calibration,
    pub calib_refused: Calibration,
    /// The policy mixture at the last audit, so a shift can be described as a change
    /// rather than a level.
    last_mixture: [f64; 6],
    last_pressure_note_ms: f64,
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
        let signals = SignalBuilder::new(2_000, cfg.features.quantile_lr);
        let predictor = Predictor::heuristic(cfg.predictor.online_lr);
        let bandit = Bandit::new(cfg.bandit.exploration, seed ^ 0x9E37);
        let consistency = Consistency::new(cfg.engine.soft_ttl_fraction);
        // The three the brief names, and only those. The full roster - GDSF, W-TinyLFU,
        // S3-FIFO, SIEVE, LeCaR and the Belady bound - runs in the benchmark, where a
        // controlled replay can say something meaningful about it. A live shadow is a
        // different measurement and does not belong in the same sentence as those.
        let shadows = ["lru", "lfu", "gds"]
            .into_iter()
            .filter_map(|name| Shadow::new(name, capacity))
            .collect();
        Self {
            profiles: ProfileStore::new(&cfg),
            resident_bytes: AHashMap::new(),
            cfg,
            store,
            features,
            signals,
            predictor,
            bandit,
            journal: Journal::new(),
            audit: AuditLog::default(),
            consistency,
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
            stale_serves: 0,
            refresh_queue: VecDeque::with_capacity(256),
            decision_overhead_us: VecDeque::with_capacity(2_048),
            calib_kept: Calibration::default(),
            calib_refused: Calibration::default(),
            last_mixture: [1.0 / 6.0; 6],
            last_pressure_note_ms: -1e12,
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
            if bytes >= before { "ScaleUp" } else { "ScaleDown" },
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

        // Validity before value. Everything below this point is an optimisation; this is
        // the part that is not negotiable, so it runs first and it does not consult the
        // model, the bandit or the economics. A cache that serves a price of $40 after the
        // database says $20 is not a fast cache, it is a wrong one.
        let snapshot = self
            .store
            .get(key)
            .map(|e| (e.inserted_ms, e.ttl_ms, e.size_bytes, e.application.clone()));
        if let Some((inserted_ms, ttl_ms, size_bytes, app)) = snapshot {
            match self.consistency.freshness(key, inserted_ms, ttl_ms, now_ms) {
                Freshness::Expired => {
                    self.drop_key(key, Removal::Expired, 0.0, 0.0);
                }
                Freshness::Stale => {
                    // Serve it once and rebuild behind the reader. Making this request wait
                    // would be correct and slow; making the *next* thousand requests wait,
                    // which is what expiring a popular key does, is neither.
                    self.stale_serves += 1;
                    self.consistency.note_stale_serve();
                    self.queue_refresh(key);
                    if self.stale_serves % 64 == 1 {
                        let remaining = if ttl_ms > 0.0 {
                            (1.0 - (now_ms - inserted_ms) / ttl_ms).clamp(0.0, 1.0)
                        } else {
                            0.0
                        };
                        let name = format!("{app}:{key}");
                        let message = format!(
                            "Served {name} while it was stale — {} of its lifetime left, and a \
                             rebuild is queued. The reader got an answer immediately instead of \
                             paying for the rebuild, and the herd behind it will get the new value.",
                            audit::percent(remaining)
                        );
                        self.audit.record(
                            self.now_ms,
                            AuditKind::Refresh,
                            Severity::Info,
                            name,
                            &app,
                            message,
                            vec![
                                Fact::new("size", audit::bytes(size_bytes)),
                                Fact::new("ttl_remaining", audit::percent(remaining)),
                            ],
                            0.0,
                        );
                    }
                }
                Freshness::Fresh => {}
            }
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
            s.access(key, bytes, saved, now_ms, entry.ttl_ms, entry.regen, penalty, hold);
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
            // Charged at this application's own weight: an application that says being slow
            // hurts it more is billed for it, which is what makes the knob mean something
            // rather than being a display preference.
            let weight =
                ctx.sla_class.penalty_weight() * self.profiles.get(&ctx.application).sla_weight;
            self.ledger.sla_penalty_usd +=
                self.cfg.pricing.sla_penalty_usd(latency - self.cfg.pricing.slo_p95_ms, weight);
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
                ctx.ttl_ms.unwrap_or(0.0),
                regen,
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
        let mut f = self.features.transform(&event, ambient);
        // The extra signals need the same read-then-fold discipline, so they are computed
        // from the same event at the same instant rather than reconstructed later.
        let extra = self.signals.transform(
            key,
            now_ms,
            &ctx.application,
            ctx.size_bytes,
            cost_usd,
            latency,
        );
        f[EXTRA_OFFSET..].copy_from_slice(&extra);

        // Priced the same way a resident is priced. An arrival has no age, so the experts
        // see it for what it is rather than crediting it with time served.
        let model_reuse = self.predictor.reuse(&f);
        let reuse = self.blended_reuse(&f, 0.0, model_reuse);
        let value_density = self.value_density(&f, reuse, ctx.size_bytes, cost_usd, &ctx.application);
        // The bar for admission is the object that would have to be evicted to make room,
        // not a running average of what has been arriving. Comparing an arrival against
        // the mean of other arrivals rejects the bottom half of the stream no matter how
        // valuable it is, and leaves capacity unused.
        let victim_bar = self.victim_bar(ctx.size_bytes, now_ms);
        let profile = *self.profiles.get(&ctx.application);
        // An application over its share of the pool does not stop being admitted; it has to
        // be better to get in. A hard cap would evict an expensive object to make room for a
        // worthless one belonging to a quieter application, which is not fairness.
        let share_penalty = {
            let cap = self.store.capacity_bytes();
            let held = *self.resident_bytes.get(&ctx.application).unwrap_or(&0);
            if cap > 0 && profile.max_pool_share < 1.0 {
                let share = held as f64 / cap as f64;
                if share > profile.max_pool_share {
                    1.0 + (share - profile.max_pool_share) * 4.0
                } else {
                    1.0
                }
            } else {
                1.0
            }
        };
        let threshold = victim_bar * profile.admission_margin * share_penalty;
        self.eviction_threshold = victim_bar;

        let too_big = ctx.size_bytes > self.store.capacity_bytes() / 4;
        // The regime detector flaps between Steady and Scan at low confidence on ordinary
        // traffic, and refusing admissions on a 33%-confident guess costs hit rate for
        // nothing. Below half confidence the classification is not evidence.
        let scan_suspect = self.regime == Regime::Scan
            && self.regime_confidence >= 0.5
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
                *self.resident_bytes.entry(ctx.application.clone()).or_insert(0) += ctx.size_bytes;
                self.ledger.cache_usd += self.cfg.pricing.holding_cost_usd(
                    ctx.size_bytes as f64,
                    ctx.ttl_ms.unwrap_or(60_000.0).min(600_000.0),
                );
                // The object is now on the hook for whatever it was derived from, and any
                // stale mark it was carrying is settled: this *is* the rebuild.
                self.consistency.register(key, &ctx.depends_on);
                self.consistency.mark_rebuilt(key);
                self.refresh_queue.retain(|k| *k != key);
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
    pub fn value_density(
        &self,
        f: &Features,
        reuse: [f64; 3],
        size_bytes: u64,
        cost_usd: f64,
        application: &str,
    ) -> f64 {
        // The profile shapes the arithmetic around the prediction, never the prediction
        // itself: `reuse` arrives here exactly as the model produced it.
        let profile = self.profiles.get(application);
        let w = profile.horizon_weights;
        let expected_reuses = reuse[0] * w[0] * 6.0 + reuse[1] * w[1] * 2.0 + reuse[2] * w[2];
        let risk = 1.0 + f[idx::COST_VARIANCE_RATIO].min(4.0) * profile.tail_risk_lambda;
        let sla_weight = risk * profile.sla_weight.max(0.05);
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
        self.sweep_expired(now_ms);
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
                    self.drop_key(k, Removal::Evicted, d, incoming_density);
                    evicted.push(k);
                }
                _ => break,
            }
        }
        evicted
    }

    /// Remove one object and account for *why*.
    ///
    /// Eviction, invalidation and expiry are three different events and the telemetry must
    /// not merge them: eviction means the cache was short of space, invalidation means it
    /// was wrong, expiry means time passed. A dashboard showing one number for all three
    /// hides the only one of the three that indicates a bug.
    fn drop_key(&mut self, key: KeyId, reason: Removal, density: f64, incoming_density: f64) {
        let removed = match reason {
            Removal::Evicted => self.store.evict(key),
            _ => self.store.remove(key),
        };
        let Some(e) = removed else { return };
        if let Some(held) = self.resident_bytes.get_mut(&e.application) {
            *held = held.saturating_sub(e.size_bytes);
        }
        if reason == Removal::Expired {
            self.store.expirations += 1;
        }
        self.consistency.forget(key);
        self.consistency.note_removal(reason);
        self.refresh_queue.retain(|k| *k != key);

        let cost = self.cfg.pricing.regen_cost_usd(&e.regen);
        let name = format!("{}:{}", e.application, key);
        let facts = vec![
            Fact::new("size", audit::bytes(e.size_bytes)),
            Fact::new("hits_while_resident", e.hits.to_string()),
            Fact::new("rebuild_cost", audit::usd(cost)),
            Fact::new("reason", reason.as_str()),
            // What the cache believed when it took the object in, beside what it believed
            // when it let it go. A wide gap is the model having been wrong, and it should
            // be visible rather than inferred.
            Fact::new("value_at_admission", format!("{:.2}", e.score)),
            Fact::new("value_at_removal", format!("{density:.2}")),
        ];
        if reason == Removal::Invalidated {
            // Counted and forgotten above; the summary line in `invalidate` is the right
            // granularity. Writing one entry per key would bury the event in its own
            // consequences.
            return;
        }
        let (kind, severity, message) = match reason {
            Removal::Evicted => (
                AuditKind::Evict,
                // Throwing away something expensive is worth a line even when routine
                // evictions are being sampled away.
                if cost > 0.001 { Severity::Notice } else { Severity::Info },
                audit::say::evicted(&name, e.size_bytes, density, incoming_density, cost),
            ),
            Removal::Expired => (
                AuditKind::Expire,
                Severity::Info,
                format!(
                    "Dropped {name}: its {} lifetime ran out before anyone asked for it again. \
                     It was served {} time{} while resident.",
                    audit::ms(e.ttl_ms),
                    e.hits,
                    if e.hits == 1 { "" } else { "s" }
                ),
            ),
            // Handled above.
            Removal::Invalidated => return,
        };
        self.audit.record(
            self.now_ms, kind, severity, name, &e.application, message, facts, -cost,
        );
    }

    /// Drop everything past its hard TTL, forgetting its dependencies as it goes.
    ///
    /// This goes through [`Engine::drop_key`] rather than `Store::sweep_expired` so that an
    /// expiry cannot leave a dangling tag behind. A tag pointing at a key that no longer
    /// exists is not merely untidy: the next invalidation matches it, reports work it did
    /// not do, and the real object built from that row survives.
    pub fn sweep_expired(&mut self, now_ms: f64) -> usize {
        let dead: Vec<KeyId> = self
            .store
            .keys()
            .iter()
            .copied()
            .filter(|k| self.store.get(*k).map(|e| e.expired(now_ms)).unwrap_or(false))
            .collect();
        let n = dead.len();
        for k in dead {
            self.drop_key(k, Removal::Expired, 0.0, 0.0);
        }
        n
    }

    fn queue_refresh(&mut self, key: KeyId) {
        if self.refresh_queue.len() >= 4_096 || self.refresh_queue.contains(&key) {
            return;
        }
        self.refresh_queue.push_back(key);
    }

    /// Value density of an object already resident, on the *same scale* as
    /// [`Engine::value_density`].
    ///
    /// These two were previously computed by different formulas in different units, which
    /// silently disabled admission control: an arrival scored in the thousands was always
    /// going to clear a bar scored below one, so nothing was ever refused. Both sides now
    /// end in the same expression, and the only thing the expert blend and the model do is
    /// supply the reuse estimate that expression consumes.
    /// Reuse probability as the engine actually believes it: the policy experts, weighted
    /// by the bandit, blended with the model according to how much the model has earned.
    ///
    /// This exists as one function because it must be applied identically to an arrival and
    /// to a resident. It was not. Residents were priced with this blend while arrivals were
    /// priced by the model alone, and the experts reward accumulated hits, so every resident
    /// was flattered against every newcomer. The pool filled once and then refused almost
    /// everything: 83% of arrivals rejected, and a 644 KB object with a 98% reuse
    /// probability turned away because "nothing resident was worth less than it". The two
    /// numbers being compared were not measuring the same thing.
    fn blended_reuse(&self, f: &Features, age_ms: f64, model: [f64; 3]) -> [f64; 3] {
        let mixture = self.bandit.mixture();
        let expert: f64 = Policy::ALL
            .iter()
            .enumerate()
            .map(|(i, p)| mixture[i] * p.utility(f, age_ms))
            .sum();
        // The experts rank objects but do not speak probabilities. Squashing puts the
        // blended score on the same [0, 1) footing as the model output so the two can be
        // mixed at all.
        let expert_reuse = expert / (1.0 + expert.max(0.0));
        let share = self
            .cfg
            .engine
            .ml_confidence_floor
            .max(self.predictor.confidence())
            .clamp(0.0, 1.0);
        let blend = |m: f64| expert_reuse * (1.0 - share) + m * share;
        [blend(model[0]), blend(model[1]), blend(model[2])]
    }

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
        let reuse = self.blended_reuse(&f, age, self.predictor.reuse_peek(&f));

        self.value_density(&f, reuse, e.size_bytes, cost, &e.application)
            * e.ttl_remaining_frac(now_ms).max(0.05)
    }

    /// The reuse estimate for a resident object, without recording an access.
    ///
    /// `peek` rather than `transform` matters: scoring an object must never look like the
    /// object was requested, or the act of considering something for eviction would make it
    /// appear popular and save it.
    pub fn reuse_peek(&self, key: KeyId, now_ms: f64) -> Option<[f64; 3]> {
        let e = self.store.get(key)?;
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
        Some(self.predictor.reuse_peek(&f))
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
                Some((k, d)) => {
                    self.drop_key(k, Removal::Evicted, d, 0.0);
                }
                None => break,
            }
        }
    }

    /// Objects close to expiry that are still valuable are queued for rebuild before anyone
    /// asks, so the expiry is never paid for on a user request.
    ///
    /// This only *nominates*. It deliberately does not touch `inserted_ms`: resetting the
    /// clock on bytes nobody rebuilt is the one thing a refresh controller must never do,
    /// because it turns a stale object into a permanently fresh-looking stale object. The
    /// rebuild itself happens in [`Engine::rebuild`], or in the application that drains
    /// this queue over `GET /v1/refresh/queue`.
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
        // Most valuable first: if only some of them can be rebuilt this tick, the ones
        // whose expiry would hurt most are the ones that get rebuilt.
        out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        out.truncate(limit);
        let keys: Vec<KeyId> = out.into_iter().map(|(k, _)| k).collect();
        for k in &keys {
            self.queue_refresh(*k);
        }
        keys
    }

    /// What the cache still owes a rebuild on, oldest first, with enough context for the
    /// application to actually rebuild it.
    ///
    /// This is the plug-and-play half of refresh: the engine knows *which* objects are
    /// about to go bad and what they cost, but only the application knows how to make one,
    /// so the engine publishes the list and the application PUTs the new value back.
    pub fn refresh_backlog(&self, limit: usize) -> Vec<(KeyId, ObjectContext, f64)> {
        self.refresh_queue
            .iter()
            .take(limit)
            .filter_map(|k| {
                let remaining = self.store.get(*k)?.ttl_remaining_frac(self.now_ms);
                Some((*k, self.store.context_of(*k)?, remaining))
            })
            .collect()
    }

    /// Rebuild an object through its origin and pay for it.
    ///
    /// The cost is charged to the backend ledger because a refresh is a real origin call:
    /// pre-emptive rebuilding is cheaper than an expiry storm, not free, and a cache that
    /// reported it as free would recommend refreshing everything constantly.
    ///
    /// `value` is what the origin returned. Passing `None` means the caller only wants the
    /// bookkeeping — used by the simulator, where the bytes are accounted rather than held.
    pub fn rebuild(&mut self, key: KeyId, value: Option<serde_json::Value>, now_ms: f64) -> bool {
        self.now_ms = now_ms.max(self.now_ms);
        let snapshot = self.store.get(key).map(|e| {
            (
                e.application.clone(),
                e.size_bytes,
                e.ttl_ms,
                e.regen,
                e.hits,
                e.ttl_remaining_frac(now_ms),
            )
        });
        let Some((app, size, ttl, regen, hits, remaining_before)) = snapshot else {
            self.refresh_queue.retain(|k| *k != key);
            return false;
        };
        let cost = self.cfg.pricing.regen_cost_usd(&regen);

        self.ledger.backend_usd += cost;
        {
            let st = self.apps.entry(app.clone()).or_default();
            st.cost_usd += cost;
            st.regen_count += 1;
            st.regen_ms_total += regen.cpu_ms + regen.gpu_ms + regen.db_ms;
        }

        if let Some(entry) = self.store.entry_mut(key) {
            if let Some(v) = value {
                entry.value = v;
            }
            entry.inserted_ms = now_ms;
        }
        self.consistency.mark_rebuilt(key);
        self.refresh_queue.retain(|k| *k != key);
        self.refreshes += 1;

        let reuse = self.reuse_peek(key, now_ms).map(|r| r[1]).unwrap_or(0.0);
        let name = format!("{app}:{key}");
        // One in eight, because a busy cache refreshes constantly and the interesting ones
        // are the expensive ones, which the severity below keeps regardless.
        if self.refreshes % 8 == 1 || cost > 0.001 {
            let message = audit::say::refreshed(&name, remaining_before, reuse.min(1.0));
            self.audit.record(
                self.now_ms,
                AuditKind::Refresh,
                if cost > 0.001 { Severity::Notice } else { Severity::Info },
                name,
                &app,
                message,
                vec![
                    Fact::new("size", audit::bytes(size)),
                    Fact::new("ttl", audit::ms(ttl)),
                    Fact::new("rebuild_cost", audit::usd(cost)),
                    Fact::new("hits_before_refresh", hits.to_string()),
                ],
                -cost,
            );
        }
        true
    }

    // ------------------------------------------------------------------ correctness

    /// Invalidate everything downstream of these tags.
    ///
    /// The whole answer to "the price changed in the database and the cache did not
    /// notice": the write emits one tag, the dependency index turns it into every affected
    /// key, and nothing needed to know in advance which rollups and rankings were built
    /// from that row.
    pub fn invalidate(
        &mut self,
        tags: &[String],
        mode: InvalidationMode,
        source: &str,
        now_ms: f64,
    ) -> InvalidationResult {
        self.now_ms = now_ms.max(self.now_ms);
        let at = self.now_ms;
        let result = self.consistency.invalidate(tags, mode, source, at);

        let mut freed = 0u64;
        for k in &result.keys_hard {
            let size = self.store.get(*k).map(|e| e.size_bytes).unwrap_or(0);
            // Through drop_key, so the removal is counted as an invalidation rather than
            // silently landing in the eviction total. Merging the two would hide the only
            // one of them that means the cache was wrong.
            self.drop_key(*k, Removal::Invalidated, 0.0, 0.0);
            freed += size;
        }
        // A soft invalidation is a promise to rebuild, so the rebuild is queued rather
        // than left to whoever happens to read next.
        for k in &result.keys_soft {
            self.queue_refresh(*k);
        }

        let hard = mode == InvalidationMode::Hard;
        let affected = if hard { result.keys_hard.len() } else { result.keys_soft.len() };
        let tag_label = tags.join(", ");
        self.audit.record(
            self.now_ms,
            AuditKind::Invalidate,
            // Never sampled away. An invalidation is a correctness event, and the one log
            // line a person actually goes looking for after a bad read.
            Severity::Notice,
            tag_label.clone(),
            source,
            audit::say::invalidated(&tag_label, affected, hard, source),
            vec![
                Fact::new("mode", if hard { "hard" } else { "soft" }),
                Fact::new("keys", affected.to_string()),
                Fact::new("bytes_freed", audit::bytes(freed)),
            ],
            0.0,
        );
        self.push_event(
            "Invalidate",
            &format!("{tag_label}: {affected} object(s), {}", if hard { "hard" } else { "soft" }),
        );
        result
    }

    /// Retire a whole generation without deleting anything.
    pub fn bump_version(&mut self, namespace: &str, now_ms: f64) -> u64 {
        self.now_ms = now_ms.max(self.now_ms);
        let at = self.now_ms;
        let version = self.consistency.bump_version(namespace, at);
        self.audit.record(
            self.now_ms,
            AuditKind::VersionBump,
            Severity::Notice,
            namespace,
            "engine",
            audit::say::version_bumped(namespace, version),
            vec![Fact::new("version", version.to_string())],
            0.0,
        );
        self.push_event("VersionBump", &format!("{namespace} -> v{version}"));
        version
    }

    pub fn consistency_stats(&self) -> ConsistencyStats {
        self.consistency.stats()
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
            if s.admitted {
                self.calib_kept.observe(s.predicted[1], s.reused[1]);
            } else {
                self.calib_refused.observe(s.predicted[1], s.reused[1]);
            }
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
        // The density is a ratio of four things and the sentence only quotes two of them.
        // When a large object with a high reuse probability is refused, the question is
        // always which term sank it, and answering that from a screenshot beats guessing:
        // value is what the object is worth over the horizons, rent is what holding it
        // costs for a minute, and density is the first divided by the second.
        let profile = *self.profiles.get(&ctx.application);
        let w = profile.horizon_weights;
        let expected_reuses = reuse[0] * w[0] * 6.0 + reuse[1] * w[1] * 2.0 + reuse[2] * w[2];
        let rent = self
            .cfg
            .pricing
            .holding_cost_usd(ctx.size_bytes as f64, 60_000.0)
            .max(1e-12);
        let facts = vec![
            Fact::new("size", audit::bytes(ctx.size_bytes)),
            Fact::new("reuse_60s", audit::percent(reuse[1])),
            Fact::new("expected_reuses", format!("{expected_reuses:.2}")),
            Fact::new("rebuild_cost", audit::usd(cost_usd)),
            Fact::new("rent_per_min", audit::usd(rent)),
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
        applied: bool,
    ) {
        let (kind, message) = if !applied {
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
            let from = self.regime.as_str();
            self.push_event("PolicyShift", &format!("{} -> {}", from, regime.as_str()));
            self.audit_regime(from, regime.as_str(), conf);
        }
        self.regime = regime;
        self.regime_confidence = conf;
        self.bandit.decay(0.995);
        self.audit_policy_shift();
        self.audit_pressure();
        self.window_requests = 0;
        self.novel_in_window = 0;
    }

    /// Say so when the strategy being rewarded has actually moved.
    ///
    /// Only a change of five points or more is reported. The mixture drifts by a fraction
    /// of a percent every tick, and a log that says so every time is a log nobody reads.
    fn audit_policy_shift(&mut self) {
        let now = self.bandit.mixture();
        let mut worst: Option<(usize, f64)> = None;
        for i in 0..6 {
            let delta = now[i] - self.last_mixture[i];
            if worst.map(|w| delta.abs() > w.1.abs()).unwrap_or(true) {
                worst = Some((i, delta));
            }
        }
        let Some((i, delta)) = worst else { return };
        if delta.abs() < 0.05 {
            return;
        }
        let toward = Policy::ALL[i].as_str();
        let because = format!(
            "The workload looks like {} and that expert has been predicting reuse better than \
             the others for the last few hundred settled decisions.",
            self.regime.as_str()
        );
        let message = audit::say::policy_shifted(toward, self.last_mixture[i], now[i], &because);
        self.audit.record(
            self.now_ms,
            AuditKind::PolicyShift,
            Severity::Notice,
            toward,
            "engine",
            message,
            vec![
                Fact::new("from", audit::percent(self.last_mixture[i])),
                Fact::new("to", audit::percent(now[i])),
                Fact::new("regime", self.regime.as_str()),
            ],
            0.0,
        );
        self.last_mixture = now;
    }

    /// A cache under pressure behaves differently, and a reader looking at a sudden drop in
    /// hit rate deserves to be told that rather than left to infer it.
    fn audit_pressure(&mut self) {
        let pressure = self.store.pressure();
        if pressure < 0.95 || self.now_ms - self.last_pressure_note_ms < 30_000.0 {
            return;
        }
        self.last_pressure_note_ms = self.now_ms;
        let elapsed_s = (self.now_ms / 1000.0).max(1.0);
        let per_s = self.store.evictions as f64 / elapsed_s;
        let (used, capacity) = (self.store.used_bytes(), self.store.capacity_bytes());
        let message = audit::say::pressure(used, capacity, per_s);
        self.audit.record(
            self.now_ms,
            AuditKind::Pressure,
            Severity::Warning,
            "capacity",
            "engine",
            message,
            vec![
                Fact::new("used", audit::bytes(used)),
                Fact::new("capacity", audit::bytes(capacity)),
                Fact::new("evictions_per_s", format!("{per_s:.0}")),
            ],
            0.0,
        );
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    fn engine(capacity_bytes: u64) -> Engine {
        let mut cfg = Config::default();
        cfg.cache.capacity_bytes = capacity_bytes;
        Engine::new(cfg, 7)
    }

    fn ctx(app: &str, size: u64, tags: &[&str], ttl_ms: f64) -> ObjectContext {
        let mut c = ObjectContext::new(app, "object", size).with_tags(tags);
        if ttl_ms > 0.0 {
            c = c.with_ttl(ttl_ms);
        }
        c
    }

    fn db_cost(ms: f64) -> CostVector {
        CostVector { db_ms: ms, latency_ms: ms, ..Default::default() }
    }

    fn admit(e: &mut Engine, key: KeyId, tags: &[&str], ttl_ms: f64, at: f64) {
        let c = ctx("analytics", 1_000, tags, ttl_ms);
        let (d, _) = e.put(key, Value::Null, &c, db_cost(40.0), at);
        assert_eq!(d.action, Action::Admit, "test setup failed to admit key {key}");
    }

    #[test]
    fn one_row_change_invalidates_everything_derived_from_it_and_nothing_else() {
        let mut e = engine(64 * 1024 * 1024);
        admit(&mut e, 1, &["row:product:7"], 0.0, 0.0);
        admit(&mut e, 2, &["row:product:7", "table:orders"], 0.0, 1.0);
        admit(&mut e, 3, &["row:product:9"], 0.0, 2.0);

        let r = e.invalidate(&["row:product:7".into()], InvalidationMode::Hard, "postgres", 100.0);
        assert_eq!(r.keys_hard.len(), 2);
        assert!(e.get(1, "analytics", 101.0).is_none(), "a stale object survived the write");
        assert!(e.get(2, "analytics", 101.0).is_none());
        assert!(
            e.get(3, "analytics", 101.0).is_some(),
            "an object built from a different row was thrown away"
        );
    }

    #[test]
    fn eviction_invalidation_and_expiry_are_counted_as_three_different_things() {
        let mut e = engine(64 * 1024 * 1024);
        admit(&mut e, 1, &["t"], 1_000.0, 0.0);
        admit(&mut e, 2, &["t"], 0.0, 0.0);

        // Expiry.
        assert!(e.get(1, "analytics", 2_000.0).is_none());
        // Invalidation.
        e.invalidate(&["t".into()], InvalidationMode::Hard, "postgres", 2_100.0);
        // Eviction: squeeze the pool until nothing fits.
        admit(&mut e, 3, &[], 0.0, 2_200.0);
        e.set_capacity(1);

        let s = e.consistency_stats();
        assert_eq!(s.expired, 1, "expiry was not counted as expiry");
        assert_eq!(s.keys_invalidated, 1, "invalidation was not counted as invalidation");
        assert!(s.evicted >= 1, "eviction was not counted as eviction");
    }

    #[test]
    fn the_last_fifth_of_a_lifetime_is_served_stale_and_queues_a_rebuild() {
        let mut e = engine(64 * 1024 * 1024);
        admit(&mut e, 1, &[], 1_000.0, 0.0);

        // Comfortably fresh: nothing is queued.
        assert!(e.get(1, "analytics", 400.0).is_some());
        assert_eq!(e.stale_serves, 0);

        // Past the soft threshold: the reader still gets an answer, and the rebuild is owed.
        assert!(e.get(1, "analytics", 900.0).is_some(), "a stale-but-valid object was withheld");
        assert_eq!(e.stale_serves, 1);
        assert!(e.refresh_queue.contains(&1), "nothing was queued for rebuild");
    }

    #[test]
    fn a_refresh_rebuilds_rather_than_resetting_the_clock() {
        let mut e = engine(64 * 1024 * 1024);
        admit(&mut e, 1, &[], 1_000.0, 0.0);
        let spent_before = e.ledger.backend_usd;

        assert!(e.rebuild(1, Some(Value::String("fresh".into())), 900.0));

        assert!(
            e.ledger.backend_usd > spent_before,
            "a refresh was recorded as free; it is an origin call and has to be paid for"
        );
        assert_eq!(e.refreshes, 1);
        assert!(e.refresh_queue.is_empty());
        let entry = e.get(1, "analytics", 1_200.0).expect("the rebuild should have extended the lifetime");
        assert_eq!(entry.value, Value::String("fresh".into()), "the clock moved but the bytes did not");
    }

    #[test]
    fn a_soft_invalidation_marks_rather_than_removes() {
        let mut e = engine(64 * 1024 * 1024);
        admit(&mut e, 1, &["table:orders"], 0.0, 0.0);

        let r = e.invalidate(&["table:orders".into()], InvalidationMode::Soft, "sdk", 10.0);
        assert_eq!(r.keys_soft.len(), 1);
        assert!(e.store.contains(1), "a soft invalidation deleted the object");
        assert!(e.refresh_queue.contains(&1));

        assert!(e.get(1, "analytics", 11.0).is_some(), "the one permitted stale serve was refused");
        assert_eq!(e.stale_serves, 1);

        // The rebuild settles the mark, and the object is fresh again.
        e.rebuild(1, None, 12.0);
        assert!(!e.consistency.is_stale(1));
    }

    #[test]
    fn a_version_bump_retires_a_generation_without_deleting_anything() {
        let mut e = engine(64 * 1024 * 1024);
        admit(&mut e, 1, &[], 0.0, 0.0);
        admit(&mut e, 2, &[], 0.0, 1.0);
        let resident = e.store.len();

        assert_eq!(e.consistency.version("recommendation"), 1);
        assert_eq!(e.bump_version("recommendation", 5.0), 2);

        assert_eq!(
            e.store.len(),
            resident,
            "the bump flushed the cache, which is the stampede it exists to avoid"
        );
        assert_eq!(e.consistency_stats().keys_invalidated, 0);
    }

    #[test]
    fn removing_an_object_forgets_what_it_depended_on() {
        let mut e = engine(64 * 1024 * 1024);
        admit(&mut e, 1, &["row:product:7"], 0.0, 0.0);
        assert_eq!(e.consistency_stats().tracked_tags, 1);

        // Squeeze everything out.
        e.set_capacity(1);

        assert!(
            e.consistency.keys_for_tag("row:product:7").is_empty(),
            "a tag survived the object it pointed at; the next invalidation would report \
             work it did not do"
        );
        assert_eq!(e.consistency_stats().tracked_tags, 0);
    }

    #[test]
    fn correctness_events_are_never_sampled_out_of_the_audit_log() {
        let mut e = engine(64 * 1024 * 1024);
        admit(&mut e, 1, &["t"], 0.0, 0.0);
        e.invalidate(&["t".into()], InvalidationMode::Hard, "postgres", 1.0);
        e.bump_version("recommendation", 2.0);

        let entries = e.audit.recent(100);
        assert!(
            entries.iter().any(|x| x.kind == "invalidate"),
            "an invalidation went unrecorded"
        );
        assert!(entries.iter().any(|x| x.kind == "version_bump"));
        // And the sentence has to actually say something.
        let inv = entries.iter().find(|x| x.kind == "invalidate").unwrap();
        assert!(inv.message.contains("postgres"), "the log does not say where the change came from");
    }
}
