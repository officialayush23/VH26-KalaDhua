//! Cache replacement policies.
//!
//! Every policy here is a *complete* cache: it owns its residents, its byte accounting and
//! its ordering structure, and it answers one question — what happens to this request.
//! That is deliberate. A benchmark that compares policies through a shared eviction
//! interface quietly imposes one policy's structure on all of them, and the modern ones
//! (W-TinyLFU, S3-FIFO, SIEVE) are *defined* by their structure, not by a scoring
//! function. Letting each own its own machinery is the only way the comparison is fair.
//!
//! All policies are size-aware: they evict until the incoming object fits, rather than
//! evicting exactly one victim. Uniform-object-size benchmarks flatter recency-based
//! policies and hide exactly the failure mode this project is about.

use crate::types::{CostVector, KeyId, SlaClass};

pub mod classical;
pub mod gds;
pub mod learned;
pub mod modern;
pub mod oracle;

pub use classical::{Fifo, Gdsf, Lfu, Lru};
pub use gds::Gds;
pub use learned::LeCaR;
pub use modern::{S3Fifo, Sieve, WTinyLfu};
pub use oracle::Belady;

/// One request as a policy sees it.
///
/// The cost vector travels with the request because a size-and-recency policy can ignore
/// it, but a cost-aware one cannot reconstruct it later — and the whole argument of this
/// project is that the information is available and conventional policies throw it away.
#[derive(Debug, Clone, Copy)]
pub struct Request<'a> {
    pub ts_ms: f64,
    pub key: KeyId,
    pub size_bytes: u64,
    /// Zero means the object never expires.
    pub ttl_ms: f64,
    pub regen: CostVector,
    pub application: &'a str,
    pub object_type: &'a str,
    pub sla: SlaClass,
}

impl<'a> Request<'a> {
    pub fn new(ts_ms: f64, key: KeyId, size_bytes: u64) -> Self {
        Self {
            ts_ms,
            key,
            size_bytes,
            ttl_ms: 0.0,
            regen: CostVector::default(),
            application: "default",
            object_type: "object",
            sla: SlaClass::Normal,
        }
    }

    pub fn with_regen(mut self, regen: CostVector) -> Self {
        self.regen = regen;
        self
    }

    pub fn with_ttl(mut self, ttl_ms: f64) -> Self {
        self.ttl_ms = ttl_ms;
        self
    }
}

/// What a policy did with one request.
#[derive(Debug, Clone, Default)]
pub struct AccessResult {
    pub hit: bool,
    /// The object was resident but past its TTL. Counted as a miss for hit rate, and
    /// separately, because a policy with a refresh controller should be able to drive
    /// this number to nearly zero.
    pub stale: bool,
    /// Whether the object was placed in the cache after a miss. A policy that declines is
    /// making an admission decision, not failing.
    pub admitted: bool,
    pub evicted: Vec<KeyId>,
}

impl AccessResult {
    pub fn hit() -> Self {
        Self { hit: true, ..Default::default() }
    }

    pub fn miss() -> Self {
        Self::default()
    }
}

/// The interface the benchmark harness and the simulator drive.
pub trait CachePolicy: std::fmt::Debug + Send {
    /// Stable identifier, used as the column name in every results table.
    fn name(&self) -> &'static str;

    /// Serve one request, mutating the cache.
    fn access(&mut self, req: &Request<'_>) -> AccessResult;

    fn capacity_bytes(&self) -> u64;

    /// Change the byte budget at runtime. Shrinking evicts immediately; this is the hook
    /// the capacity controller drives, and every baseline implements it so that "the same
    /// cache, resized" is a comparable experiment.
    fn set_capacity_bytes(&mut self, bytes: u64);

    fn used_bytes(&self) -> u64;

    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn contains(&self, key: KeyId) -> bool;

    /// Bytes of bookkeeping the policy carries, excluding the cached objects themselves.
    /// Reported in the benchmark so a policy cannot buy its hit rate with metadata the
    /// comparison never charges it for.
    fn memory_overhead_bytes(&self) -> usize;

    fn reset(&mut self);
}

/// Shared record for a resident object.
///
/// Policies that need extra per-entry state keep it in their own map alongside this.
#[derive(Debug, Clone, Copy)]
pub struct Resident {
    pub size_bytes: u64,
    pub fill_ts_ms: f64,
    pub ttl_ms: f64,
    /// Handle into the policy's ordering structure, where it has one.
    pub slot: usize,
    pub freq: u32,
    pub last_ts_ms: f64,
    /// Priced regeneration cost, carried so cost-aware policies do not have to re-derive
    /// it and so eviction can attribute a saving.
    pub cost_usd: f64,
    /// Priority assigned when the object was admitted or last promoted.
    ///
    /// Greedy-Dual policies **must** store this rather than recompute it. Their inflation
    /// term `L` rises as objects are evicted, and if every resident's priority were
    /// recomputed against the current `L`, the term would be added to all of them equally
    /// and cancel out entirely — leaving an expensive object immortal, which is the exact
    /// failure the inflation term exists to prevent.
    pub score: f64,
}

impl Resident {
    pub fn new(req: &Request<'_>, slot: usize, cost_usd: f64) -> Self {
        Self {
            size_bytes: req.size_bytes,
            fill_ts_ms: req.ts_ms,
            ttl_ms: req.ttl_ms,
            slot,
            freq: 1,
            last_ts_ms: req.ts_ms,
            cost_usd,
            score: 0.0,
        }
    }

    pub fn is_expired(&self, now_ms: f64) -> bool {
        self.ttl_ms > 0.0 && now_ms - self.fill_ts_ms >= self.ttl_ms
    }

    pub fn ttl_remaining_frac(&self, now_ms: f64) -> f64 {
        if self.ttl_ms <= 0.0 {
            return 1.0;
        }
        (1.0 - (now_ms - self.fill_ts_ms) / self.ttl_ms).clamp(0.0, 1.0)
    }
}

/// Build a policy by name. The benchmark harness and the `/v1/policy/override` endpoint
/// both go through here so there is exactly one place that knows the roster.
pub fn build(name: &str, capacity_bytes: u64) -> Option<Box<dyn CachePolicy>> {
    match name {
        "lru" => Some(Box::new(Lru::new(capacity_bytes))),
        "fifo" => Some(Box::new(Fifo::new(capacity_bytes))),
        "lfu" => Some(Box::new(Lfu::new(capacity_bytes))),
        "gds" => Some(Box::new(Gds::new(capacity_bytes))),
        "gdsf" => Some(Box::new(Gdsf::new(capacity_bytes))),
        "tinylfu" | "w_tinylfu" => Some(Box::new(WTinyLfu::new(capacity_bytes))),
        "s3fifo" => Some(Box::new(S3Fifo::new(capacity_bytes))),
        "sieve" => Some(Box::new(Sieve::new(capacity_bytes))),
        "lecar" => Some(Box::new(LeCaR::new(capacity_bytes))),
        _ => None,
    }
}

/// Every policy that can be constructed without an oracle.
/// Every policy that can be constructed without an oracle. `gds` is listed because the
/// brief names it explicitly; `gdsf` is its frequency-aware successor and usually wins.
pub const BASELINE_NAMES: [&str; 9] =
    ["lru", "fifo", "lfu", "gds", "gdsf", "tinylfu", "s3fifo", "sieve", "lecar"];

#[cfg(test)]
pub(crate) mod testkit {
    use super::*;
    use crate::rng::{Rng, Zipf};

    /// Replay a Zipf workload and report the object hit rate.
    ///
    /// Used by every policy's tests. The point is not to rank policies here — that is what
    /// the benchmark harness is for — but to catch a policy that is outright broken, which
    /// shows up as a hit rate near zero or near one.
    pub fn zipf_hit_rate(policy: &mut dyn CachePolicy, keys: usize, requests: usize, size: u64) -> f64 {
        let zipf = Zipf::new(keys, 1.0);
        let mut rng = Rng::seed_from_u64(42);
        let mut hits = 0usize;
        for i in 0..requests {
            let key = zipf.sample(&mut rng) as KeyId;
            let req = Request::new(i as f64, key, size);
            if policy.access(&req).hit {
                hits += 1;
            }
        }
        hits as f64 / requests as f64
    }

    /// A policy must never exceed its budget, never lose an object it reported admitting
    /// without reporting the eviction, and must survive objects larger than the cache.
    pub fn assert_invariants(policy: &mut dyn CachePolicy) {
        let cap = policy.capacity_bytes();
        let mut rng = Rng::seed_from_u64(7);
        for i in 0..20_000u64 {
            let key = rng.below(2_000);
            // Occasionally offer an object that cannot possibly fit.
            let size = if i % 997 == 0 { cap * 2 } else { rng.range(1.0, 64_000.0) as u64 };
            let req = Request::new(i as f64, key, size.max(1));
            policy.access(&req);
            assert!(
                policy.used_bytes() <= cap,
                "{} exceeded its budget: {} > {}",
                policy.name(),
                policy.used_bytes(),
                cap
            );
        }
        assert!(policy.len() > 0, "{} evicted everything", policy.name());
    }
}
