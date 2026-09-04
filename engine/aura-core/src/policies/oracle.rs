//! Belady's optimal replacement, and its cost-aware sibling.
//!
//! Neither is implementable online — both need to know the future — and that is exactly why
//! they belong in the benchmark. Reporting "we improved the hit rate by 8%" means nothing
//! without knowing whether the remaining headroom was 9% or 40%. Belady closes that
//! question: it is the ceiling, and every real policy is measured as a fraction of the gap
//! between LRU and it.
//!
//! The cost-aware variant matters more here than the classical one. Belady maximises
//! *hits*; this project optimises *money*. On a workload where 5% of the objects carry 80%
//! of the regeneration cost, the hit-rate-optimal schedule and the cost-optimal schedule
//! are different, and showing that gap is a large part of the argument.

use super::{AccessResult, CachePolicy, Request, Resident};
use crate::config::Pricing;
use crate::types::KeyId;
use ahash::AHashMap;

/// Objective the oracle optimises for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Objective {
    /// Classical Belady: evict whatever is needed furthest in the future.
    HitRate,
    /// Evict whatever has the worst `cost / (size * urgency)`. An approximation — the truly
    /// optimal cost-aware schedule is NP-hard — but a far tighter ceiling than hit-rate
    /// Belady on cost-skewed workloads.
    Cost,
}

/// Offline optimal replacement.
///
/// Construction requires the whole request sequence, because the policy is defined in terms
/// of it. [`Belady::next_access_times`] does the one preprocessing pass.
#[derive(Debug)]
pub struct Belady {
    capacity_bytes: u64,
    used_bytes: u64,
    entries: AHashMap<KeyId, (Resident, f64)>,
    /// For request `i`, when that same key is next requested. `f64::INFINITY` if never.
    next_access: Vec<f64>,
    cursor: usize,
    objective: Objective,
    pricing: Pricing,
}

impl Belady {
    pub fn new(capacity_bytes: u64, next_access: Vec<f64>, objective: Objective) -> Self {
        Self {
            capacity_bytes,
            used_bytes: 0,
            entries: AHashMap::new(),
            next_access,
            cursor: 0,
            objective,
            pricing: Pricing::default(),
        }
    }

    /// One backwards pass over a request sequence, producing the next-access time of each
    /// request's key. This is the only preprocessing the oracle needs.
    pub fn next_access_times(keys: &[KeyId], timestamps: &[f64]) -> Vec<f64> {
        assert_eq!(keys.len(), timestamps.len(), "keys and timestamps must align");
        let mut next = vec![f64::INFINITY; keys.len()];
        let mut seen: AHashMap<KeyId, f64> = AHashMap::new();
        for i in (0..keys.len()).rev() {
            if let Some(t) = seen.get(&keys[i]) {
                next[i] = *t;
            }
            seen.insert(keys[i], timestamps[i]);
        }
        next
    }

    /// Score a resident object: lower is evicted first.
    fn retention_score(&self, entry: &Resident, next_ms: f64, now_ms: f64) -> f64 {
        match self.objective {
            Objective::HitRate => next_ms,
            Objective::Cost => {
                if next_ms.is_infinite() {
                    return f64::NEG_INFINITY;
                }
                let wait = (next_ms - now_ms).max(1.0);
                // Value per byte, discounted by how long we have to hold it to collect.
                -(wait * entry.size_bytes.max(1) as f64) / entry.cost_usd.max(1e-12)
            }
        }
    }

    fn evict_until_fits(&mut self, needed: u64, now_ms: f64, evicted: &mut Vec<KeyId>) {
        while self.used_bytes + needed > self.capacity_bytes && !self.entries.is_empty() {
            let mut worst_key = None;
            let mut worst_score = f64::NEG_INFINITY;
            for (key, (entry, next_ms)) in self.entries.iter() {
                // For hit rate, "worst" is the largest next-access time; for cost, the
                // score is already oriented so that larger means more disposable.
                let score = match self.objective {
                    Objective::HitRate => self.retention_score(entry, *next_ms, now_ms),
                    Objective::Cost => -self.retention_score(entry, *next_ms, now_ms),
                };
                if score > worst_score {
                    worst_score = score;
                    worst_key = Some(*key);
                }
            }
            let Some(key) = worst_key else { break };
            if let Some((entry, _)) = self.entries.remove(&key) {
                self.used_bytes -= entry.size_bytes;
                evicted.push(key);
            }
        }
    }

    fn current_next_access(&self) -> f64 {
        self.next_access.get(self.cursor).copied().unwrap_or(f64::INFINITY)
    }
}

impl CachePolicy for Belady {
    fn name(&self) -> &'static str {
        match self.objective {
            Objective::HitRate => "belady",
            Objective::Cost => "belady_cost",
        }
    }

    fn access(&mut self, req: &Request<'_>) -> AccessResult {
        let next_ms = self.current_next_access();
        self.cursor += 1;

        if let Some((entry, slot_next)) = self.entries.get_mut(&req.key) {
            if !entry.is_expired(req.ts_ms) {
                entry.freq += 1;
                entry.last_ts_ms = req.ts_ms;
                *slot_next = next_ms;
                return AccessResult::hit();
            }
            let size = entry.size_bytes;
            self.entries.remove(&req.key);
            self.used_bytes -= size;
        }

        let mut result = AccessResult::miss();
        if req.size_bytes > self.capacity_bytes {
            return result;
        }
        // An object that is never requested again is worth nothing: the optimal schedule
        // does not cache it at all. Declining here is not a heuristic, it is the definition.
        if next_ms.is_infinite() {
            return result;
        }
        self.evict_until_fits(req.size_bytes, req.ts_ms, &mut result.evicted);
        let cost = self.pricing.regen_cost_usd(&req.regen);
        self.entries.insert(req.key, (Resident::new(req, 0, cost), next_ms));
        self.used_bytes += req.size_bytes;
        result.admitted = true;
        result
    }

    fn capacity_bytes(&self) -> u64 {
        self.capacity_bytes
    }

    fn set_capacity_bytes(&mut self, bytes: u64) {
        self.capacity_bytes = bytes;
        let mut dropped = Vec::new();
        self.evict_until_fits(0, f64::MAX, &mut dropped);
    }

    fn used_bytes(&self) -> u64 {
        self.used_bytes
    }

    fn len(&self) -> usize {
        self.entries.len()
    }

    fn contains(&self, key: KeyId) -> bool {
        self.entries.contains_key(&key)
    }

    fn memory_overhead_bytes(&self) -> usize {
        // Charged honestly, including the future the oracle is allowed to see — which is
        // the point: it is not a policy anyone could deploy.
        self.entries.capacity() * (std::mem::size_of::<KeyId>() + std::mem::size_of::<(Resident, f64)>())
            + self.next_access.capacity() * std::mem::size_of::<f64>()
    }

    fn reset(&mut self) {
        self.entries.clear();
        self.used_bytes = 0;
        self.cursor = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policies::Lru;
    use crate::rng::{Rng, Zipf};
    use crate::types::CostVector;

    #[test]
    fn next_access_times_are_computed_correctly() {
        let keys = vec![1u64, 2, 1, 3, 2];
        let ts = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let next = Belady::next_access_times(&keys, &ts);
        assert_eq!(next[0], 2.0); // key 1 is next seen at t=2
        assert_eq!(next[1], 4.0); // key 2 is next seen at t=4
        assert!(next[2].is_infinite()); // key 1 never again
        assert!(next[3].is_infinite());
        assert!(next[4].is_infinite());
    }

    #[test]
    fn belady_is_an_upper_bound_on_lru() {
        let zipf = Zipf::new(2_000, 0.9);
        let mut rng = Rng::seed_from_u64(4);
        let keys: Vec<KeyId> = (0..40_000).map(|_| zipf.sample(&mut rng) as KeyId).collect();
        let ts: Vec<f64> = (0..keys.len()).map(|i| i as f64).collect();
        let next = Belady::next_access_times(&keys, &ts);

        let capacity = 200 * 100;
        let mut opt = Belady::new(capacity, next, Objective::HitRate);
        let mut lru = Lru::new(capacity);

        let (mut opt_hits, mut lru_hits) = (0usize, 0usize);
        for (i, key) in keys.iter().enumerate() {
            let req = Request::new(ts[i], *key, 100);
            if opt.access(&req).hit {
                opt_hits += 1;
            }
            if lru.access(&req).hit {
                lru_hits += 1;
            }
        }
        assert!(
            opt_hits > lru_hits,
            "the oracle must not lose to LRU: opt {opt_hits} lru {lru_hits}"
        );
    }

    #[test]
    fn the_cost_objective_protects_the_expensive_tail() {
        // Same access pattern for every key, but 5% of them are 50x more expensive to
        // regenerate. A hit-rate-optimal schedule is indifferent; a cost-optimal one is not.
        let mut rng = Rng::seed_from_u64(12);
        let keys: Vec<KeyId> = (0..20_000).map(|_| rng.below(400)).collect();
        let ts: Vec<f64> = (0..keys.len()).map(|i| i as f64).collect();
        let next = Belady::next_access_times(&keys, &ts);
        let expensive = |k: KeyId| k % 20 == 0;

        let capacity = 100 * 100;
        let mut cost_opt = Belady::new(capacity, next.clone(), Objective::Cost);
        let mut hit_opt = Belady::new(capacity, next, Objective::HitRate);

        for (i, key) in keys.iter().enumerate() {
            let regen = if expensive(*key) {
                CostVector { db_ms: 5000.0, cpu_ms: 500.0, ..Default::default() }
            } else {
                CostVector { db_ms: 100.0, ..Default::default() }
            };
            let req = Request::new(ts[i], *key, 100).with_regen(regen);
            cost_opt.access(&req);
            hit_opt.access(&req);
        }

        let cost_kept = (0..400).filter(|k| expensive(*k) && cost_opt.contains(*k)).count();
        let hit_kept = (0..400).filter(|k| expensive(*k) && hit_opt.contains(*k)).count();
        assert!(
            cost_kept >= hit_kept,
            "the cost objective should retain at least as much of the expensive tail: {cost_kept} vs {hit_kept}"
        );
    }
}
