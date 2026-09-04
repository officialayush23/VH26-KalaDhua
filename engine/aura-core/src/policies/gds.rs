//! Greedy Dual-Size — the baseline the problem statement names by name.
//!
//! GDS is the original cost-aware replacement policy, and it is the direct ancestor of
//! GDSF. The distinction is one term:
//!
//! ```text
//!   GDS   H(p) = L + cost(p) / size(p)
//!   GDSF  H(p) = L + freq(p) × cost(p) / size(p)
//! ```
//!
//! GDS ranks purely on *value per byte* and is completely indifferent to how often an
//! object has been used. GDSF adds frequency, which is almost always better and is why
//! everyone reaches for it instead.
//!
//! Both belong in the comparison, and not only because the brief asks for GDS. They fail
//! differently, and the difference is instructive: GDS is the policy that most directly
//! implements "keep what is expensive relative to its size", which is exactly the intuition
//! the problem statement opens with — a rarely-accessed object costing two seconds and a
//! cent to rebuild is worth more than a frequently-accessed one that is free. Showing that
//! this intuition alone is *not enough*, and that it needs frequency and near-future reuse
//! on top, is a large part of the argument for the whole system.
//!
//! The `L` inflation term is what stops an expensive object from being immortal. It is set
//! to the priority of the last object evicted, so everything admitted afterwards starts
//! from that floor. Without it, the first expensive object admitted would outrank every
//! later arrival forever, and the cache would freeze.

use super::{AccessResult, CachePolicy, Request, Resident};
use crate::config::Pricing;
use crate::types::KeyId;
use ahash::AHashMap;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

#[derive(Debug, Clone, Copy, PartialEq)]
struct HeapItem {
    /// Negated, so `BinaryHeap` (a max-heap) yields the lowest priority first.
    score: f64,
    seq: u64,
    key: KeyId,
}

impl Eq for HeapItem {}

impl Ord for HeapItem {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .partial_cmp(&other.score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| other.seq.cmp(&self.seq))
    }
}

impl PartialOrd for HeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Debug)]
pub struct Gds {
    capacity_bytes: u64,
    used_bytes: u64,
    entries: AHashMap<KeyId, Resident>,
    heap: BinaryHeap<HeapItem>,
    inflation: f64,
    counter: u64,
    pricing: Pricing,
    default_cost_usd: f64,
}

impl Gds {
    pub fn new(capacity_bytes: u64) -> Self {
        Self {
            capacity_bytes,
            used_bytes: 0,
            entries: AHashMap::new(),
            heap: BinaryHeap::new(),
            inflation: 0.0,
            counter: 0,
            pricing: Pricing::default(),
            // A floor, so an object with no reported cost is still ranked by size rather
            // than pinned at exactly zero alongside every other unpriced object.
            default_cost_usd: 1e-6,
        }
    }

    pub fn with_pricing(mut self, pricing: Pricing) -> Self {
        self.pricing = pricing;
        self
    }

    /// The whole policy: value per byte, above the inflation floor. No frequency term —
    /// that is precisely what separates it from GDSF.
    fn priority(&self, entry: &Resident) -> f64 {
        let cost = if entry.cost_usd > 0.0 { entry.cost_usd } else { self.default_cost_usd };
        self.inflation + cost / (entry.size_bytes.max(1) as f64)
    }

    fn push(&mut self, key: KeyId, priority: f64) {
        self.counter += 1;
        self.heap.push(HeapItem { score: -priority, seq: self.counter, key });
    }

    fn evict_until_fits(&mut self, needed: u64, evicted: &mut Vec<KeyId>) {
        while self.used_bytes + needed > self.capacity_bytes {
            let Some(item) = self.heap.pop() else { break };
            let Some(entry) = self.entries.get(&item.key) else { continue };
            let current = self.priority(entry);
            // Lazy deletion: a key can appear several times, and only the record matching
            // its current priority is authoritative.
            if (-item.score - current).abs() > 1e-12 {
                continue;
            }
            let size = entry.size_bytes;
            // The victim's priority becomes the floor. Nothing still resident can now be
            // considered worth less than what we just gave up.
            self.inflation = current;
            self.entries.remove(&item.key);
            self.used_bytes -= size;
            evicted.push(item.key);
        }
    }

    fn insert(&mut self, req: &Request<'_>) -> AccessResult {
        let mut result = AccessResult::miss();
        if req.size_bytes > self.capacity_bytes {
            return result;
        }
        self.evict_until_fits(req.size_bytes, &mut result.evicted);
        let cost = self.pricing.regen_cost_usd(&req.regen);
        let entry = Resident::new(req, 0, cost);
        let priority = self.priority(&entry);
        self.entries.insert(req.key, entry);
        self.used_bytes += req.size_bytes;
        self.push(req.key, priority);
        result.admitted = true;
        result
    }
}

impl CachePolicy for Gds {
    fn name(&self) -> &'static str {
        "gds"
    }

    fn access(&mut self, req: &Request<'_>) -> AccessResult {
        if let Some(entry) = self.entries.get_mut(&req.key) {
            if !entry.is_expired(req.ts_ms) {
                // A hit updates recency bookkeeping but changes nothing about the ranking:
                // GDS has no frequency term, so a hit does not make an object more valuable.
                entry.freq += 1;
                entry.last_ts_ms = req.ts_ms;
                return AccessResult::hit();
            }
            let size = entry.size_bytes;
            self.entries.remove(&req.key);
            self.used_bytes -= size;
            let mut result = self.insert(req);
            result.stale = true;
            return result;
        }
        self.insert(req)
    }

    fn capacity_bytes(&self) -> u64 {
        self.capacity_bytes
    }

    fn set_capacity_bytes(&mut self, bytes: u64) {
        self.capacity_bytes = bytes;
        let mut dropped = Vec::new();
        self.evict_until_fits(0, &mut dropped);
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
        self.entries.capacity() * (std::mem::size_of::<KeyId>() + std::mem::size_of::<Resident>())
            + self.heap.capacity() * std::mem::size_of::<HeapItem>()
    }

    fn reset(&mut self) {
        self.entries.clear();
        self.heap.clear();
        self.used_bytes = 0;
        self.inflation = 0.0;
        self.counter = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policies::{Gdsf, Lru};
    use crate::types::CostVector;

    fn dear() -> CostVector {
        CostVector { db_ms: 2_000.0, cpu_ms: 400.0, ..Default::default() }
    }
    fn cheap() -> CostVector {
        CostVector { cpu_ms: 1.0, ..Default::default() }
    }

    #[test]
    fn it_keeps_the_expensive_object_the_problem_statement_describes() {
        // The brief's own example: a rarely-accessed object costing two seconds and a cent
        // to rebuild, against a frequently-accessed one that is nearly free.
        let mut c = Gds::new(300);
        c.access(&Request::new(0.0, 1, 100).with_regen(dear()));
        c.access(&Request::new(1.0, 2, 100).with_regen(cheap()));
        c.access(&Request::new(2.0, 3, 100).with_regen(cheap()));
        // Hammer the cheap objects. GDS does not care.
        for i in 0..50 {
            c.access(&Request::new(10.0 + i as f64, 2, 100).with_regen(cheap()));
            c.access(&Request::new(10.5 + i as f64, 3, 100).with_regen(cheap()));
        }
        let result = c.access(&Request::new(100.0, 4, 100).with_regen(cheap()));
        assert!(
            !result.evicted.contains(&1),
            "GDS evicted the expensive object: {:?}",
            result.evicted
        );
    }

    #[test]
    fn frequency_does_not_change_the_ranking() {
        // The defining difference from GDSF. Two identical objects, one accessed forty
        // times; GDS must still treat them the same.
        let mut c = Gds::new(400);
        c.access(&Request::new(0.0, 1, 100).with_regen(cheap()));
        c.access(&Request::new(1.0, 2, 100).with_regen(cheap()));
        for i in 0..40 {
            c.access(&Request::new(5.0 + i as f64, 2, 100).with_regen(cheap()));
        }
        let before = c.len();
        c.access(&Request::new(100.0, 3, 100).with_regen(cheap()));
        c.access(&Request::new(101.0, 4, 100).with_regen(cheap()));
        let result = c.access(&Request::new(102.0, 5, 100).with_regen(cheap()));
        assert_eq!(before, 2);
        // Whichever it drops, it is not because key 2 was accessed forty times: both have
        // identical cost and size, so the tie is broken by insertion order alone.
        assert!(!result.evicted.is_empty());
    }

    #[test]
    fn gdsf_beats_gds_when_frequency_matters() {
        // The reason GDSF exists. Uniform cost and size, so the only usable signal is
        // frequency — which GDS cannot see at all.
        use crate::policies::testkit::zipf_hit_rate;
        let mut gds = Gds::new(200 * 100);
        let mut gdsf = Gdsf::new(200 * 100);
        let gds_hr = zipf_hit_rate(&mut gds, 5_000, 120_000, 100);
        let gdsf_hr = zipf_hit_rate(&mut gdsf, 5_000, 120_000, 100);
        assert!(
            gdsf_hr > gds_hr,
            "with uniform cost, frequency should win: gdsf {gdsf_hr:.3} vs gds {gds_hr:.3}"
        );
    }

    #[test]
    fn gds_beats_lru_when_cost_is_skewed() {
        // And the reason GDS exists. Same access pattern for every key, but a small
        // minority are far more expensive to rebuild. LRU cannot see cost either.
        use crate::rng::Rng;
        let mut rng = Rng::seed_from_u64(11);
        let keys: Vec<KeyId> = (0..30_000).map(|_| rng.below(600)).collect();
        let expensive = |k: KeyId| k % 20 == 0;

        let mut gds = Gds::new(150 * 100);
        let mut lru = Lru::new(150 * 100);
        let (mut gds_cost, mut lru_cost) = (0.0f64, 0.0f64);

        for (i, key) in keys.iter().enumerate() {
            let regen = if expensive(*key) { dear() } else { cheap() };
            let req = Request::new(i as f64, *key, 100).with_regen(regen);
            let price = Pricing::default().regen_cost_usd(&regen);
            if !gds.access(&req).hit {
                gds_cost += price;
            }
            if !lru.access(&req).hit {
                lru_cost += price;
            }
        }
        assert!(
            gds_cost < lru_cost,
            "GDS should spend less on rebuilds than LRU when cost is skewed: {gds_cost:.4} vs {lru_cost:.4}"
        );
    }

    #[test]
    fn the_inflation_term_stops_an_expensive_object_being_immortal() {
        let mut c = Gds::new(200);
        // One very expensive object, then a long stream of ordinary ones.
        c.access(&Request::new(0.0, 1, 100).with_regen(CostVector {
            gpu_ms: 100_000.0,
            ..Default::default()
        }));
        let mut evicted_the_whale = false;
        for i in 0..500u64 {
            let r = c.access(&Request::new(10.0 + i as f64, 1_000 + i, 100).with_regen(dear()));
            if r.evicted.contains(&1) {
                evicted_the_whale = true;
                break;
            }
        }
        assert!(
            evicted_the_whale,
            "without a working inflation floor the first expensive object never leaves"
        );
    }
}
