//! The classical baselines: FIFO, LRU, LFU and GDSF.
//!
//! These are the policies the problem statement names, and they are the floor the adaptive
//! engine has to clear. They are implemented properly rather than approximately — a
//! straw-man LRU that loses is not evidence of anything.

use super::{AccessResult, CachePolicy, Request, Resident};
use crate::config::Pricing;
use crate::list::IntrusiveList;
use crate::types::KeyId;
use ahash::AHashMap;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// First in, first out. No reordering on a hit.
///
/// Worth keeping as a baseline because it is what S3-FIFO and SIEVE are built out of, and
/// because on some workloads it beats LRU while costing nothing to maintain.
#[derive(Debug)]
pub struct Fifo {
    capacity_bytes: u64,
    used_bytes: u64,
    entries: AHashMap<KeyId, Resident>,
    order: IntrusiveList,
}

impl Fifo {
    pub fn new(capacity_bytes: u64) -> Self {
        Self {
            capacity_bytes,
            used_bytes: 0,
            entries: AHashMap::new(),
            order: IntrusiveList::new(),
        }
    }

    fn evict_until_fits(&mut self, needed: u64, evicted: &mut Vec<KeyId>) {
        while self.used_bytes + needed > self.capacity_bytes {
            let Some(victim) = self.order.pop_front() else { break };
            if let Some(entry) = self.entries.remove(&victim) {
                self.used_bytes -= entry.size_bytes;
                evicted.push(victim);
            }
        }
    }
}

impl CachePolicy for Fifo {
    fn name(&self) -> &'static str {
        "fifo"
    }

    fn access(&mut self, req: &Request<'_>) -> AccessResult {
        if let Some(entry) = self.entries.get_mut(&req.key) {
            if !entry.is_expired(req.ts_ms) {
                entry.freq += 1;
                entry.last_ts_ms = req.ts_ms;
                return AccessResult::hit();
            }
            let stale = *entry;
            self.order.remove(stale.slot);
            self.entries.remove(&req.key);
            self.used_bytes -= stale.size_bytes;
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
            + self.order.memory_bytes()
    }

    fn reset(&mut self) {
        self.entries.clear();
        self.order.clear();
        self.used_bytes = 0;
    }
}

impl Fifo {
    fn insert(&mut self, req: &Request<'_>) -> AccessResult {
        let mut result = AccessResult::miss();
        if req.size_bytes > self.capacity_bytes {
            return result;
        }
        self.evict_until_fits(req.size_bytes, &mut result.evicted);
        let slot = self.order.push_back(req.key);
        self.entries.insert(req.key, Resident::new(req, slot, 0.0));
        self.used_bytes += req.size_bytes;
        result.admitted = true;
        result
    }
}

/// Least recently used.
///
/// The reference point for every cache argument, and the policy a sequential scan destroys
/// completely — which is why the benchmark includes a scan.
#[derive(Debug)]
pub struct Lru {
    capacity_bytes: u64,
    used_bytes: u64,
    entries: AHashMap<KeyId, Resident>,
    order: IntrusiveList,
}

impl Lru {
    pub fn new(capacity_bytes: u64) -> Self {
        Self {
            capacity_bytes,
            used_bytes: 0,
            entries: AHashMap::new(),
            order: IntrusiveList::new(),
        }
    }

    fn evict_until_fits(&mut self, needed: u64, evicted: &mut Vec<KeyId>) {
        while self.used_bytes + needed > self.capacity_bytes {
            let Some(victim) = self.order.pop_back() else { break };
            if let Some(entry) = self.entries.remove(&victim) {
                self.used_bytes -= entry.size_bytes;
                evicted.push(victim);
            }
        }
    }

    fn insert(&mut self, req: &Request<'_>) -> AccessResult {
        let mut result = AccessResult::miss();
        if req.size_bytes > self.capacity_bytes {
            return result;
        }
        self.evict_until_fits(req.size_bytes, &mut result.evicted);
        let slot = self.order.push_front(req.key);
        self.entries.insert(req.key, Resident::new(req, slot, 0.0));
        self.used_bytes += req.size_bytes;
        result.admitted = true;
        result
    }
}

impl CachePolicy for Lru {
    fn name(&self) -> &'static str {
        "lru"
    }

    fn access(&mut self, req: &Request<'_>) -> AccessResult {
        if let Some(entry) = self.entries.get_mut(&req.key) {
            if !entry.is_expired(req.ts_ms) {
                entry.freq += 1;
                entry.last_ts_ms = req.ts_ms;
                let slot = entry.slot;
                self.order.move_to_front(slot);
                return AccessResult::hit();
            }
            let stale = *entry;
            self.order.remove(stale.slot);
            self.entries.remove(&req.key);
            self.used_bytes -= stale.size_bytes;
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
            + self.order.memory_bytes()
    }

    fn reset(&mut self) {
        self.entries.clear();
        self.order.clear();
        self.used_bytes = 0;
    }
}

/// Least frequently used, with ageing.
///
/// Plain LFU never forgets, so a key that was popular an hour ago outranks one that is
/// popular now — the "popularity inversion" failure the benchmark attacks deliberately.
/// Halving every counter periodically is the standard fix and makes this a fair opponent
/// rather than a caricature.
#[derive(Debug)]
pub struct Lfu {
    capacity_bytes: u64,
    used_bytes: u64,
    entries: AHashMap<KeyId, Resident>,
    /// Lazy min-heap on (frequency, insertion order). Stale entries are discarded on pop
    /// rather than eagerly updated, which keeps the hit path free of heap work.
    heap: BinaryHeap<HeapItem>,
    counter: u64,
    accesses_since_age: u64,
    age_every: u64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct HeapItem {
    /// Negated so `BinaryHeap`, which is a max-heap, yields the smallest score first.
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

impl Lfu {
    pub fn new(capacity_bytes: u64) -> Self {
        Self {
            capacity_bytes,
            used_bytes: 0,
            entries: AHashMap::new(),
            heap: BinaryHeap::new(),
            counter: 0,
            accesses_since_age: 0,
            // Halve counters roughly once per full turnover of a mid-sized cache.
            age_every: 100_000,
        }
    }

    fn push(&mut self, key: KeyId, freq: u32) {
        self.counter += 1;
        self.heap.push(HeapItem { score: -(freq as f64), seq: self.counter, key });
    }

    fn evict_until_fits(&mut self, needed: u64, evicted: &mut Vec<KeyId>) {
        while self.used_bytes + needed > self.capacity_bytes {
            let Some(item) = self.heap.pop() else { break };
            let Some(entry) = self.entries.get(&item.key) else { continue };
            // A key can appear in the heap several times; only the record matching the
            // current frequency is authoritative.
            if -(entry.freq as f64) != item.score {
                continue;
            }
            let size = entry.size_bytes;
            self.entries.remove(&item.key);
            self.used_bytes -= size;
            evicted.push(item.key);
        }
    }

    fn maybe_age(&mut self) {
        self.accesses_since_age += 1;
        if self.accesses_since_age < self.age_every {
            return;
        }
        self.accesses_since_age = 0;
        let keys: Vec<KeyId> = self.entries.keys().copied().collect();
        for key in keys {
            if let Some(e) = self.entries.get_mut(&key) {
                e.freq = (e.freq / 2).max(1);
                let freq = e.freq;
                self.push(key, freq);
            }
        }
    }

    fn insert(&mut self, req: &Request<'_>) -> AccessResult {
        let mut result = AccessResult::miss();
        if req.size_bytes > self.capacity_bytes {
            return result;
        }
        self.evict_until_fits(req.size_bytes, &mut result.evicted);
        self.entries.insert(req.key, Resident::new(req, 0, 0.0));
        self.used_bytes += req.size_bytes;
        self.push(req.key, 1);
        result.admitted = true;
        result
    }
}

impl CachePolicy for Lfu {
    fn name(&self) -> &'static str {
        "lfu"
    }

    fn access(&mut self, req: &Request<'_>) -> AccessResult {
        self.maybe_age();
        if let Some(entry) = self.entries.get_mut(&req.key) {
            if !entry.is_expired(req.ts_ms) {
                entry.freq += 1;
                entry.last_ts_ms = req.ts_ms;
                let freq = entry.freq;
                self.push(req.key, freq);
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
        self.counter = 0;
        self.accesses_since_age = 0;
    }
}

/// Greedy Dual-Size Frequency.
///
/// The strongest of the required baselines and the one worth beating. Each object gets
///
/// ```text
///   H(p) = L + freq(p) * cost(p) / size(p)
/// ```
///
/// where `L` is an inflation value set to the key of the last evicted object. That `L`
/// term is what stops old high-value objects from being immortal, and it is the reason
/// GDSF handles heterogeneous object sizes far better than LRU or LFU.
///
/// AURA's advantage over this is not that it knows about cost — GDSF already does — but
/// that GDSF's `freq` is *historical* while AURA's is a prediction of near-future reuse,
/// and that GDSF's `cost` is a point estimate while AURA carries the shape of the
/// distribution. The benchmark is built to test exactly that difference.
#[derive(Debug)]
pub struct Gdsf {
    capacity_bytes: u64,
    used_bytes: u64,
    entries: AHashMap<KeyId, Resident>,
    heap: BinaryHeap<HeapItem>,
    inflation: f64,
    counter: u64,
    pricing: Pricing,
    /// Cost floor, so an object with no reported regeneration cost still has a size- and
    /// frequency-aware ranking rather than a value of exactly zero.
    default_cost_usd: f64,
}

impl Gdsf {
    pub fn new(capacity_bytes: u64) -> Self {
        Self {
            capacity_bytes,
            used_bytes: 0,
            entries: AHashMap::new(),
            heap: BinaryHeap::new(),
            inflation: 0.0,
            counter: 0,
            pricing: Pricing::default(),
            default_cost_usd: 1e-6,
        }
    }

    pub fn with_pricing(mut self, pricing: Pricing) -> Self {
        self.pricing = pricing;
        self
    }

    fn priority(&self, entry: &Resident) -> f64 {
        let cost = if entry.cost_usd > 0.0 { entry.cost_usd } else { self.default_cost_usd };
        self.inflation + (entry.freq as f64) * cost / (entry.size_bytes.max(1) as f64)
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
            // Skip records that no longer describe the entry's current priority.
            if (-item.score - current).abs() > 1e-12 {
                continue;
            }
            let size = entry.size_bytes;
            // The evicted object's priority becomes the inflation floor: nothing already
            // resident can now be worth less than what we just gave up.
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

impl CachePolicy for Gdsf {
    fn name(&self) -> &'static str {
        "gdsf"
    }

    fn access(&mut self, req: &Request<'_>) -> AccessResult {
        if let Some(entry) = self.entries.get_mut(&req.key) {
            if !entry.is_expired(req.ts_ms) {
                entry.freq += 1;
                entry.last_ts_ms = req.ts_ms;
                let snapshot = *entry;
                let priority = self.priority(&snapshot);
                self.push(req.key, priority);
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
    use super::super::testkit::{assert_invariants, zipf_hit_rate};
    use super::*;
    use crate::types::CostVector;

    #[test]
    fn lru_keeps_the_most_recent() {
        let mut c = Lru::new(300);
        for k in 0..3u64 {
            c.access(&Request::new(k as f64, k, 100));
        }
        // Touch 0 so it is no longer the coldest, then force one eviction.
        c.access(&Request::new(10.0, 0, 100));
        let result = c.access(&Request::new(11.0, 99, 100));
        assert_eq!(result.evicted, vec![1], "LRU must drop the least recently used key");
        assert!(c.contains(0) && c.contains(2) && c.contains(99));
    }

    #[test]
    fn fifo_ignores_recency() {
        let mut c = Fifo::new(300);
        for k in 0..3u64 {
            c.access(&Request::new(k as f64, k, 100));
        }
        c.access(&Request::new(10.0, 0, 100)); // hit, but FIFO does not reorder
        let result = c.access(&Request::new(11.0, 99, 100));
        assert_eq!(result.evicted, vec![0], "FIFO must drop the oldest insertion");
    }

    #[test]
    fn lfu_keeps_the_most_frequent() {
        let mut c = Lfu::new(300);
        for k in 0..3u64 {
            c.access(&Request::new(k as f64, k, 100));
        }
        for i in 0..10 {
            c.access(&Request::new(10.0 + i as f64, 0, 100));
            c.access(&Request::new(20.0 + i as f64, 2, 100));
        }
        let result = c.access(&Request::new(100.0, 99, 100));
        assert_eq!(result.evicted, vec![1], "LFU must drop the least frequently used key");
    }

    #[test]
    fn gdsf_protects_expensive_objects_over_equally_popular_cheap_ones() {
        let mut c = Gdsf::new(300);
        let cheap = CostVector { cpu_ms: 1.0, ..Default::default() };
        let dear = CostVector { gpu_ms: 900.0, db_ms: 500.0, ..Default::default() };
        c.access(&Request::new(0.0, 1, 100).with_regen(dear));
        c.access(&Request::new(1.0, 2, 100).with_regen(cheap));
        c.access(&Request::new(2.0, 3, 100).with_regen(cheap));
        // Equal frequency, equal size: only the regeneration cost separates them.
        let result = c.access(&Request::new(3.0, 4, 100).with_regen(cheap));
        assert!(
            !result.evicted.contains(&1),
            "GDSF evicted the expensive object: {:?}",
            result.evicted
        );
    }

    #[test]
    fn gdsf_prefers_small_objects_when_cost_is_equal() {
        let mut c = Gdsf::new(1000);
        let cost = CostVector { db_ms: 100.0, ..Default::default() };
        c.access(&Request::new(0.0, 1, 100).with_regen(cost)); // dense
        c.access(&Request::new(1.0, 2, 800).with_regen(cost)); // sparse
        let result = c.access(&Request::new(2.0, 3, 200).with_regen(cost));
        assert!(
            result.evicted.contains(&2),
            "GDSF should give up the object with the worse value per byte: {:?}",
            result.evicted
        );
    }

    #[test]
    fn expired_entries_are_reported_as_stale_misses() {
        let mut c = Lru::new(1000);
        c.access(&Request::new(0.0, 1, 100).with_ttl(50.0));
        let fresh = c.access(&Request::new(10.0, 1, 100).with_ttl(50.0));
        assert!(fresh.hit && !fresh.stale);
        let expired = c.access(&Request::new(100.0, 1, 100).with_ttl(50.0));
        assert!(!expired.hit && expired.stale && expired.admitted);
    }

    #[test]
    fn objects_larger_than_the_cache_are_declined_without_flushing_it() {
        let mut c = Lru::new(1000);
        c.access(&Request::new(0.0, 1, 500));
        let result = c.access(&Request::new(1.0, 2, 5000));
        assert!(!result.admitted);
        assert!(result.evicted.is_empty(), "an unadmittable object must not evict anything");
        assert!(c.contains(1));
    }

    #[test]
    fn shrinking_capacity_evicts_immediately() {
        let mut c = Lru::new(1000);
        for k in 0..10u64 {
            c.access(&Request::new(k as f64, k, 100));
        }
        assert_eq!(c.used_bytes(), 1000);
        c.set_capacity_bytes(400);
        assert!(c.used_bytes() <= 400);
        assert_eq!(c.len(), 4);
    }

    #[test]
    fn every_policy_holds_its_invariants() {
        for name in super::super::BASELINE_NAMES {
            let mut p = super::super::build(name, 1_000_000).expect("policy exists");
            assert_invariants(p.as_mut());
        }
    }

    #[test]
    fn every_policy_beats_chance_on_a_zipf_workload() {
        // A cache holding a tenth of a Zipf(1.0) key space should land well above zero and
        // well below one. Anything outside that is a broken implementation, not a policy
        // preference.
        for name in super::super::BASELINE_NAMES {
            let mut p = super::super::build(name, 1000 * 100).expect("policy exists");
            let hr = zipf_hit_rate(p.as_mut(), 10_000, 200_000, 100);
            assert!(
                (0.20..0.95).contains(&hr),
                "{name} produced an implausible hit rate of {hr:.3}"
            );
        }
    }
}
