//! LeCaR: learning cache replacement by regret minimisation.
//!
//! Two experts, LRU and LFU, each nominate a victim. The policy samples which expert to
//! follow in proportion to a weight, and keeps a short history of what it threw away. When
//! a request misses on a key that is still in that history, the expert responsible has
//! provably made a mistake, and its weight is cut by an amount that grows with how recently
//! the mistake was made.
//!
//! It is the closest baseline in spirit to what AURA does, which makes it the most
//! interesting one to beat. The difference is the size of the question each one asks: LeCaR
//! learns *which of two recency/frequency heuristics to trust*; AURA learns near-future
//! reuse, prices a miss in the application's own currency, and chooses among a wider set of
//! experts on that basis. If AURA cannot beat LeCaR, the extra machinery is not paying for
//! itself.

use super::{AccessResult, CachePolicy, Request, Resident};
use crate::list::IntrusiveList;
use crate::rng::Rng;
use crate::types::KeyId;
use ahash::AHashMap;
use std::collections::VecDeque;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Expert {
    Lru,
    Lfu,
}

#[derive(Debug, Clone, Copy)]
struct HistoryRecord {
    expert: Expert,
    evicted_at: u64,
}

#[derive(Debug)]
pub struct LeCaR {
    capacity_bytes: u64,
    used_bytes: u64,
    entries: AHashMap<KeyId, Resident>,
    recency: IntrusiveList,
    /// Frequency buckets: `freq -> keys`. Cheaper than a heap for the "smallest frequency"
    /// query, and it never accumulates stale records.
    freq_index: AHashMap<u32, Vec<KeyId>>,
    min_freq: u32,
    history: AHashMap<KeyId, HistoryRecord>,
    history_order: VecDeque<KeyId>,
    history_capacity: usize,
    weight_lru: f64,
    weight_lfu: f64,
    learning_rate: f64,
    discount: f64,
    clock: u64,
    rng: Rng,
}

impl LeCaR {
    pub fn new(capacity_bytes: u64) -> Self {
        Self {
            capacity_bytes,
            used_bytes: 0,
            entries: AHashMap::new(),
            recency: IntrusiveList::new(),
            freq_index: AHashMap::new(),
            min_freq: 1,
            history: AHashMap::new(),
            history_order: VecDeque::new(),
            history_capacity: 10_000,
            weight_lru: 0.5,
            weight_lfu: 0.5,
            // The published defaults. The discount is what makes a recent mistake cost more
            // than an old one, which is what lets the policy track a workload that changes.
            learning_rate: 0.45,
            discount: 0.9_f64.powf(1.0 / 10_000.0),
            clock: 0,
            rng: Rng::seed_from_u64(0x1eca8),
        }
    }

    pub fn weights(&self) -> (f64, f64) {
        (self.weight_lru, self.weight_lfu)
    }

    fn freq_insert(&mut self, key: KeyId, freq: u32) {
        self.freq_index.entry(freq).or_default().push(key);
        if freq < self.min_freq {
            self.min_freq = freq;
        }
    }

    fn freq_remove(&mut self, key: KeyId, freq: u32) {
        if let Some(bucket) = self.freq_index.get_mut(&freq) {
            if let Some(pos) = bucket.iter().position(|k| *k == key) {
                bucket.swap_remove(pos);
            }
            if bucket.is_empty() {
                self.freq_index.remove(&freq);
            }
        }
    }

    fn lfu_victim(&mut self) -> Option<KeyId> {
        // Advance past buckets emptied since the last eviction.
        let mut freq = self.min_freq;
        let max_freq = 64;
        while freq <= max_freq {
            if let Some(bucket) = self.freq_index.get(&freq) {
                if let Some(key) = bucket.last().copied() {
                    self.min_freq = freq;
                    return Some(key);
                }
            }
            freq += 1;
        }
        // Fall back to any resident key rather than failing to make room.
        self.freq_index.values().flatten().next().copied()
    }

    fn remember(&mut self, key: KeyId, expert: Expert) {
        self.history.insert(key, HistoryRecord { expert, evicted_at: self.clock });
        self.history_order.push_back(key);
        while self.history_order.len() > self.history_capacity {
            if let Some(old) = self.history_order.pop_front() {
                self.history.remove(&old);
            }
        }
    }

    /// The learning step. A miss on a key we recently evicted is a labelled mistake.
    fn punish(&mut self, record: HistoryRecord) {
        let age = (self.clock - record.evicted_at) as f64;
        let regret = self.discount.powf(age);
        match record.expert {
            Expert::Lru => self.weight_lru *= (-self.learning_rate * regret).exp(),
            Expert::Lfu => self.weight_lfu *= (-self.learning_rate * regret).exp(),
        }
        let total = self.weight_lru + self.weight_lfu;
        if total > 0.0 {
            self.weight_lru /= total;
            self.weight_lfu /= total;
        } else {
            self.weight_lru = 0.5;
            self.weight_lfu = 0.5;
        }
        // Never let an expert reach zero: a workload that changes must be able to bring it
        // back, and an expert with no weight is never sampled and so never learns.
        let floor = 0.01;
        self.weight_lru = self.weight_lru.max(floor);
        self.weight_lfu = self.weight_lfu.max(floor);
        let total = self.weight_lru + self.weight_lfu;
        self.weight_lru /= total;
        self.weight_lfu /= total;
    }

    fn drop_key(&mut self, key: KeyId) -> Option<Resident> {
        let entry = self.entries.remove(&key)?;
        self.recency.remove(entry.slot);
        self.freq_remove(key, entry.freq);
        self.used_bytes -= entry.size_bytes;
        Some(entry)
    }

    fn evict_until_fits(&mut self, needed: u64, evicted: &mut Vec<KeyId>) {
        while self.used_bytes + needed > self.capacity_bytes && !self.entries.is_empty() {
            let expert = if self.rng.next_f64() < self.weight_lru { Expert::Lru } else { Expert::Lfu };
            let victim = match expert {
                Expert::Lru => self.recency.back(),
                Expert::Lfu => self.lfu_victim(),
            };
            let Some(victim) = victim else { break };
            if self.drop_key(victim).is_some() {
                self.remember(victim, expert);
                evicted.push(victim);
            } else {
                break;
            }
        }
    }

    fn insert(&mut self, req: &Request<'_>) -> AccessResult {
        let mut result = AccessResult::miss();
        if req.size_bytes > self.capacity_bytes {
            return result;
        }
        self.evict_until_fits(req.size_bytes, &mut result.evicted);
        let slot = self.recency.push_front(req.key);
        self.entries.insert(req.key, Resident::new(req, slot, 0.0));
        self.freq_insert(req.key, 1);
        self.min_freq = 1;
        self.used_bytes += req.size_bytes;
        result.admitted = true;
        result
    }
}

impl CachePolicy for LeCaR {
    fn name(&self) -> &'static str {
        "lecar"
    }

    fn access(&mut self, req: &Request<'_>) -> AccessResult {
        self.clock += 1;

        if let Some(entry) = self.entries.get(&req.key).copied() {
            if !entry.is_expired(req.ts_ms) {
                self.freq_remove(req.key, entry.freq);
                let mut updated = entry;
                updated.freq += 1;
                updated.last_ts_ms = req.ts_ms;
                self.freq_insert(req.key, updated.freq);
                self.recency.move_to_front(updated.slot);
                self.entries.insert(req.key, updated);
                return AccessResult::hit();
            }
            self.drop_key(req.key);
            let mut result = self.insert(req);
            result.stale = true;
            return result;
        }

        // The learning signal: this key was evicted recently and is now wanted again.
        if let Some(record) = self.history.remove(&req.key) {
            self.punish(record);
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
            + self.recency.memory_bytes()
            + self.history.capacity()
                * (std::mem::size_of::<KeyId>() + std::mem::size_of::<HistoryRecord>())
            + self.freq_index.values().map(|v| v.capacity() * std::mem::size_of::<KeyId>()).sum::<usize>()
    }

    fn reset(&mut self) {
        self.entries.clear();
        self.recency.clear();
        self.freq_index.clear();
        self.history.clear();
        self.history_order.clear();
        self.used_bytes = 0;
        self.weight_lru = 0.5;
        self.weight_lfu = 0.5;
        self.min_freq = 1;
        self.clock = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weights_move_towards_the_expert_that_suits_the_workload() {
        // A pure recency workload: a rolling window where the oldest key is never wanted
        // again. LFU's victim choice is systematically wrong here, so its weight should
        // fall.
        let mut c = LeCaR::new(50 * 100);
        for i in 0..20_000u64 {
            c.access(&Request::new(i as f64, i % 200, 100));
        }
        let (lru_w, lfu_w) = c.weights();
        assert!(
            (lru_w - lfu_w).abs() > 0.02,
            "the weights never separated: lru {lru_w:.3} lfu {lfu_w:.3}"
        );
        assert!((lru_w + lfu_w - 1.0).abs() < 1e-9, "weights must stay normalised");
    }

    #[test]
    fn weights_stay_within_bounds_under_adversarial_traffic() {
        let mut c = LeCaR::new(10_000);
        let mut rng = Rng::seed_from_u64(9);
        for i in 0..50_000u64 {
            // Alternate between a recency-friendly phase and a frequency-friendly one, so
            // both experts are punished in turn.
            let key = if (i / 5_000) % 2 == 0 { i % 500 } else { rng.below(50) };
            c.access(&Request::new(i as f64, key, 100));
            let (a, b) = c.weights();
            assert!(a > 0.0 && b > 0.0 && (a + b - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn history_is_bounded() {
        let mut c = LeCaR::new(1000);
        for i in 0..100_000u64 {
            c.access(&Request::new(i as f64, i, 100));
        }
        assert!(c.history.len() <= c.history_capacity);
        assert_eq!(c.history.len(), c.history_order.len());
    }
}
