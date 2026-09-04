//! Modern heuristic baselines: W-TinyLFU, S3-FIFO and SIEVE.
//!
//! These are much stronger opponents than LRU or LFU and, unlike them, they are defined by
//! their *structure* rather than by a score. Each is implemented with its own queues here,
//! adapted to byte-sized objects rather than fixed-size slots — a uniform-size benchmark
//! would flatter all three and hide the size-pollution behaviour the project cares about.

use super::{AccessResult, CachePolicy, Request, Resident};
use crate::list::IntrusiveList;
use crate::sketch::CountMinSketch;
use crate::types::KeyId;
use ahash::AHashMap;
use std::collections::VecDeque;

/// Which internal queue an entry currently lives in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Segment {
    Window,
    Probation,
    Protected,
}

/// Window TinyLFU, in the shape Caffeine popularised.
///
/// A small LRU window absorbs new arrivals so that a one-off burst never has to argue with
/// the main cache. When the window overflows, its victim is compared against the main
/// cache's victim using a frequency sketch, and only the more frequent of the two survives.
/// Separating *admission* from *eviction* like this is the single most valuable idea to
/// take from it, and AURA keeps the structure while replacing the sketch comparison with an
/// economic one.
#[derive(Debug)]
pub struct WTinyLfu {
    capacity_bytes: u64,
    used_bytes: u64,
    window_capacity: u64,
    protected_capacity: u64,
    window_bytes: u64,
    probation_bytes: u64,
    protected_bytes: u64,
    entries: AHashMap<KeyId, (Resident, Segment)>,
    window: IntrusiveList,
    probation: IntrusiveList,
    protected: IntrusiveList,
    sketch: CountMinSketch,
}

impl WTinyLfu {
    pub fn new(capacity_bytes: u64) -> Self {
        // Caffeine's default split: a 1% window, and 80% of the main cache protected.
        let window_capacity = (capacity_bytes / 100).max(1);
        let main = capacity_bytes.saturating_sub(window_capacity);
        Self {
            capacity_bytes,
            used_bytes: 0,
            window_capacity,
            protected_capacity: (main * 8) / 10,
            window_bytes: 0,
            probation_bytes: 0,
            protected_bytes: 0,
            entries: AHashMap::new(),
            window: IntrusiveList::new(),
            probation: IntrusiveList::new(),
            protected: IntrusiveList::new(),
            sketch: CountMinSketch::new(100_000),
        }
    }

    fn list_mut(&mut self, segment: Segment) -> &mut IntrusiveList {
        match segment {
            Segment::Window => &mut self.window,
            Segment::Probation => &mut self.probation,
            Segment::Protected => &mut self.protected,
        }
    }

    fn add_bytes(&mut self, segment: Segment, delta: i64) {
        let field = match segment {
            Segment::Window => &mut self.window_bytes,
            Segment::Probation => &mut self.probation_bytes,
            Segment::Protected => &mut self.protected_bytes,
        };
        *field = (*field as i64 + delta).max(0) as u64;
    }

    fn detach(&mut self, key: KeyId) -> Option<(Resident, Segment)> {
        let (entry, segment) = self.entries.remove(&key)?;
        self.list_mut(segment).remove(entry.slot);
        self.add_bytes(segment, -(entry.size_bytes as i64));
        self.used_bytes -= entry.size_bytes;
        Some((entry, segment))
    }

    fn attach(&mut self, key: KeyId, mut entry: Resident, segment: Segment) {
        entry.slot = self.list_mut(segment).push_front(key);
        self.add_bytes(segment, entry.size_bytes as i64);
        self.used_bytes += entry.size_bytes;
        self.entries.insert(key, (entry, segment));
    }

    /// Demote from the protected segment until it fits. Demotion is not eviction: the
    /// object moves to probation, where it is a candidate but not yet gone.
    fn drain_protected(&mut self) {
        while self.protected_bytes > self.protected_capacity {
            let Some(victim) = self.protected.back() else { break };
            let Some((entry, _)) = self.detach(victim) else { break };
            self.attach(victim, entry, Segment::Probation);
        }
    }

    /// The admission contest that gives the policy its name.
    fn evict_from_main(&mut self, needed: u64, evicted: &mut Vec<KeyId>) {
        while self.probation_bytes + self.protected_bytes + needed
            > self.capacity_bytes.saturating_sub(self.window_capacity)
        {
            let victim = self.probation.back().or_else(|| self.protected.back());
            let Some(victim) = victim else { break };
            if let Some((entry, _)) = self.detach(victim) {
                evicted.push(victim);
                let _ = entry;
            } else {
                break;
            }
        }
    }

    fn admit_from_window(&mut self, evicted: &mut Vec<KeyId>) {
        while self.window_bytes > self.window_capacity {
            let Some(candidate) = self.window.back() else { break };
            let Some((entry, _)) = self.detach(candidate) else { break };

            let main_victim = self.probation.back().or_else(|| self.protected.back());
            let main_limit = self.capacity_bytes.saturating_sub(self.window_capacity);
            let fits = self.probation_bytes + self.protected_bytes + entry.size_bytes <= main_limit;

            if fits {
                self.attach(candidate, entry, Segment::Probation);
                continue;
            }

            match main_victim {
                Some(victim) => {
                    let candidate_freq = self.sketch.estimate(candidate);
                    let victim_freq = self.sketch.estimate(victim);
                    if candidate_freq > victim_freq {
                        // Make room for the winner, then install it.
                        let mut dropped = Vec::new();
                        self.evict_from_main(entry.size_bytes, &mut dropped);
                        evicted.extend(dropped);
                        self.attach(candidate, entry, Segment::Probation);
                    } else {
                        // The candidate loses the contest and is simply not admitted.
                        evicted.push(candidate);
                    }
                }
                None => evicted.push(candidate),
            }
        }
    }

    fn insert(&mut self, req: &Request<'_>) -> AccessResult {
        let mut result = AccessResult::miss();
        if req.size_bytes > self.capacity_bytes {
            return result;
        }
        self.attach(req.key, Resident::new(req, 0, 0.0), Segment::Window);
        self.admit_from_window(&mut result.evicted);
        self.drain_protected();
        result.admitted = self.entries.contains_key(&req.key);
        result
    }
}

impl CachePolicy for WTinyLfu {
    fn name(&self) -> &'static str {
        "tinylfu"
    }

    fn access(&mut self, req: &Request<'_>) -> AccessResult {
        // The sketch records every request, hit or miss. It is a record of demand, not of
        // what the cache happened to hold.
        self.sketch.increment(req.key);

        if let Some((entry, segment)) = self.entries.get(&req.key).copied() {
            if !entry.is_expired(req.ts_ms) {
                let mut entry = entry;
                entry.freq += 1;
                entry.last_ts_ms = req.ts_ms;
                match segment {
                    Segment::Window => {
                        self.window.move_to_front(entry.slot);
                        self.entries.insert(req.key, (entry, segment));
                    }
                    Segment::Protected => {
                        self.protected.move_to_front(entry.slot);
                        self.entries.insert(req.key, (entry, segment));
                    }
                    Segment::Probation => {
                        // A second hit promotes the object out of probation.
                        self.detach(req.key);
                        self.attach(req.key, entry, Segment::Protected);
                        self.drain_protected();
                    }
                }
                return AccessResult::hit();
            }
            self.detach(req.key);
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
        self.window_capacity = (bytes / 100).max(1);
        let main = bytes.saturating_sub(self.window_capacity);
        self.protected_capacity = (main * 8) / 10;
        let mut dropped = Vec::new();
        self.evict_from_main(0, &mut dropped);
        self.admit_from_window(&mut dropped);
        self.drain_protected();
        while self.used_bytes > self.capacity_bytes {
            let victim = self
                .probation
                .back()
                .or_else(|| self.protected.back())
                .or_else(|| self.window.back());
            let Some(victim) = victim else { break };
            self.detach(victim);
        }
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
        self.entries.capacity()
            * (std::mem::size_of::<KeyId>() + std::mem::size_of::<(Resident, Segment)>())
            + self.window.memory_bytes()
            + self.probation.memory_bytes()
            + self.protected.memory_bytes()
            + self.sketch.memory_bytes()
    }

    fn reset(&mut self) {
        self.entries.clear();
        self.window.clear();
        self.probation.clear();
        self.protected.clear();
        self.used_bytes = 0;
        self.window_bytes = 0;
        self.probation_bytes = 0;
        self.protected_bytes = 0;
    }
}

/// S3-FIFO: a small probationary FIFO, a large main FIFO, and a ghost queue.
///
/// The observation it is built on is that most objects in a web workload are requested
/// exactly once, so the cache should make them prove themselves in a small queue before
/// they are allowed to occupy the main one. Objects evicted from the small queue leave a
/// ghost record; a later hit on a ghost means the object was worth keeping after all and it
/// enters the main queue directly.
#[derive(Debug)]
pub struct S3Fifo {
    capacity_bytes: u64,
    used_bytes: u64,
    small_capacity: u64,
    small_bytes: u64,
    main_bytes: u64,
    entries: AHashMap<KeyId, (Resident, bool)>,
    small: VecDeque<KeyId>,
    main: VecDeque<KeyId>,
    ghost: VecDeque<KeyId>,
    ghost_set: ahash::AHashSet<KeyId>,
    ghost_capacity: usize,
}

impl S3Fifo {
    pub fn new(capacity_bytes: u64) -> Self {
        Self {
            capacity_bytes,
            used_bytes: 0,
            // The published split: 10% small, 90% main.
            small_capacity: (capacity_bytes / 10).max(1),
            small_bytes: 0,
            main_bytes: 0,
            entries: AHashMap::new(),
            small: VecDeque::new(),
            main: VecDeque::new(),
            ghost: VecDeque::new(),
            ghost_set: ahash::AHashSet::new(),
            ghost_capacity: 10_000,
        }
    }

    fn remember_ghost(&mut self, key: KeyId) {
        if self.ghost_set.insert(key) {
            self.ghost.push_back(key);
            while self.ghost.len() > self.ghost_capacity {
                if let Some(old) = self.ghost.pop_front() {
                    self.ghost_set.remove(&old);
                }
            }
        }
    }

    /// Evict from the small queue. An object accessed at least twice is promoted to main
    /// instead of being dropped; everything else leaves a ghost.
    fn evict_small(&mut self, evicted: &mut Vec<KeyId>) {
        while self.small_bytes > self.small_capacity {
            let Some(key) = self.small.pop_front() else { break };
            let Some((entry, _)) = self.entries.get(&key).copied() else { continue };
            self.small_bytes -= entry.size_bytes;
            if entry.freq >= 2 {
                self.main.push_back(key);
                self.main_bytes += entry.size_bytes;
                self.entries.insert(key, (entry, true));
                self.evict_main(evicted);
            } else {
                self.entries.remove(&key);
                self.used_bytes -= entry.size_bytes;
                self.remember_ghost(key);
                evicted.push(key);
            }
        }
    }

    /// Evict from the main queue, giving each object one second chance per pass.
    fn evict_main(&mut self, evicted: &mut Vec<KeyId>) {
        let main_capacity = self.capacity_bytes.saturating_sub(self.small_capacity);
        let mut guard = self.main.len() * 2 + 8;
        while self.main_bytes > main_capacity && guard > 0 {
            guard -= 1;
            let Some(key) = self.main.pop_front() else { break };
            let Some((mut entry, _)) = self.entries.get(&key).copied() else { continue };
            if entry.freq > 1 {
                entry.freq -= 1;
                self.entries.insert(key, (entry, true));
                self.main.push_back(key);
            } else {
                self.main_bytes -= entry.size_bytes;
                self.used_bytes -= entry.size_bytes;
                self.entries.remove(&key);
                evicted.push(key);
            }
        }
    }

    fn insert(&mut self, req: &Request<'_>) -> AccessResult {
        let mut result = AccessResult::miss();
        if req.size_bytes > self.capacity_bytes {
            return result;
        }
        // A ghost hit is evidence the object deserves the main queue directly.
        let seen_before = self.ghost_set.remove(&req.key);
        let entry = Resident::new(req, 0, 0.0);
        if seen_before {
            self.main.push_back(req.key);
            self.main_bytes += req.size_bytes;
            self.entries.insert(req.key, (entry, true));
            self.used_bytes += req.size_bytes;
            self.evict_main(&mut result.evicted);
        } else {
            self.small.push_back(req.key);
            self.small_bytes += req.size_bytes;
            self.entries.insert(req.key, (entry, false));
            self.used_bytes += req.size_bytes;
            self.evict_small(&mut result.evicted);
        }
        result.admitted = self.entries.contains_key(&req.key);
        result
    }

    fn drop_key(&mut self, key: KeyId) {
        if let Some((entry, in_main)) = self.entries.remove(&key) {
            self.used_bytes -= entry.size_bytes;
            if in_main {
                self.main_bytes -= entry.size_bytes;
                self.main.retain(|k| *k != key);
            } else {
                self.small_bytes -= entry.size_bytes;
                self.small.retain(|k| *k != key);
            }
        }
    }
}

impl CachePolicy for S3Fifo {
    fn name(&self) -> &'static str {
        "s3fifo"
    }

    fn access(&mut self, req: &Request<'_>) -> AccessResult {
        if let Some((entry, in_main)) = self.entries.get(&req.key).copied() {
            if !entry.is_expired(req.ts_ms) {
                let mut entry = entry;
                // Frequency saturates at three: S3-FIFO deliberately keeps a two-bit
                // counter, which is enough to separate "once" from "more than once"
                // without letting a historical spike dominate.
                entry.freq = (entry.freq + 1).min(3);
                entry.last_ts_ms = req.ts_ms;
                self.entries.insert(req.key, (entry, in_main));
                return AccessResult::hit();
            }
            self.drop_key(req.key);
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
        self.small_capacity = (bytes / 10).max(1);
        let mut dropped = Vec::new();
        self.evict_small(&mut dropped);
        self.evict_main(&mut dropped);
        while self.used_bytes > self.capacity_bytes {
            let key = self.small.front().copied().or_else(|| self.main.front().copied());
            let Some(key) = key else { break };
            self.drop_key(key);
        }
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
        self.entries.capacity() * (std::mem::size_of::<KeyId>() + std::mem::size_of::<(Resident, bool)>())
            + (self.small.capacity() + self.main.capacity() + self.ghost.capacity())
                * std::mem::size_of::<KeyId>()
            + self.ghost_set.capacity() * std::mem::size_of::<KeyId>()
    }

    fn reset(&mut self) {
        self.entries.clear();
        self.small.clear();
        self.main.clear();
        self.ghost.clear();
        self.ghost_set.clear();
        self.used_bytes = 0;
        self.small_bytes = 0;
        self.main_bytes = 0;
    }
}

/// SIEVE: FIFO order, one visited bit per object, and a hand that sweeps for a victim.
///
/// It is the simplest policy here — no promotion, no reordering on a hit, one bit of state
/// per object — and it is very hard to beat on skewed web workloads. Including it keeps the
/// comparison honest: if a learned policy cannot beat two hundred lines of FIFO with a
/// bit, the learning is not earning its keep.
#[derive(Debug)]
pub struct Sieve {
    capacity_bytes: u64,
    used_bytes: u64,
    entries: AHashMap<KeyId, (Resident, bool)>,
    order: IntrusiveList,
    /// Sweeps from the oldest end towards the newest, and wraps.
    hand: Option<usize>,
}

impl Sieve {
    pub fn new(capacity_bytes: u64) -> Self {
        Self {
            capacity_bytes,
            used_bytes: 0,
            entries: AHashMap::new(),
            order: IntrusiveList::new(),
            hand: None,
        }
    }

    fn evict_until_fits(&mut self, needed: u64, evicted: &mut Vec<KeyId>) {
        while self.used_bytes + needed > self.capacity_bytes && !self.order.is_empty() {
            let mut cursor = self.hand.or_else(|| self.order.back_slot());
            let mut guard = self.order.len() * 2 + 4;
            loop {
                guard -= 1;
                let Some(slot) = cursor else {
                    cursor = self.order.back_slot();
                    if cursor.is_none() {
                        return;
                    }
                    continue;
                };
                if guard == 0 {
                    // Every object has been given a chance; take the oldest unconditionally
                    // so the loop always terminates.
                    let Some(victim) = self.order.back() else { return };
                    self.remove_key(victim, evicted);
                    self.hand = None;
                    break;
                }
                let Some(key) = self.order.key_at(slot) else {
                    cursor = self.order.prev_of(slot);
                    continue;
                };
                let visited = self.entries.get(&key).map(|(_, v)| *v).unwrap_or(false);
                if visited {
                    // Clear the bit and keep sweeping: a second chance, not a reprieve.
                    if let Some(e) = self.entries.get_mut(&key) {
                        e.1 = false;
                    }
                    cursor = self.order.prev_of(slot);
                } else {
                    let next = self.order.prev_of(slot);
                    self.remove_key(key, evicted);
                    self.hand = next;
                    break;
                }
            }
        }
    }

    fn remove_key(&mut self, key: KeyId, evicted: &mut Vec<KeyId>) {
        if let Some((entry, _)) = self.entries.remove(&key) {
            self.order.remove(entry.slot);
            self.used_bytes -= entry.size_bytes;
            evicted.push(key);
        }
    }

    fn insert(&mut self, req: &Request<'_>) -> AccessResult {
        let mut result = AccessResult::miss();
        if req.size_bytes > self.capacity_bytes {
            return result;
        }
        self.evict_until_fits(req.size_bytes, &mut result.evicted);
        let slot = self.order.push_front(req.key);
        self.entries.insert(req.key, (Resident::new(req, slot, 0.0), false));
        self.used_bytes += req.size_bytes;
        result.admitted = true;
        result
    }
}

impl CachePolicy for Sieve {
    fn name(&self) -> &'static str {
        "sieve"
    }

    fn access(&mut self, req: &Request<'_>) -> AccessResult {
        if let Some((entry, visited)) = self.entries.get_mut(&req.key) {
            if !entry.is_expired(req.ts_ms) {
                entry.freq += 1;
                entry.last_ts_ms = req.ts_ms;
                // The whole policy: a hit sets a bit. Nothing moves.
                *visited = true;
                return AccessResult::hit();
            }
            let key = req.key;
            let mut dropped = Vec::new();
            self.remove_key(key, &mut dropped);
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
        self.entries.capacity()
            * (std::mem::size_of::<KeyId>() + std::mem::size_of::<(Resident, bool)>())
            + self.order.memory_bytes()
    }

    fn reset(&mut self) {
        self.entries.clear();
        self.order.clear();
        self.used_bytes = 0;
        self.hand = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The scan is the point. A million single-use keys should not be able to flush a
    /// working set out of a policy that has an admission filter or a probationary queue,
    /// whereas LRU loses everything.
    fn scan_resistance(policy: &mut dyn CachePolicy) -> f64 {
        let hot: Vec<KeyId> = (0..50).collect();
        // Warm the working set until it is thoroughly established.
        for round in 0..40 {
            for k in &hot {
                policy.access(&Request::new((round * 100 + *k) as f64, *k, 1000));
            }
        }
        // Now scan through keys that are never requested twice.
        for i in 0..5_000u64 {
            policy.access(&Request::new(100_000.0 + i as f64, 1_000_000 + i, 1000));
        }
        // How much of the working set survived?
        hot.iter().filter(|k| policy.contains(**k)).count() as f64 / hot.len() as f64
    }

    #[test]
    fn tinylfu_survives_a_scan_that_destroys_lru() {
        let mut lru = super::super::Lru::new(100_000);
        let mut tiny = WTinyLfu::new(100_000);
        let lru_survival = scan_resistance(&mut lru);
        let tiny_survival = scan_resistance(&mut tiny);
        assert!(
            tiny_survival > lru_survival,
            "W-TinyLFU kept {tiny_survival:.2} of the working set, LRU kept {lru_survival:.2}"
        );
    }

    #[test]
    fn s3fifo_survives_a_scan() {
        let mut lru = super::super::Lru::new(100_000);
        let mut s3 = S3Fifo::new(100_000);
        assert!(scan_resistance(&mut s3) >= scan_resistance(&mut lru));
    }

    #[test]
    fn sieve_survives_a_scan() {
        let mut lru = super::super::Lru::new(100_000);
        let mut sieve = Sieve::new(100_000);
        assert!(scan_resistance(&mut sieve) >= scan_resistance(&mut lru));
    }

    #[test]
    fn sieve_keeps_visited_objects_over_unvisited_ones() {
        let mut c = Sieve::new(300);
        c.access(&Request::new(0.0, 1, 100));
        c.access(&Request::new(1.0, 2, 100));
        c.access(&Request::new(2.0, 3, 100));
        c.access(&Request::new(3.0, 1, 100)); // sets the visited bit on key 1
        let result = c.access(&Request::new(4.0, 4, 100));
        assert!(!result.evicted.contains(&1), "the visited object should have survived");
    }

    #[test]
    fn s3fifo_promotes_on_a_second_access() {
        let mut c = S3Fifo::new(10_000);
        c.access(&Request::new(0.0, 1, 100));
        c.access(&Request::new(1.0, 1, 100));
        // Flood the small queue; key 1 has been seen twice so it should be promoted rather
        // than dropped.
        for i in 0..200u64 {
            c.access(&Request::new(10.0 + i as f64, 1000 + i, 100));
        }
        assert!(c.contains(1), "a twice-accessed object was dropped from the small queue");
    }

    #[test]
    fn tinylfu_promotes_from_probation_to_protected() {
        let mut c = WTinyLfu::new(100_000);
        c.access(&Request::new(0.0, 1, 1000));
        for i in 0..300u64 {
            c.access(&Request::new(1.0 + i as f64, 1000 + i, 1000));
        }
        // Access key 1 repeatedly so it earns protection, then flood again.
        for i in 0..20 {
            c.access(&Request::new(1000.0 + i as f64, 1, 1000));
        }
        for i in 0..300u64 {
            c.access(&Request::new(2000.0 + i as f64, 5000 + i, 1000));
        }
        assert!(c.contains(1), "a repeatedly accessed object lost its protection");
    }
}
