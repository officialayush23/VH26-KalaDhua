use ahash::AHashMap;
use aura_core::types::{CostVector, KeyId, ObjectContext};

/// One resident object. `value` is kept opaque: the cache never inspects it.
#[derive(Debug, Clone)]
pub struct Entry {
    pub key: KeyId,
    pub value: serde_json::Value,
    pub size_bytes: u64,
    pub application: String,
    pub object_type: String,
    pub ttl_ms: f64,
    pub regen: CostVector,
    pub inserted_ms: f64,
    pub last_hit_ms: f64,
    pub hits: u64,
    pub score: f64,
}

impl Entry {
    pub fn age_ms(&self, now_ms: f64) -> f64 {
        (now_ms - self.inserted_ms).max(0.0)
    }

    pub fn ttl_remaining_frac(&self, now_ms: f64) -> f64 {
        if self.ttl_ms <= 0.0 {
            return 1.0;
        }
        (1.0 - self.age_ms(now_ms) / self.ttl_ms).clamp(0.0, 1.0)
    }

    pub fn expired(&self, now_ms: f64) -> bool {
        self.ttl_ms > 0.0 && self.age_ms(now_ms) >= self.ttl_ms
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct LayerStats {
    pub hits: u64,
    pub misses: u64,
    pub hit_bytes: u64,
    pub miss_bytes: u64,
}

impl LayerStats {
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

    pub fn record(&mut self, hit: bool, bytes: u64) {
        if hit {
            self.hits += 1;
            self.hit_bytes += bytes;
        } else {
            self.misses += 1;
            self.miss_bytes += bytes;
        }
    }
}

/// Size-aware store with an L1 admission window in front of L2. L1 is small and pure
/// recency; anything that survives it is a candidate for the scored L2.
#[derive(Debug)]
pub struct Store {
    entries: AHashMap<KeyId, Entry>,
    order: Vec<KeyId>,
    l1: AHashMap<KeyId, f64>,
    l1_order: Vec<KeyId>,
    l1_capacity: usize,
    capacity_bytes: u64,
    used_bytes: u64,
    pub l1_stats: LayerStats,
    pub l2_stats: LayerStats,
    pub cdn_stats: LayerStats,
    pub evictions: u64,
    pub admissions: u64,
    pub rejections: u64,
    pub expirations: u64,
}

impl Store {
    pub fn new(capacity_bytes: u64, l1_capacity: usize) -> Self {
        Self {
            entries: AHashMap::new(),
            order: Vec::new(),
            l1: AHashMap::new(),
            l1_order: Vec::new(),
            l1_capacity: l1_capacity.max(1),
            capacity_bytes,
            used_bytes: 0,
            l1_stats: LayerStats::default(),
            l2_stats: LayerStats::default(),
            cdn_stats: LayerStats::default(),
            evictions: 0,
            admissions: 0,
            rejections: 0,
            expirations: 0,
        }
    }

    pub fn capacity_bytes(&self) -> u64 {
        self.capacity_bytes
    }

    pub fn set_capacity(&mut self, bytes: u64) {
        self.capacity_bytes = bytes.max(1);
    }

    pub fn used_bytes(&self) -> u64 {
        self.used_bytes
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn pressure(&self) -> f64 {
        if self.capacity_bytes == 0 {
            1.0
        } else {
            (self.used_bytes as f64 / self.capacity_bytes as f64).clamp(0.0, 1.0)
        }
    }

    pub fn get(&self, key: KeyId) -> Option<&Entry> {
        self.entries.get(&key)
    }

    pub fn keys(&self) -> &[KeyId] {
        &self.order
    }

    pub fn entry_mut(&mut self, key: KeyId) -> Option<&mut Entry> {
        self.entries.get_mut(&key)
    }

    pub fn contains(&self, key: KeyId) -> bool {
        self.entries.contains_key(&key)
    }

    /// Record a touch in the L1 window. Returns true when the key was already there,
    /// which is the cheapest possible signal that a key is not a one-shot scan.
    pub fn touch_l1(&mut self, key: KeyId, now_ms: f64) -> bool {
        let seen = self.l1.insert(key, now_ms).is_some();
        if !seen {
            self.l1_order.push(key);
            while self.l1_order.len() > self.l1_capacity {
                let victim = self.l1_order.remove(0);
                self.l1.remove(&victim);
            }
        }
        seen
    }

    pub fn insert(&mut self, entry: Entry) {
        if let Some(old) = self.entries.remove(&entry.key) {
            self.used_bytes = self.used_bytes.saturating_sub(old.size_bytes);
            self.order.retain(|k| *k != entry.key);
        }
        self.used_bytes += entry.size_bytes;
        self.order.push(entry.key);
        self.entries.insert(entry.key, entry);
        self.admissions += 1;
    }

    pub fn remove(&mut self, key: KeyId) -> Option<Entry> {
        let e = self.entries.remove(&key)?;
        self.used_bytes = self.used_bytes.saturating_sub(e.size_bytes);
        self.order.retain(|k| *k != key);
        Some(e)
    }

    pub fn evict(&mut self, key: KeyId) -> Option<Entry> {
        let e = self.remove(key)?;
        self.evictions += 1;
        Some(e)
    }

    pub fn sweep_expired(&mut self, now_ms: f64) -> usize {
        let dead: Vec<KeyId> = self
            .entries
            .values()
            .filter(|e| e.expired(now_ms))
            .map(|e| e.key)
            .collect();
        let n = dead.len();
        for k in dead {
            self.remove(k);
        }
        self.expirations += n as u64;
        n
    }

    pub fn fits(&self, size_bytes: u64) -> bool {
        self.used_bytes + size_bytes <= self.capacity_bytes
    }

    pub fn needed_bytes(&self, size_bytes: u64) -> u64 {
        (self.used_bytes + size_bytes).saturating_sub(self.capacity_bytes)
    }

    pub fn context_of(&self, key: KeyId) -> Option<ObjectContext> {
        let e = self.entries.get(&key)?;
        Some(
            ObjectContext::new(&e.application, &e.object_type, e.size_bytes)
                .with_regen(e.regen)
                .with_ttl(e.ttl_ms),
        )
    }

    pub fn total_hits(&self) -> u64 {
        self.l1_stats.hits + self.l2_stats.hits
    }
}
