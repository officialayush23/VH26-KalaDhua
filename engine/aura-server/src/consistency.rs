//! Correctness: knowing when a cached value has stopped being true.
//!
//! Everything else in this engine is optimisation — how much a value is worth, whether it
//! earns its space. This file is the part that is not negotiable. A cache that serves a
//! price of ₹40 after the database says ₹20 is not a fast cache, it is a broken one, and no
//! score is high enough to justify it.
//!
//! So validity is checked *before* value, and the two never mix:
//!
//! ```text
//!   request ──► is it still valid? ──no──► invalidate, go to origin
//!                      │
//!                     yes
//!                      │
//!                      ▼
//!               is it worth keeping?   ◄── the ML and economics live here
//! ```
//!
//! Three mechanisms, because one is never enough:
//!
//! - **Dependency tags.** An object declares what it was derived from — `row:product:1292`,
//!   `table:orders`. One write invalidates every object downstream of it, however many
//!   there are and whoever built them. CDNs have called this surrogate keys for a decade.
//! - **Namespace versions.** Bumping `recommendation` from `v7` to `v8` retires every
//!   object in that namespace at once, without touching a single key. This is how a model
//!   redeploy is handled: not by flushing half a gigabyte and stampeding the origin, but by
//!   letting the old generation age out while new requests build the new one.
//! - **TTL, hard and soft.** Below the soft threshold an object is fresh. Between soft and
//!   hard it may still be served *once* while a rebuild runs behind it, which is what keeps
//!   a popular key from turning into a thundering herd the moment it expires. Past hard, it
//!   is gone.

use std::collections::HashSet;

use ahash::AHashMap;
use aura_core::types::KeyId;
use serde::Serialize;

/// Why an object left the cache. These are three different events and the telemetry must
/// not merge them: eviction means we were short of space, invalidation means we were
/// wrong, expiry means time passed. A dashboard that shows one number for all three hides
/// the only one that indicates a bug.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Removal {
    Evicted,
    Invalidated,
    Expired,
}

impl Removal {
    pub fn as_str(self) -> &'static str {
        match self {
            Removal::Evicted => "evicted",
            Removal::Invalidated => "invalidated",
            Removal::Expired => "expired",
        }
    }
}

/// How hard an invalidation is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum InvalidationMode {
    /// Remove immediately. For anything where being wrong is not acceptable — prices,
    /// permissions, balances.
    Hard,
    /// Mark stale. The next request is served the old value once and triggers a rebuild.
    /// Correct for derived, tolerant data — a dashboard rollup, a recommendation list —
    /// where one slightly stale answer is much cheaper than a stampede.
    Soft,
}

/// Freshness state of a resident object.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Freshness {
    Fresh,
    /// Past the soft threshold or softly invalidated: serve once, rebuild behind it.
    Stale,
    Expired,
}

/// Result of one invalidation request.
#[derive(Debug, Clone, Serialize)]
pub struct InvalidationResult {
    pub tags: Vec<String>,
    pub keys_hard: Vec<KeyId>,
    pub keys_soft: Vec<KeyId>,
    pub matched: usize,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct ConsistencyStats {
    pub tracked_keys: usize,
    pub tracked_tags: usize,
    pub namespaces: usize,
    pub invalidations: u64,
    pub keys_invalidated: u64,
    pub soft_invalidations: u64,
    pub version_bumps: u64,
    pub stale_serves: u64,
    pub expired: u64,
    pub evicted: u64,
}

/// The dependency index and the version register.
#[derive(Debug, Default)]
pub struct Consistency {
    tag_to_keys: AHashMap<String, HashSet<KeyId>>,
    key_to_tags: AHashMap<KeyId, Vec<String>>,
    /// Keys softly invalidated and awaiting rebuild.
    stale: HashSet<KeyId>,
    /// Namespace to its currently active version.
    versions: AHashMap<String, u64>,
    /// Fraction of the TTL at which an object becomes soft-stale. 0.8 means the last fifth
    /// of the lifetime is the refresh-ahead window.
    soft_ttl_fraction: f64,
    stats: ConsistencyStats,
    recent: std::collections::VecDeque<InvalidationEvent>,
}

#[derive(Debug, Clone, Serialize)]
pub struct InvalidationEvent {
    pub t: f64,
    pub source: String,
    pub tag: String,
    pub mode: InvalidationMode,
    pub keys: usize,
}

impl Consistency {
    pub fn new(soft_ttl_fraction: f64) -> Self {
        Self {
            soft_ttl_fraction: soft_ttl_fraction.clamp(0.0, 1.0),
            recent: std::collections::VecDeque::with_capacity(64),
            ..Default::default()
        }
    }

    /// Record what an object was derived from. Called on every admission.
    ///
    /// Tags are free-form strings the application chooses, because only the application
    /// knows what its objects depend on. The engine never interprets them — `row:product:42`
    /// and `tenant:acme` are the same kind of thing to it.
    pub fn register(&mut self, key: KeyId, tags: &[String]) {
        if tags.is_empty() {
            return;
        }
        self.forget(key);
        for tag in tags {
            self.tag_to_keys.entry(tag.clone()).or_default().insert(key);
        }
        self.key_to_tags.insert(key, tags.to_vec());
    }

    /// Drop all dependency records for a key. Called whenever it leaves the cache, for any
    /// of the three reasons.
    pub fn forget(&mut self, key: KeyId) {
        self.stale.remove(&key);
        let Some(tags) = self.key_to_tags.remove(&key) else { return };
        for tag in tags {
            let empty = match self.tag_to_keys.get_mut(&tag) {
                Some(set) => {
                    set.remove(&key);
                    set.is_empty()
                }
                None => false,
            };
            if empty {
                self.tag_to_keys.remove(&tag);
            }
        }
    }

    pub fn tags_of(&self, key: KeyId) -> Option<&[String]> {
        self.key_to_tags.get(&key).map(|v| v.as_slice())
    }

    pub fn keys_for_tag(&self, tag: &str) -> Vec<KeyId> {
        self.tag_to_keys.get(tag).map(|s| s.iter().copied().collect()).unwrap_or_default()
    }

    /// Invalidate everything downstream of these tags.
    ///
    /// This is the whole answer to "the price changed in the database and the cache did not
    /// notice". The write emits one tag; the index turns it into every affected key,
    /// however many rollups and rankings were built from that row.
    pub fn invalidate(
        &mut self,
        tags: &[String],
        mode: InvalidationMode,
        source: &str,
        now_ms: f64,
    ) -> InvalidationResult {
        let mut hard = Vec::new();
        let mut soft = Vec::new();

        for tag in tags {
            let keys = self.keys_for_tag(tag);
            match mode {
                InvalidationMode::Hard => {
                    for k in &keys {
                        self.forget(*k);
                    }
                    hard.extend(keys.iter().copied());
                }
                InvalidationMode::Soft => {
                    for k in &keys {
                        self.stale.insert(*k);
                    }
                    soft.extend(keys.iter().copied());
                }
            }
            self.push_event(InvalidationEvent {
                t: now_ms,
                source: source.to_string(),
                tag: tag.clone(),
                mode,
                keys: keys.len(),
            });
        }

        hard.sort_unstable();
        hard.dedup();
        soft.sort_unstable();
        soft.dedup();

        self.stats.invalidations += 1;
        self.stats.keys_invalidated += hard.len() as u64;
        self.stats.soft_invalidations += soft.len() as u64;

        InvalidationResult {
            matched: hard.len() + soft.len(),
            tags: tags.to_vec(),
            keys_hard: hard,
            keys_soft: soft,
        }
    }

    /// Current version of a namespace. Unknown namespaces start at 1 rather than 0, so a
    /// key built before anyone bumped anything is still well formed.
    pub fn version(&self, namespace: &str) -> u64 {
        self.versions.get(namespace).copied().unwrap_or(1)
    }

    /// Retire a whole generation of objects without touching a single key.
    ///
    /// A recommendation model redeploy is the motivating case. Deleting every affected
    /// object would empty a large fraction of the cache at once and send the entire miss
    /// stream at the ranking service — the cache causing the outage it exists to prevent.
    /// Bumping the version means new requests carry `:v8`, miss cleanly, and the `:v7`
    /// generation ages out under ordinary eviction pressure, which is exactly the gradual
    /// behaviour we want.
    pub fn bump_version(&mut self, namespace: &str, now_ms: f64) -> u64 {
        let next = self.version(namespace) + 1;
        self.versions.insert(namespace.to_string(), next);
        self.stats.version_bumps += 1;
        self.push_event(InvalidationEvent {
            t: now_ms,
            source: "version_bump".to_string(),
            tag: format!("{namespace}:v{next}"),
            mode: InvalidationMode::Soft,
            keys: 0,
        });
        next
    }

    /// Every namespace and the generation it is currently on.
    ///
    /// A `BTreeMap` rather than the internal hash map: this goes straight into telemetry,
    /// and a dashboard that reorders its own rows every frame is unreadable. Ordering it
    /// once here is cheaper than sorting it at every reader.
    pub fn versions(&self) -> std::collections::BTreeMap<String, u64> {
        self.versions.iter().map(|(k, v)| (k.clone(), *v)).collect()
    }

    /// Freshness of a resident object, given when it was filled and its TTL.
    ///
    /// `ttl_ms <= 0` means the object never expires on its own and can only leave by
    /// eviction or invalidation.
    pub fn freshness(&self, key: KeyId, inserted_ms: f64, ttl_ms: f64, now_ms: f64) -> Freshness {
        if self.stale.contains(&key) {
            return Freshness::Stale;
        }
        if ttl_ms <= 0.0 {
            return Freshness::Fresh;
        }
        let age = now_ms - inserted_ms;
        if age >= ttl_ms {
            Freshness::Expired
        } else if age >= ttl_ms * self.soft_ttl_fraction {
            Freshness::Stale
        } else {
            Freshness::Fresh
        }
    }

    pub fn is_stale(&self, key: KeyId) -> bool {
        self.stale.contains(&key)
    }

    /// Clear the stale mark once a rebuild has replaced the value.
    pub fn mark_rebuilt(&mut self, key: KeyId) {
        self.stale.remove(&key);
    }

    pub fn note_stale_serve(&mut self) {
        self.stats.stale_serves += 1;
    }

    pub fn note_removal(&mut self, reason: Removal) {
        match reason {
            Removal::Evicted => self.stats.evicted += 1,
            Removal::Expired => self.stats.expired += 1,
            Removal::Invalidated => {}
        }
    }

    fn push_event(&mut self, ev: InvalidationEvent) {
        if self.recent.len() >= 64 {
            self.recent.pop_front();
        }
        self.recent.push_back(ev);
    }

    pub fn recent_events(&self) -> Vec<InvalidationEvent> {
        self.recent.iter().cloned().collect()
    }

    pub fn stats(&self) -> ConsistencyStats {
        ConsistencyStats {
            tracked_keys: self.key_to_tags.len(),
            tracked_tags: self.tag_to_keys.len(),
            namespaces: self.versions.len(),
            ..self.stats.clone()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tags(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn one_row_change_invalidates_every_object_derived_from_it() {
        let mut c = Consistency::new(0.8);
        // Three rollups and two rankings, all built from product 1292.
        c.register(1, &tags(&["table:orders", "row:product:1292"]));
        c.register(2, &tags(&["row:product:1292"]));
        c.register(3, &tags(&["row:product:1292", "row:region:14"]));
        c.register(4, &tags(&["row:product:77"]));

        let r = c.invalidate(&tags(&["row:product:1292"]), InvalidationMode::Hard, "postgres", 0.0);
        assert_eq!(r.keys_hard.len(), 3);
        assert!(r.keys_hard.contains(&1) && r.keys_hard.contains(&2) && r.keys_hard.contains(&3));
        assert!(!r.keys_hard.contains(&4), "an unrelated object was invalidated");
        // The index is cleaned up, so a second invalidation matches nothing.
        assert_eq!(
            c.invalidate(&tags(&["row:product:1292"]), InvalidationMode::Hard, "postgres", 0.0).matched,
            0
        );
    }

    #[test]
    fn soft_invalidation_marks_rather_than_removes() {
        let mut c = Consistency::new(0.8);
        c.register(1, &tags(&["table:orders"]));
        let r = c.invalidate(&tags(&["table:orders"]), InvalidationMode::Soft, "sdk", 0.0);
        assert_eq!(r.keys_soft, vec![1]);
        assert!(c.is_stale(1));
        assert_eq!(c.freshness(1, 0.0, 600_000.0, 1_000.0), Freshness::Stale);
        c.mark_rebuilt(1);
        assert_eq!(c.freshness(1, 0.0, 600_000.0, 1_000.0), Freshness::Fresh);
    }

    #[test]
    fn ttl_has_a_soft_window_before_it_expires() {
        let c = Consistency::new(0.8);
        // TTL 1000 ms, soft at 800 ms.
        assert_eq!(c.freshness(1, 0.0, 1_000.0, 500.0), Freshness::Fresh);
        assert_eq!(c.freshness(1, 0.0, 1_000.0, 850.0), Freshness::Stale);
        assert_eq!(c.freshness(1, 0.0, 1_000.0, 1_000.0), Freshness::Expired);
    }

    #[test]
    fn objects_without_a_ttl_never_expire_on_their_own() {
        let c = Consistency::new(0.8);
        assert_eq!(c.freshness(1, 0.0, 0.0, 1e12), Freshness::Fresh);
    }

    #[test]
    fn a_version_bump_retires_a_generation_without_touching_keys() {
        let mut c = Consistency::new(0.8);
        assert_eq!(c.version("recommendation"), 1);
        c.register(1, &tags(&["row:product:1"]));
        let v = c.bump_version("recommendation", 0.0);
        assert_eq!(v, 2);
        assert_eq!(c.version("recommendation"), 2);
        // Crucially: nothing was invalidated. The old generation ages out instead.
        assert_eq!(c.stats().keys_invalidated, 0);
        assert!(!c.is_stale(1));
    }

    #[test]
    fn forgetting_a_key_removes_it_from_every_tag() {
        let mut c = Consistency::new(0.8);
        c.register(1, &tags(&["a", "b", "c"]));
        assert_eq!(c.stats().tracked_tags, 3);
        c.forget(1);
        assert_eq!(c.stats().tracked_tags, 0, "empty tag buckets must not linger");
        assert_eq!(c.stats().tracked_keys, 0);
    }

    #[test]
    fn re_registering_a_key_replaces_its_old_dependencies() {
        let mut c = Consistency::new(0.8);
        c.register(1, &tags(&["old"]));
        c.register(1, &tags(&["new"]));
        assert!(c.keys_for_tag("old").is_empty(), "stale dependency survived a rebuild");
        assert_eq!(c.keys_for_tag("new"), vec![1]);
    }

    #[test]
    fn removal_reasons_are_counted_separately() {
        let mut c = Consistency::new(0.8);
        c.register(1, &tags(&["t"]));
        c.invalidate(&tags(&["t"]), InvalidationMode::Hard, "test", 0.0);
        c.note_removal(Removal::Evicted);
        c.note_removal(Removal::Expired);
        c.note_removal(Removal::Expired);
        let s = c.stats();
        assert_eq!(s.keys_invalidated, 1);
        assert_eq!(s.evicted, 1);
        assert_eq!(s.expired, 2);
    }
}
