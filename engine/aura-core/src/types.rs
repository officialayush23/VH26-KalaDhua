//! Value types shared by the cache, the simulator and the HTTP surface.
//!
//! The cache never learns what an application *means*. It only sees the shape of an
//! object: how big it is, how expensive it was to produce, how long it stays valid and
//! how much its absence hurts. Everything in this module exists to keep that boundary
//! honest.

use serde::{Deserialize, Serialize};

/// Internal key handle. Keys are interned to a `u64` so the hot path never hashes or
/// compares strings, and so traces stay compact.
pub type KeyId = u64;

/// Milliseconds on the engine clock. The live server feeds it wall time; the simulator
/// feeds it virtual time. Nothing below this line can tell the difference, which is what
/// lets one implementation serve both.
pub type Millis = f64;

/// Service level expectation attached to an object by the application that produced it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum SlaClass {
    Critical,
    High,
    #[default]
    Normal,
    Low,
}

impl SlaClass {
    /// Multiplier applied to the latency penalty term of the economic model.
    pub fn penalty_weight(self) -> f64 {
        match self {
            SlaClass::Critical => 6.0,
            SlaClass::High => 3.0,
            SlaClass::Normal => 1.0,
            SlaClass::Low => 0.25,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            SlaClass::Critical => "critical",
            SlaClass::High => "high",
            SlaClass::Normal => "normal",
            SlaClass::Low => "low",
        }
    }
}

/// What it costs to produce one object once, in physical units.
///
/// Deliberately *not* a single number. Two objects that both take 300 ms to regenerate can
/// have completely different economics: one burns a GPU, the other holds a database
/// connection open. Collapsing that into "latency" is exactly the information loss that
/// makes conventional policies mis-rank objects.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub struct CostVector {
    #[serde(default)]
    pub cpu_ms: f64,
    #[serde(default)]
    pub gpu_ms: f64,
    #[serde(default)]
    pub db_ms: f64,
    #[serde(default)]
    pub network_bytes: f64,
    #[serde(default)]
    pub api_calls: f64,
    #[serde(default)]
    pub api_cost_usd: f64,
    /// Wall-clock latency the user actually waited for.
    #[serde(default)]
    pub latency_ms: f64,
}

impl CostVector {
    pub fn is_empty(&self) -> bool {
        self.cpu_ms == 0.0
            && self.gpu_ms == 0.0
            && self.db_ms == 0.0
            && self.network_bytes == 0.0
            && self.api_cost_usd == 0.0
    }

    pub fn blend(&self, other: &CostVector, alpha: f64) -> CostVector {
        let b = 1.0 - alpha;
        CostVector {
            cpu_ms: self.cpu_ms * b + other.cpu_ms * alpha,
            gpu_ms: self.gpu_ms * b + other.gpu_ms * alpha,
            db_ms: self.db_ms * b + other.db_ms * alpha,
            network_bytes: self.network_bytes * b + other.network_bytes * alpha,
            api_calls: self.api_calls * b + other.api_calls * alpha,
            api_cost_usd: self.api_cost_usd * b + other.api_cost_usd * alpha,
            latency_ms: self.latency_ms * b + other.latency_ms * alpha,
        }
    }
}

/// Everything an application tells the cache about one object.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ObjectContext {
    #[serde(default = "default_application")]
    pub application: String,
    #[serde(default = "default_object_type")]
    pub object_type: String,
    pub size_bytes: u64,
    #[serde(default)]
    pub ttl_ms: Option<f64>,
    #[serde(default)]
    pub sla_class: SlaClass,
    #[serde(default)]
    pub regen: CostVector,
    /// What this object was derived from, as free-form tags the application chooses —
    /// `row:product:1292`, `table:orders`, `tenant:acme`.
    ///
    /// The cache never interprets a tag. It only records the edge, so that when the
    /// application says "this row changed" it can name every object downstream of it
    /// without knowing what any of them are. Without this the only way to be correct after
    /// a write is a short TTL, which pays for staleness on every object all the time
    /// instead of paying for accuracy once, when something actually changes.
    #[serde(default)]
    pub depends_on: Vec<String>,
    /// The generation this object belongs to. Bumping a namespace retires everything in it
    /// without deleting anything. `None` means the object is not versioned.
    #[serde(default)]
    pub namespace: Option<String>,
}

fn default_application() -> String {
    "default".to_string()
}
fn default_object_type() -> String {
    "object".to_string()
}

impl ObjectContext {
    pub fn new(application: &str, object_type: &str, size_bytes: u64) -> Self {
        Self {
            application: application.to_string(),
            object_type: object_type.to_string(),
            size_bytes,
            ttl_ms: None,
            sla_class: SlaClass::Normal,
            regen: CostVector::default(),
            depends_on: Vec::new(),
            namespace: None,
        }
    }

    pub fn with_tags(mut self, tags: &[&str]) -> Self {
        self.depends_on = tags.iter().map(|s| s.to_string()).collect();
        self
    }

    pub fn with_namespace(mut self, namespace: &str) -> Self {
        self.namespace = Some(namespace.to_string());
        self
    }

    pub fn with_regen(mut self, regen: CostVector) -> Self {
        self.regen = regen;
        self
    }

    pub fn with_ttl(mut self, ttl_ms: f64) -> Self {
        self.ttl_ms = Some(ttl_ms);
        self
    }

    pub fn with_sla(mut self, sla: SlaClass) -> Self {
        self.sla_class = sla;
        self
    }
}

/// The four questions the engine answers, and nothing else.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Action {
    Admit,
    Reject,
    Keep,
    Evict,
    Refresh,
}

impl Action {
    pub fn as_str(self) -> &'static str {
        match self {
            Action::Admit => "Admit",
            Action::Reject => "Reject",
            Action::Keep => "Keep",
            Action::Evict => "Evict",
            Action::Refresh => "Refresh",
        }
    }
}

/// A decision plus the machine-readable reason it was taken. The reason code is what the
/// dashboard turns into a sentence; the engine never produces a decision without one.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Decision {
    pub action: Action,
    pub reason_code: &'static str,
}

impl Decision {
    pub fn new(action: Action, reason_code: &'static str) -> Self {
        Self { action, reason_code }
    }
}

/// Outcome of a single cache access, as seen by the caller.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Outcome {
    Hit,
    Miss,
    /// The object was present but past its TTL: a miss that also proves the refresh
    /// controller had an opportunity it did not take.
    StaleMiss,
}

/// Which tier answered a request.
///
/// There are two, and naming them honestly matters more than the names being pretty.
/// `AdmissionWindow` is a small recency set holding **keys only, no values** -- it exists to
/// spot one-shot keys and it can never answer a request, so it is not a cache tier. `Cache`
/// is the scored pool: the thing this project is about. `Backend` is the application going
/// to its own origin.
///
/// A `Cdn` variant used to sit at the top of this list, with a `CdnConfig` beside it in the
/// configuration, and neither was ever implemented. A tier that appears in the vocabulary,
/// the config file and the dashboard but nowhere in the request path is not a roadmap item,
/// it is a claim the code does not support -- so it is gone rather than pending.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Layer {
    AdmissionWindow,
    Cache,
    Backend,
}

impl Layer {
    pub fn as_str(self) -> &'static str {
        match self {
            Layer::AdmissionWindow => "admission window",
            Layer::Cache => "cache",
            Layer::Backend => "backend",
        }
    }
}

/// Stable small integer for an application name.
///
/// The first three ids are pinned because the trained models carry them as a feature and
/// the training pipeline pins the same values. Anything else hashes into a fixed range so
/// a brand new application still gets a usable, deterministic id without retraining.
pub fn app_id(name: &str) -> u16 {
    match name {
        "recommendation" => 0,
        "analytics" => 1,
        "content" => 2,
        other => {
            let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
            for b in other.as_bytes() {
                hash ^= *b as u64;
                hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
            }
            3 + (hash % 1021) as u16
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pinned_application_ids_match_the_training_pipeline() {
        assert_eq!(app_id("recommendation"), 0);
        assert_eq!(app_id("analytics"), 1);
        assert_eq!(app_id("content"), 2);
        // Pinned in training/tests/golden/feature_vectors.json.
        assert_eq!(app_id("billing"), 192);
    }

    #[test]
    fn unknown_applications_are_deterministic_and_in_range() {
        for name in ["a", "search", "ledger", "very-long-application-name"] {
            let id = app_id(name);
            assert!((3..=1023).contains(&id));
            assert_eq!(id, app_id(name));
        }
    }
}
