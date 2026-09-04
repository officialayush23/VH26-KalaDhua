//! Value types shared by the cache, the simulator and the HTTP surface.

use serde::{Deserialize, Serialize};

/// Internal key handle. Keys are interned to a `u64` so the hot path never hashes or
/// compares strings, and so traces stay compact.
pub type KeyId = u64;

/// Milliseconds on the engine clock.
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
        }
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

/// Which tier answered a request. Kept in the request record so the dashboard can draw
/// the journey a request took through the universe.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Layer {
    Cdn,
    L1,
    L2,
    Backend,
}

impl Layer {
    pub fn as_str(self) -> &'static str {
        match self {
            Layer::Cdn => "CDN",
            Layer::L1 => "L1",
            Layer::L2 => "L2",
            Layer::Backend => "Backend",
        }
    }
}

/// Stable small integer for an application name.
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
