//! Per-application policy profiles: what each application wants the cache to optimise.
//!
//! One cache serves a ranking service whose objects cost 300 ms of GPU and a dashboard
//! whose objects cost a 400 ms aggregate, and the correct treatment of the two is not the
//! same. The engine's global config sets one admission bar, one horizon mix and one view of
//! how much a slow response costs, which means whichever application the numbers were tuned
//! for wins and the other is merely tolerated.
//!
//! A profile is the operator saying, per application, what "valuable" means here. Nothing in
//! it touches the model: the 24 features go into the trained head exactly as they were
//! trained, and the profile changes only the arithmetic the engine composes *around* the
//! prediction — how far into the future to care, how much a slow rebuild is worth avoiding,
//! how much better than the victim an arrival must be, and how much of the pool one
//! application may hold. Scaling the model's own inputs would move the input distribution
//! away from the one the model was fitted on and quietly invalidate its calibration, which
//! is why that is not offered.
//!
//! Every knob carries its own documentation ([`knobs`]), served from the same place the
//! engine reads it, so the dashboard cannot describe a knob the engine does not have.

use ahash::AHashMap;
use aura_core::config::Config;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

/// What an application is asking the cache to be good at. A preset, not a mode: it fills
/// the knobs below with a sensible starting point, and each one can then be moved.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Objective {
    /// Spend the least money in total. The default, and what the benchmark measures.
    Cost,
    /// Keep the slow tail off users, even when that means holding objects a pure cost
    /// argument would drop.
    Latency,
    /// Protect the thing behind the cache. Favours never having to rebuild at all.
    Origin,
    /// Nothing preset; every knob is whatever the operator set.
    Custom,
}

impl Objective {
    pub fn as_str(self) -> &'static str {
        match self {
            Objective::Cost => "cost",
            Objective::Latency => "latency",
            Objective::Origin => "origin",
            Objective::Custom => "custom",
        }
    }

    fn parse(s: &str) -> Option<Self> {
        match s {
            "cost" => Some(Objective::Cost),
            "latency" => Some(Objective::Latency),
            "origin" => Some(Objective::Origin),
            "custom" => Some(Objective::Custom),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct AppProfile {
    pub objective: Objective,
    /// How the three reuse horizons (10 s, 60 s, 600 s) are weighted against each other.
    /// Normalised on write, so an operator can type any three numbers.
    pub horizon_weights: [f64; 3],
    /// How much extra an object is worth when its rebuild cost is erratic. Zero treats a
    /// predictable 50 ms rebuild and a 50 ms mean with a 2 s tail identically.
    pub tail_risk_lambda: f64,
    /// How much better than the object it would displace an arrival must be. 1.0 admits
    /// anything at least as good; above 1.0 protects what is already resident.
    pub admission_margin: f64,
    /// Multiplier on the SLA penalty this application's slow rebuilds are charged. It is
    /// how loudly this application says "being slow costs me more than it costs you".
    pub sla_weight: f64,
    /// Where in an object's life the refresh-ahead window starts, as a fraction of TTL.
    pub soft_ttl_fraction: f64,
    /// The largest share of the pool this application may occupy before its arrivals have
    /// to clear a higher bar. 1.0 is uncapped.
    pub max_pool_share: f64,
}

impl AppProfile {
    /// The engine's global settings, expressed as a profile. This is what every application
    /// gets until someone changes it, so an untouched deployment behaves exactly as before.
    pub fn from_config(cfg: &Config) -> Self {
        Self {
            objective: Objective::Cost,
            horizon_weights: cfg.engine.horizon_weights,
            tail_risk_lambda: cfg.engine.tail_risk_lambda,
            admission_margin: cfg.engine.admission_margin,
            sla_weight: 1.0,
            soft_ttl_fraction: cfg.engine.soft_ttl_fraction,
            max_pool_share: 1.0,
        }
    }

    /// Apply a partial update. Unknown fields are ignored rather than rejected, so a newer
    /// dashboard against an older engine degrades instead of failing, and every value is
    /// clamped to a range the engine can actually run in.
    pub fn patch(&mut self, body: &Value) {
        if let Some(v) = body.get("objective").and_then(|v| v.as_str()) {
            if let Some(o) = Objective::parse(v) {
                self.objective = o;
            }
        }
        if let Some(arr) = body.get("horizon_weights").and_then(|v| v.as_array()) {
            if arr.len() == 3 {
                let mut w = [0.0; 3];
                for (i, v) in arr.iter().enumerate() {
                    w[i] = v.as_f64().unwrap_or(0.0).clamp(0.0, 1.0);
                }
                let sum: f64 = w.iter().sum();
                if sum > 1e-9 {
                    for v in w.iter_mut() {
                        *v /= sum;
                    }
                    self.horizon_weights = w;
                    self.objective = Objective::Custom;
                }
            }
        }
        let num = |name: &str, lo: f64, hi: f64, slot: &mut f64| {
            if let Some(v) = body.get(name).and_then(|v| v.as_f64()) {
                *slot = v.clamp(lo, hi);
                true
            } else {
                false
            }
        };
        let touched = [
            num("tail_risk_lambda", 0.0, 2.0, &mut self.tail_risk_lambda),
            num("admission_margin", 0.8, 3.0, &mut self.admission_margin),
            num("sla_weight", 0.0, 10.0, &mut self.sla_weight),
            num("soft_ttl_fraction", 0.2, 0.99, &mut self.soft_ttl_fraction),
            num("max_pool_share", 0.05, 1.0, &mut self.max_pool_share),
        ];
        if touched.iter().any(|t| *t) && body.get("objective").is_none() {
            self.objective = Objective::Custom;
        }
    }
}

/// Profiles by application, with the global defaults behind them.
#[derive(Debug, Clone)]
pub struct ProfileStore {
    defaults: AppProfile,
    map: AHashMap<String, AppProfile>,
}

impl ProfileStore {
    pub fn new(cfg: &Config) -> Self {
        Self { defaults: AppProfile::from_config(cfg), map: AHashMap::new() }
    }

    /// The profile in force for an application. Falls back to the defaults, so an
    /// application that has never been configured is never a special case in the hot path.
    pub fn get(&self, application: &str) -> &AppProfile {
        self.map.get(application).unwrap_or(&self.defaults)
    }

    pub fn defaults(&self) -> &AppProfile {
        &self.defaults
    }

    /// Update one application's profile, creating it from the defaults if it is the first
    /// change. Returns the profile as it now stands.
    pub fn patch(&mut self, application: &str, body: &Value) -> AppProfile {
        let base = *self.map.get(application).unwrap_or(&self.defaults);
        let mut next = match body.get("objective").and_then(|v| v.as_str()).and_then(Objective::parse) {
            // Choosing an objective starts from that preset rather than from whatever the
            // knobs happened to be, which is what makes the presets usable as a reset.
            Some(o) if o != Objective::Custom => AppProfile { objective: o, ..preset_from(&base, o) },
            _ => base,
        };
        next.patch(body);
        self.map.insert(application.to_string(), next);
        next
    }

    /// Forget an application's profile; it returns to the defaults.
    pub fn reset(&mut self, application: &str) -> bool {
        self.map.remove(application).is_some()
    }

    pub fn is_customised(&self, application: &str) -> bool {
        self.map.contains_key(application)
    }

    pub fn customised(&self) -> Vec<(&String, &AppProfile)> {
        self.map.iter().collect()
    }
}

/// Presets are computed from the defaults the store was built with, not from a fresh config
/// read, so a profile never silently picks up a config change mid-run.
fn preset_from(base: &AppProfile, objective: Objective) -> AppProfile {
    let mut p = *base;
    p.objective = objective;
    match objective {
        Objective::Latency => {
            p.horizon_weights = [0.6, 0.3, 0.1];
            p.tail_risk_lambda = 0.6;
            p.sla_weight = 2.5;
            p.soft_ttl_fraction = 0.65;
        }
        Objective::Origin => {
            p.horizon_weights = [0.35, 0.35, 0.3];
            p.admission_margin = 1.0;
            p.soft_ttl_fraction = 0.75;
        }
        _ => {}
    }
    p
}

/// What each knob does, in the words the dashboard shows.
///
/// This lives in the engine rather than in the UI on purpose. A control whose label is
/// written somewhere else drifts from the behaviour it controls, and a slider that lies
/// about what it does is worse than no slider.
pub fn knobs() -> Value {
    json!({
        "objectives": [
            { "id": "cost", "label": "Lowest spend",
              "effect": "The default. Keeps whatever saves the most money per byte held, across all three reuse horizons." },
            { "id": "latency", "label": "Protect the tail",
              "effect": "Weights the near horizon and erratic rebuilds, and prices a slow response above its machine cost. Holds objects a pure cost argument would drop." },
            { "id": "origin", "label": "Protect the origin",
              "effect": "Admits more readily and holds longer. Fewer rebuilds reach the database or the GPU, at the price of a larger pool." },
            { "id": "custom", "label": "Custom", "effect": "Whatever the knobs below are set to." }
        ],
        "knobs": [
            {
                "id": "horizon_weights", "label": "Reuse horizons", "kind": "weights",
                "parts": ["10 seconds", "1 minute", "10 minutes"],
                "what": "How far ahead the cache should care about an object being wanted again.",
                "raise": "Weighting the near horizon keeps the hot set tight and reacts fast to a spike.",
                "lower": "Weighting the far horizon keeps slow-burning objects that are expensive to rebuild but rarely touched."
            },
            {
                "id": "tail_risk_lambda", "label": "Rebuild risk", "kind": "number",
                "min": 0.0, "max": 2.0, "step": 0.05,
                "what": "Extra value given to objects whose rebuild cost is unpredictable.",
                "raise": "An object that usually takes 50 ms but sometimes 2 s is treated as worth more than its average. Fewer nasty surprises, slightly lower average hit rate.",
                "lower": "Objects are valued on their mean rebuild cost alone. Cheaper on paper, spikier in practice."
            },
            {
                "id": "admission_margin", "label": "Admission bar", "kind": "number",
                "min": 0.8, "max": 3.0, "step": 0.05,
                "what": "How much better than the object it would displace an arrival must be.",
                "raise": "The resident set is protected and churn falls. Push it too high and the cache stops learning what is new.",
                "lower": "New objects get in easily. Good while the cache is cold, expensive once it is full."
            },
            {
                "id": "sla_weight", "label": "Cost of being slow", "kind": "number",
                "min": 0.0, "max": 10.0, "step": 0.1,
                "what": "Multiplier on the penalty charged when this application's rebuild is slower than the SLO.",
                "raise": "Latency starts outweighing money. This application's slow objects get kept even when they are cheap to rebuild.",
                "lower": "Only machine cost counts. A slow rebuild is no worse than a fast one of the same price."
            },
            {
                "id": "soft_ttl_fraction", "label": "Refresh-ahead point", "kind": "number",
                "min": 0.2, "max": 0.99, "step": 0.01,
                "what": "How far into an object's life the cache starts rebuilding it behind the reader.",
                "raise": "Rebuilds happen later. Less background work, more chance a reader meets an expired object.",
                "lower": "Rebuilds start earlier. Fresher objects and smoother latency, paid for in extra origin calls."
            },
            {
                "id": "max_pool_share", "label": "Pool share cap", "kind": "number",
                "min": 0.05, "max": 1.0, "step": 0.05,
                "what": "The most of the shared pool this application may hold before its arrivals face a higher bar.",
                "raise": "This application can take as much of the cache as its objects are worth.",
                "lower": "Its footprint is capped, which protects the other applications from one noisy neighbour at the cost of its own hit rate."
            }
        ]
    })
}
