use std::path::Path;

use aura_core::features::{Features, N_FEATURES};
use serde::{Deserialize, Serialize};

/// A model bundle as produced by `training/aura_train/export.py`. Loading it needs no ML
/// runtime: the tree walker below is the whole inference path.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelBundle {
    pub schema_version: u32,
    pub name: String,
    pub kind: String,
    pub horizon_ms: f64,
    pub version: String,
    #[serde(default)]
    pub git_sha: String,
    pub feature_names: Vec<String>,
    #[serde(default)]
    pub normalization: Option<Normalization>,
    #[serde(default)]
    pub sigmoid_output: bool,
    #[serde(default)]
    pub trees: Vec<Tree>,
    #[serde(default)]
    pub linear_weights: Option<LinearWeights>,
    #[serde(default)]
    pub metrics: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Normalization {
    pub mean: Vec<f64>,
    pub scale: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearWeights {
    pub bias: f64,
    pub weights: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tree {
    pub split_feature: Vec<i32>,
    pub threshold: Vec<f64>,
    pub left: Vec<i32>,
    pub right: Vec<i32>,
    pub leaf_value: Vec<f64>,
    #[serde(default)]
    pub decision_type: Vec<i32>,
}

impl Tree {
    fn eval(&self, x: &[f64]) -> f64 {
        if self.split_feature.is_empty() {
            return self.leaf_value.first().copied().unwrap_or(0.0);
        }
        let mut node: i32 = 0;
        for _ in 0..64 {
            let i = node as usize;
            if i >= self.split_feature.len() {
                return 0.0;
            }
            let f = self.split_feature[i];
            let v = if f >= 0 && (f as usize) < x.len() {
                x[f as usize]
            } else {
                f64::NAN
            };
            // LightGBM's dump sends missing values left, and so does this.
            let go_left = !v.is_finite() || v <= self.threshold[i];
            let next = if go_left { self.left[i] } else { self.right[i] };
            if next < 0 {
                let leaf = (-next - 1) as usize;
                return self.leaf_value.get(leaf).copied().unwrap_or(0.0);
            }
            node = next;
        }
        0.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PredictorKind {
    Heuristic,
    Linear,
    Gbdt,
}

impl PredictorKind {
    pub fn as_str(self) -> &'static str {
        match self {
            PredictorKind::Heuristic => "heuristic",
            PredictorKind::Linear => "linear",
            PredictorKind::Gbdt => "gbdt",
        }
    }
}

/// Three horizons, because "will this be reused" is meaningless without a time bound.
/// Ten seconds decides admission, sixty decides eviction, ten minutes decides refresh.
#[derive(Debug)]
pub struct Predictor {
    kind: PredictorKind,
    h10: Option<ModelBundle>,
    h60: Option<ModelBundle>,
    h600: Option<ModelBundle>,
    online: OnlineLogistic,
    pub calls: u64,
    pub source: String,
}

impl Predictor {
    pub fn heuristic(online_lr: f64) -> Self {
        Self {
            kind: PredictorKind::Heuristic,
            h10: None,
            h60: None,
            h600: None,
            online: OnlineLogistic::new(online_lr),
            calls: 0,
            source: "builtin".to_string(),
        }
    }

    pub fn load_dir(dir: &Path, online_lr: f64) -> anyhow::Result<Self> {
        let mut p = Self::heuristic(online_lr);
        p.h10 = read_bundle(&dir.join("reuse_gbdt_h10s.json"));
        p.h60 = read_bundle(&dir.join("reuse_gbdt_h60s.json"));
        p.h600 = read_bundle(&dir.join("reuse_gbdt_h600s.json"));
        if p.h60.is_none() {
            p.h60 = read_bundle(&dir.join("reuse_linear_h60s.json"));
        }
        if p.h60.is_some() {
            p.kind = if p.h60.as_ref().map(|b| b.trees.is_empty()).unwrap_or(true) {
                PredictorKind::Linear
            } else {
                PredictorKind::Gbdt
            };
            p.source = dir.display().to_string();
        }
        Ok(p)
    }

    pub fn load_bundle(&mut self, bundle: ModelBundle, source: &str) {
        self.kind = if bundle.trees.is_empty() {
            PredictorKind::Linear
        } else {
            PredictorKind::Gbdt
        };
        self.source = source.to_string();
        let h = bundle.horizon_ms;
        if h <= 20_000.0 {
            self.h10 = Some(bundle);
        } else if h <= 120_000.0 {
            self.h60 = Some(bundle);
        } else {
            self.h600 = Some(bundle);
        }
    }

    pub fn kind(&self) -> PredictorKind {
        self.kind
    }

    pub fn loaded_horizons(&self) -> Vec<&'static str> {
        let mut v = Vec::new();
        if self.h10.is_some() {
            v.push("h10s");
        }
        if self.h60.is_some() {
            v.push("h60s");
        }
        if self.h600.is_some() {
            v.push("h600s");
        }
        v
    }

    /// Reuse probability at the three horizons. When no bundle is loaded the online
    /// logistic model carries the cold start, so the engine is never blind.
    pub fn reuse(&mut self, f: &Features) -> [f64; 3] {
        self.calls += 1;
        let fallback = self.online.predict(f);
        [
            self.h10.as_ref().map(|b| score(b, f)).unwrap_or(fallback * 0.55),
            self.h60.as_ref().map(|b| score(b, f)).unwrap_or(fallback),
            self.h600
                .as_ref()
                .map(|b| score(b, f))
                .unwrap_or(1.0 - (1.0 - fallback).powf(0.55)),
        ]
    }

    pub fn observe(&mut self, f: &Features, reused: bool) {
        self.online.update(f, if reused { 1.0 } else { 0.0 });
    }

    pub fn confidence(&self) -> f64 {
        match self.kind {
            PredictorKind::Gbdt => 0.86,
            PredictorKind::Linear => 0.64,
            PredictorKind::Heuristic => self.online.confidence(),
        }
    }

    pub fn contributions(&self, f: &Features) -> Vec<(String, f64)> {
        let names = aura_core::features::FEATURE_NAMES;
        let w = match self.h60.as_ref().and_then(|b| b.linear_weights.as_ref()) {
            Some(lw) => lw.weights.clone(),
            None => self.online.weights.to_vec(),
        };
        let mut out: Vec<(String, f64)> = names
            .iter()
            .enumerate()
            .map(|(i, n)| {
                let c = w.get(i).copied().unwrap_or(0.0) * f[i];
                (n.to_string(), c)
            })
            .collect();
        out.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap_or(std::cmp::Ordering::Equal));
        out.truncate(6);
        out
    }
}

fn read_bundle(path: &Path) -> Option<ModelBundle> {
    let text = std::fs::read_to_string(path).ok()?;
    serde_json::from_str(&text).ok()
}

fn score(b: &ModelBundle, f: &Features) -> f64 {
    let x = normalize(b, f);
    let raw = if let Some(lw) = &b.linear_weights {
        let mut s = lw.bias;
        for (i, w) in lw.weights.iter().enumerate() {
            s += w * x.get(i).copied().unwrap_or(0.0);
        }
        s
    } else {
        b.trees.iter().map(|t| t.eval(&x)).sum()
    };
    if b.sigmoid_output || b.linear_weights.is_some() {
        sigmoid(raw)
    } else {
        raw.clamp(0.0, 1.0)
    }
}

fn normalize(b: &ModelBundle, f: &Features) -> Vec<f64> {
    match &b.normalization {
        Some(n) => f
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let m = n.mean.get(i).copied().unwrap_or(0.0);
                let s = n.scale.get(i).copied().unwrap_or(1.0);
                if s.abs() < 1e-12 {
                    0.0
                } else {
                    (v - m) / s
                }
            })
            .collect(),
        None => f.to_vec(),
    }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x.clamp(-30.0, 30.0)).exp())
}

/// Logistic regression trained online from realised outcomes. It is what runs before any
/// offline bundle exists, and it keeps running afterwards so the engine has a live signal
/// to compare a stale bundle against.
#[derive(Debug)]
pub struct OnlineLogistic {
    pub weights: [f64; N_FEATURES],
    pub bias: f64,
    lr: f64,
    seen: u64,
    running_loss: f64,
}

impl OnlineLogistic {
    pub fn new(lr: f64) -> Self {
        let mut weights = [0.0f64; N_FEATURES];
        // A cold model that predicts nothing is worse than a cold model that knows
        // frequency and recency matter. These priors are replaced within a few thousand
        // observations.
        weights[aura_core::features::idx::FREQ_1M] = 0.55;
        weights[aura_core::features::idx::FREQ_5M] = 0.30;
        weights[aura_core::features::idx::EWMA_FAST] = 0.40;
        weights[aura_core::features::idx::TREND] = 0.25;
        weights[aura_core::features::idx::LOG_AGE_MS] = -0.18;
        weights[aura_core::features::idx::LOG_INTER_ARRIVAL_MS] = -0.22;
        Self {
            weights,
            bias: -0.4,
            lr,
            seen: 0,
            running_loss: 0.7,
        }
    }

    pub fn predict(&self, f: &Features) -> f64 {
        let mut s = self.bias;
        for i in 0..N_FEATURES {
            s += self.weights[i] * squash(f[i]);
        }
        sigmoid(s)
    }

    pub fn update(&mut self, f: &Features, target: f64) {
        let p = self.predict(f);
        let err = p - target;
        for i in 0..N_FEATURES {
            self.weights[i] -= self.lr * err * squash(f[i]);
        }
        self.bias -= self.lr * err;
        self.seen += 1;
        let loss = -(target * p.max(1e-9).ln() + (1.0 - target) * (1.0 - p).max(1e-9).ln());
        self.running_loss = self.running_loss * 0.999 + loss * 0.001;
    }

    pub fn confidence(&self) -> f64 {
        let warmup = (self.seen as f64 / 5_000.0).clamp(0.0, 1.0);
        let quality = (1.0 - self.running_loss / 0.7).clamp(0.0, 1.0);
        (0.25 + 0.6 * warmup * quality).clamp(0.0, 0.95)
    }
}

/// Raw features span many orders of magnitude. Squashing keeps a single learning rate
/// usable across all of them without maintaining a separate scaler.
fn squash(v: f64) -> f64 {
    (v / (1.0 + v.abs())).clamp(-1.0, 1.0)
}
