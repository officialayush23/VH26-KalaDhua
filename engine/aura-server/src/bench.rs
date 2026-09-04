use ahash::AHashMap;
use aura_core::config::Config;
use aura_core::types::KeyId;
use aura_sim::{Generator, Scenario};
use serde::Serialize;

use crate::engine::Engine;
use crate::policy::Policy;

#[derive(Debug, Clone, Serialize)]
pub struct BenchRow {
    pub policy: String,
    pub object_hit_rate: f64,
    pub byte_hit_rate: f64,
    pub p95_latency_ms: f64,
    pub backend_requests: u64,
    pub total_cost_usd: f64,
    pub regen_cost_usd: f64,
    pub sla_penalty_usd: f64,
    pub decision_overhead_us_p50: f64,
    pub memory_overhead_bytes: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct BenchmarkReport {
    pub run_id: String,
    pub scenario: String,
    pub requests: u64,
    pub capacity_bytes: u64,
    pub seed: u64,
    pub rows: Vec<BenchRow>,
    pub belady_upper_bound: BeladyBound,
    pub winner: String,
    pub improvement_vs: serde_json::Value,
    pub finished: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct BeladyBound {
    pub object_hit_rate: f64,
    pub total_cost_usd: f64,
}

/// Every policy sees the identical request sequence, generated once from the same seed.
/// A comparison built on separately generated streams proves nothing.
pub fn run(
    cfg: &Config,
    scenario: Scenario,
    policies: &[String],
    capacity_bytes: u64,
    requests: usize,
    seed: u64,
) -> BenchmarkReport {
    let mut gen = Generator::new(scenario, seed);
    let stream = gen.take(requests);

    let mut rows = Vec::new();
    for name in policies {
        let row = match name.as_str() {
            "aura" => run_aura(cfg, &stream, capacity_bytes, seed),
            other => match parse_policy(other) {
                Some(p) => run_classical(cfg, &stream, capacity_bytes, p),
                None => continue,
            },
        };
        rows.push(row);
    }

    let belady = run_belady(cfg, &stream, capacity_bytes);

    let winner = rows
        .iter()
        .min_by(|a, b| {
            a.total_cost_usd
                .partial_cmp(&b.total_cost_usd)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|r| r.policy.clone())
        .unwrap_or_default();

    let aura_cost = rows
        .iter()
        .find(|r| r.policy == "aura")
        .map(|r| r.total_cost_usd)
        .unwrap_or(0.0);
    let mut improvement = serde_json::Map::new();
    for r in rows.iter().filter(|r| r.policy != "aura") {
        let gain = if r.total_cost_usd > 0.0 {
            (r.total_cost_usd - aura_cost) / r.total_cost_usd
        } else {
            0.0
        };
        improvement.insert(r.policy.clone(), serde_json::json!(round4(gain)));
    }

    BenchmarkReport {
        run_id: uuid::Uuid::new_v4().to_string(),
        scenario: scenario.id().to_string(),
        requests: requests as u64,
        capacity_bytes,
        seed,
        rows,
        belady_upper_bound: belady,
        winner,
        improvement_vs: serde_json::Value::Object(improvement),
        finished: true,
    }
}

fn parse_policy(s: &str) -> Option<Policy> {
    Policy::ALL.iter().copied().find(|p| p.as_str() == s)
}

fn run_aura(cfg: &Config, stream: &[aura_sim::Request], capacity: u64, seed: u64) -> BenchRow {
    let mut c = cfg.clone();
    c.cache.capacity_bytes = capacity;
    c.capacity.auto = false;
    let mut eng = Engine::new(c, seed);
    let mut latencies = Vec::new();
    let mut hit_bytes = 0u64;
    let mut total_bytes = 0u64;

    for (i, r) in stream.iter().enumerate() {
        total_bytes += r.context.size_bytes;
        match eng.get(r.key_id, &r.context.application, r.ts_ms) {
            Some(_) => {
                hit_bytes += r.context.size_bytes;
                latencies.push(0.4);
            }
            None => {
                let mut measured = r.context.regen;
                measured.latency_ms = r.regen_latency_ms;
                latencies.push(r.regen_latency_ms);
                eng.put(
                    r.key_id,
                    serde_json::Value::Null,
                    &r.context,
                    measured,
                    r.ts_ms,
                );
            }
        }
        if i % 10_000 == 0 {
            eng.recompute_workload();
        }
    }

    BenchRow {
        policy: "aura".into(),
        object_hit_rate: round4(eng.store.l2_stats.hit_rate()),
        byte_hit_rate: round4(hit_bytes as f64 / total_bytes.max(1) as f64),
        p95_latency_ms: round2(quantile(&mut latencies, 0.95)),
        backend_requests: eng.store.l2_stats.misses,
        total_cost_usd: round4(eng.ledger.total()),
        regen_cost_usd: round4(eng.ledger.backend_usd),
        sla_penalty_usd: round4(eng.ledger.sla_penalty_usd),
        decision_overhead_us_p50: round2(eng.overhead_p50()),
        memory_overhead_bytes: (eng.store.len() as u64) * 96,
    }
}

fn run_classical(cfg: &Config, stream: &[aura_sim::Request], capacity: u64, policy: Policy) -> BenchRow {
    let mut resident: AHashMap<KeyId, (u64, f64, f64, f64)> = AHashMap::new();
    let mut used = 0u64;
    let (mut hits, mut misses) = (0u64, 0u64);
    let (mut hit_bytes, mut total_bytes) = (0u64, 0u64);
    let mut cost = 0.0f64;
    let mut penalty = 0.0f64;
    let mut latencies = Vec::new();

    for r in stream {
        let size = r.context.size_bytes;
        total_bytes += size;
        let c = cfg.pricing.regen_cost_usd(&r.context.regen);
        if let Some(slot) = resident.get_mut(&r.key_id) {
            hits += 1;
            hit_bytes += size;
            slot.1 = r.ts_ms;
            slot.3 += 1.0;
            latencies.push(0.4);
            continue;
        }
        misses += 1;
        cost += c;
        latencies.push(r.regen_latency_ms);
        if r.regen_latency_ms > cfg.pricing.slo_p95_ms {
            penalty += cfg.pricing.sla_penalty_usd(
                r.regen_latency_ms - cfg.pricing.slo_p95_ms,
                r.context.sla_class.penalty_weight(),
            );
        }
        if size > capacity {
            continue;
        }
        while used + size > capacity {
            let victim = resident
                .iter()
                .map(|(k, v)| (*k, rank(policy, v, r.ts_ms)))
                .fold(None::<(KeyId, f64)>, |acc, cur| match acc {
                    Some(a) if a.1 <= cur.1 => Some(a),
                    _ => Some(cur),
                });
            match victim {
                Some((k, _)) => {
                    if let Some(v) = resident.remove(&k) {
                        used = used.saturating_sub(v.0);
                    }
                }
                None => break,
            }
        }
        resident.insert(r.key_id, (size, r.ts_ms, c, 1.0));
        used += size;
    }

    let holding = cfg.pricing.holding_cost_usd(used as f64, 60_000.0) * (stream.len() as f64 / 5_000.0);

    BenchRow {
        policy: policy.as_str().into(),
        object_hit_rate: round4(hits as f64 / (hits + misses).max(1) as f64),
        byte_hit_rate: round4(hit_bytes as f64 / total_bytes.max(1) as f64),
        p95_latency_ms: round2(quantile(&mut latencies, 0.95)),
        backend_requests: misses,
        total_cost_usd: round4(cost + penalty + holding),
        regen_cost_usd: round4(cost),
        sla_penalty_usd: round4(penalty),
        decision_overhead_us_p50: 0.2,
        memory_overhead_bytes: (resident.len() as u64) * 48,
    }
}

/// Belady needs the future, so it only exists offline. It is reported as the ceiling the
/// online policies are being measured against, not as a competitor.
fn run_belady(cfg: &Config, stream: &[aura_sim::Request], capacity: u64) -> BeladyBound {
    let mut next_use: AHashMap<KeyId, Vec<usize>> = AHashMap::new();
    for (i, r) in stream.iter().enumerate() {
        next_use.entry(r.key_id).or_default().push(i);
    }
    let mut cursor: AHashMap<KeyId, usize> = AHashMap::new();
    let mut resident: AHashMap<KeyId, u64> = AHashMap::new();
    let mut used = 0u64;
    let (mut hits, mut misses) = (0u64, 0u64);
    let mut cost = 0.0f64;

    for (i, r) in stream.iter().enumerate() {
        let c = cursor.entry(r.key_id).or_insert(0);
        *c += 1;
        let size = r.context.size_bytes;
        if resident.contains_key(&r.key_id) {
            hits += 1;
            continue;
        }
        misses += 1;
        cost += cfg.pricing.regen_cost_usd(&r.context.regen);
        if size > capacity {
            continue;
        }
        while used + size > capacity {
            let victim = resident
                .keys()
                .map(|k| {
                    let idx = cursor.get(k).copied().unwrap_or(0);
                    let nxt = next_use
                        .get(k)
                        .and_then(|v| v.get(idx))
                        .copied()
                        .unwrap_or(usize::MAX);
                    (*k, nxt)
                })
                .fold(None::<(KeyId, usize)>, |acc, cur| match acc {
                    Some(a) if a.1 >= cur.1 => Some(a),
                    _ => Some(cur),
                });
            match victim {
                Some((k, _)) => {
                    if let Some(sz) = resident.remove(&k) {
                        used = used.saturating_sub(sz);
                    }
                }
                None => break,
            }
        }
        resident.insert(r.key_id, size);
        used += size;
        let _ = i;
    }

    BeladyBound {
        object_hit_rate: round4(hits as f64 / (hits + misses).max(1) as f64),
        total_cost_usd: round4(cost),
    }
}

fn rank(policy: Policy, v: &(u64, f64, f64, f64), now_ms: f64) -> f64 {
    let (size, last_ms, cost, freq) = *v;
    match policy {
        Policy::Lru => last_ms,
        Policy::Lfu => freq,
        Policy::Gdsf => (freq * cost.max(1e-12)) / (size as f64).max(1.0) * 1e9,
        Policy::TinyLfu => freq * 0.8 + last_ms / 1e6,
        Policy::CostAware => cost.max(1e-12) / (size as f64).max(1.0) * 1e9,
        Policy::TrendAware => freq / (1.0 + (now_ms - last_ms) / 1000.0),
    }
}

fn quantile(v: &mut [f64], q: f64) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    v[(((v.len() - 1) as f64) * q).round() as usize]
}

fn round4(v: f64) -> f64 {
    (v * 10_000.0).round() / 10_000.0
}

fn round2(v: f64) -> f64 {
    (v * 100.0).round() / 100.0
}
