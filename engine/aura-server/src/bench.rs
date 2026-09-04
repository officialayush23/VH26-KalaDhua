use ahash::AHashMap;
use aura_core::config::Config;
use aura_core::types::KeyId;
use aura_sim::{Generator, Scenario};
use serde::Serialize;

use aura_core::policies::{self, CachePolicy, Request as PolicyRequest};

use crate::capacity::CapacityController;
use crate::engine::Engine;

#[derive(Debug, Clone, Serialize)]
pub struct BenchRow {
    pub policy: String,
    pub capacity_start_bytes: u64,
    pub capacity_end_bytes: u64,
    pub mean_resident_bytes: u64,
    pub holding_cost_usd: f64,
    pub object_hit_rate: f64,
    pub byte_hit_rate: f64,
    pub p95_latency_ms: f64,
    pub backend_requests: u64,
    /// Reads answered from a value that was past its TTL. A policy with a working refresh
    /// controller should drive this to nearly zero; one without a refresh controller cannot
    /// influence it at all, which is the point of reporting it separately from the hit rate.
    pub stale_hits: u64,
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
            "aura" => Some(run_aura(cfg, &stream, capacity_bytes, seed)),
            other => run_baseline(cfg, &stream, capacity_bytes, other),
        };
        if let Some(row) = row {
            rows.push(row);
        }
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

fn run_aura(cfg: &Config, stream: &[aura_sim::Request], capacity: u64, seed: u64) -> BenchRow {
    let mut c = cfg.clone();
    c.cache.capacity_bytes = capacity;
    // Adaptive capacity is one of the engine's two differentiators and switching it off
    // here was measuring the other one alone. It stays on, and every byte it decides to
    // rent is charged at the same price the baselines pay, integrated over time. The
    // baselines keep a fixed pool because having no way to resize is exactly the gap.
    c.capacity.auto = true;
    c.capacity.min_bytes = capacity / 4;
    c.capacity.max_bytes = capacity * 4;
    let mut controller = CapacityController::new(&c);
    let cfg_owned = c.clone();
    let mut eng = Engine::new(c, seed);
    let mut latencies = Vec::new();
    let mut hit_bytes = 0u64;
    let mut total_bytes = 0u64;
    let mut byte_ms = 0.0f64;
    let mut last_ts = stream.first().map(|r| r.ts_ms).unwrap_or(0.0);
    let mut capacity_end = capacity;

    for (i, r) in stream.iter().enumerate() {
        total_bytes += r.context.size_bytes;
        // Occupancy integrated over time is what memory actually costs. Charging the
        // final size, as this used to, lets a policy hold a huge pool for the whole run
        // and pay for the instant it happened to end on.
        byte_ms += eng.store.used_bytes() as f64 * (r.ts_ms - last_ts).max(0.0);
        last_ts = r.ts_ms;
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
        if i % 2_000 == 0 {
            eng.recompute_workload();
            controller.maybe_apply(&mut eng, &cfg_owned);
            capacity_end = eng.store.capacity_bytes();
        }
    }

    let span_ms = (last_ts - stream.first().map(|r| r.ts_ms).unwrap_or(0.0)).max(1.0);
    let mean_resident = (byte_ms / span_ms) as u64;
    let holding = cfg.pricing.holding_cost_usd(mean_resident as f64, span_ms);

    BenchRow {
        policy: "aura".into(),
        capacity_start_bytes: capacity,
        capacity_end_bytes: capacity_end,
        mean_resident_bytes: mean_resident,
        holding_cost_usd: round4(holding),
        object_hit_rate: round4(eng.store.l2_stats.hit_rate()),
        byte_hit_rate: round4(hit_bytes as f64 / total_bytes.max(1) as f64),
        p95_latency_ms: round2(quantile(&mut latencies, 0.95)),
        backend_requests: eng.store.l2_stats.misses,
        stale_hits: eng.stale_serves,
        total_cost_usd: round4(eng.ledger.backend_usd + eng.ledger.sla_penalty_usd + holding),
        regen_cost_usd: round4(eng.ledger.backend_usd),
        sla_penalty_usd: round4(eng.ledger.sla_penalty_usd),
        decision_overhead_us_p50: round2(eng.overhead_p50()),
        memory_overhead_bytes: (eng.store.len() as u64) * 96,
    }
}

/// Replay the stream through one real baseline implementation.
///
/// The point of going through [`aura_core::policies`] rather than re-deriving each policy
/// with a score function here is that W-TinyLFU, S3-FIFO and SIEVE are *defined* by their
/// structure — an admission filter, three queues, a scanning hand — not by a ranking
/// expression. A benchmark that reduces them to `rank(size, age, freq, cost)` is not
/// beating them, it is beating a caricature of them, and a result obtained that way says
/// nothing. Each policy here owns its own machinery and its own byte accounting, and the
/// metadata it carries to do so is charged back in `memory_overhead_bytes` so a policy
/// cannot buy its hit rate with bookkeeping the comparison never prices.
fn run_baseline(
    cfg: &Config,
    stream: &[aura_sim::Request],
    capacity: u64,
    name: &str,
) -> Option<BenchRow> {
    let mut policy = policies::build(name, capacity)?;

    let (mut hits, mut misses, mut stale) = (0u64, 0u64, 0u64);
    let (mut hit_bytes, mut total_bytes) = (0u64, 0u64);
    let mut cost = 0.0f64;
    let mut penalty = 0.0f64;
    let mut latencies = Vec::with_capacity(stream.len());
    let mut byte_ms = 0.0f64;
    let mut last_ts = stream.first().map(|r| r.ts_ms).unwrap_or(0.0);

    for r in stream {
        total_bytes += r.context.size_bytes;
        // Occupancy integrated over time is what memory actually costs. Charging the final
        // size would let a policy hold a huge pool all run and pay for the instant it
        // happened to end on.
        byte_ms += policy.used_bytes() as f64 * (r.ts_ms - last_ts).max(0.0);
        last_ts = r.ts_ms;

        let req = PolicyRequest {
            ts_ms: r.ts_ms,
            key: r.key_id,
            size_bytes: r.context.size_bytes,
            ttl_ms: r.context.ttl_ms.unwrap_or(0.0),
            regen: r.context.regen,
            application: &r.context.application,
            object_type: &r.context.object_type,
            sla: r.context.sla_class,
        };
        let result = policy.access(&req);
        if result.stale {
            stale += 1;
        }
        if result.hit {
            hits += 1;
            hit_bytes += r.context.size_bytes;
            latencies.push(0.4);
            continue;
        }
        misses += 1;
        cost += cfg.pricing.regen_cost_usd(&r.context.regen);
        latencies.push(r.regen_latency_ms);
        if r.regen_latency_ms > cfg.pricing.slo_p95_ms {
            penalty += cfg.pricing.sla_penalty_usd(
                r.regen_latency_ms - cfg.pricing.slo_p95_ms,
                r.context.sla_class.penalty_weight(),
            );
        }
    }

    let span_ms = (last_ts - stream.first().map(|r| r.ts_ms).unwrap_or(0.0)).max(1.0);
    let mean_resident = (byte_ms / span_ms) as u64;
    let holding = cfg.pricing.holding_cost_usd(mean_resident as f64, span_ms);

    Some(BenchRow {
        policy: policy.name().to_string(),
        capacity_start_bytes: capacity,
        // A baseline has no way to resize itself. That is not an oversight in the
        // comparison, it is the gap the adaptive controller exists to fill.
        capacity_end_bytes: capacity,
        mean_resident_bytes: mean_resident,
        holding_cost_usd: round4(holding),
        object_hit_rate: round4(hits as f64 / (hits + misses).max(1) as f64),
        byte_hit_rate: round4(hit_bytes as f64 / total_bytes.max(1) as f64),
        p95_latency_ms: round2(quantile(&mut latencies, 0.95)),
        backend_requests: misses,
        stale_hits: stale,
        total_cost_usd: round4(cost + penalty + holding),
        regen_cost_usd: round4(cost),
        sla_penalty_usd: round4(penalty),
        decision_overhead_us_p50: 0.2,
        memory_overhead_bytes: policy.memory_overhead_bytes() as u64,
    })
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
