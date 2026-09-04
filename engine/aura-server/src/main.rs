#![forbid(unsafe_code)]

mod audit;
mod auth;
mod bench;
mod capacity;
mod consistency;
mod engine;
mod feedback;
mod hostmem;
mod policy;
mod predictor;
mod profiles;
mod store;
mod supabase;

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::{Path, Query, State};
use axum::http::{HeaderValue, StatusCode};
use axum::response::IntoResponse;
use axum::routing::{delete, get, post};
use axum::{Json, Router};
use ahash::AHashMap;
use aura_core::config::Config;
use aura_core::types::{CostVector, KeyId, ObjectContext};
use aura_sim::{Attack, Generator, Scenario};
use clap::Parser;
use parking_lot::Mutex;
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::sync::broadcast;

use crate::bench::BenchmarkReport;
use crate::capacity::CapacityController;
use crate::consistency::InvalidationMode;
use crate::engine::Engine;
use crate::policy::Policy;

/// Resolve what to listen on, in the order a deployed process should: the explicit flag,
/// then the platform's `PORT`, then the local default. A bare port is expanded to all
/// interfaces, because binding to loopback inside a container makes the service
/// unreachable while looking perfectly healthy from the inside.
fn listen_addr(flag: Option<&str>) -> anyhow::Result<SocketAddr> {
    let raw = match flag {
        Some(v) => v.to_string(),
        None => match std::env::var("PORT") {
            Ok(p) if !p.trim().is_empty() => format!("0.0.0.0:{}", p.trim()),
            _ => "0.0.0.0:8080".to_string(),
        },
    };
    if let Ok(port) = raw.parse::<u16>() {
        return Ok(SocketAddr::from(([0, 0, 0, 0], port)));
    }
    raw.parse::<SocketAddr>()
        .map_err(|err| anyhow::anyhow!("cannot parse listen address {raw:?}: {err}"))
}

#[derive(Parser, Debug)]
#[command(name = "aura", version, about = "AURA cache server")]
struct Args {
    /// Listen address. When the flag is absent the platform's `PORT` is used if it set one
    /// (Railway, Render and Fly all do and route to nothing else), otherwise 0.0.0.0:8080.
    /// A bare port number is accepted as well.
    #[arg(long)]
    bind: Option<String>,
    #[arg(long)]
    config: Option<PathBuf>,
    #[arg(long, default_value = "models")]
    models: PathBuf,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    /// Store the object bytes for real instead of accounting for them. Without this the
    /// cache tracks a byte budget but holds no payload, so the reported pool size is a
    /// simulated budget rather than resident memory.
    #[arg(long)]
    real_values: bool,
    /// Calibrate the analytics rebuild cost from real Supabase queries instead of the
    /// generator's synthetic figure. Requires Supabase credentials and the seeded tables.
    #[arg(long)]
    real_backend: bool,
    /// Start the simulator immediately. Without this the server is a plain cache and only
    /// serves what applications put into it.
    #[arg(long)]
    scenario: Option<String>,
    /// Pool size in megabytes, overriding the config file. Convenient because the number
    /// that matters on a laptop is different from the one that matters on a server.
    #[arg(long)]
    pool_mb: Option<u64>,
}

struct Sim {
    generator: Generator,
    running: bool,
    speed: f64,
}

/// The last measurement taken against the real analytics backend. `measured_ms` of zero
/// means nothing has been measured yet and the synthetic figure still applies.
#[derive(Debug, Default, Clone, Copy)]
struct BackendProbe {
    enabled: bool,
    measured_ms: f64,
    p95_ms: f64,
    samples: u64,
    failures: u64,
    bytes: u64,
}

/// Who is currently allowed to rebuild a missing key, and until when.
///
/// A cache miss on a popular key is the moment a cache is most dangerous: a thousand
/// readers all discover the same absence within a few milliseconds and all call the origin,
/// which is how a cache causes the outage it exists to prevent. The engine cannot make the
/// origin call itself — only the application knows how to build the object — so it hands
/// out a lease instead: the first caller is told to rebuild, everyone else is told someone
/// already is and how long to wait. That is single-flight for a cache that lives outside
/// the application, and it is the whole reason this map exists.
#[derive(Debug, Default)]
struct Leases {
    held: AHashMap<KeyId, f64>,
    granted: u64,
    /// Origin calls this prevented. The number the flash-crowd demo is about.
    suppressed: u64,
}

impl Leases {
    /// Returns `true` if this caller should rebuild, plus how long a losing caller should
    /// wait before assuming the winner died.
    fn acquire(&mut self, key: KeyId, now_ms: f64, lease_ms: f64) -> (bool, f64) {
        // Opportunistic sweep, so a cache that is missing constantly cannot grow this map
        // without bound while nothing ever calls back to release a lease.
        if self.held.len() > 8_192 {
            self.held.retain(|_, until| *until > now_ms);
        }
        match self.held.get(&key) {
            Some(until) if *until > now_ms => {
                self.suppressed += 1;
                (false, *until - now_ms)
            }
            _ => {
                self.held.insert(key, now_ms + lease_ms);
                self.granted += 1;
                (true, 0.0)
            }
        }
    }

    fn release(&mut self, key: KeyId) {
        self.held.remove(&key);
    }
}

struct App {
    engine: Mutex<Engine>,
    auth: Mutex<auth::Auth>,
    capacity: Mutex<CapacityController>,
    leases: Mutex<Leases>,
    sim: Mutex<Option<Sim>>,
    cfg: Config,
    bench: Mutex<Option<BenchmarkReport>>,
    real_values: bool,
    probe: Mutex<BackendProbe>,
    tx: broadcast::Sender<String>,
    started: std::time::Instant,
    supabase: Option<supabase::Supabase>,
}

type Shared = Arc<App>;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "aura=info,tower_http=warn".into()),
        )
        .init();

    let args = Args::parse();
    if let Some(path) = supabase::load_dotenv() {
        tracing::info!(env = %path.display(), "loaded environment file");
    }
    let mut cfg = Config::load(args.config.as_deref());

    if let Some(mb) = args.pool_mb {
        // An explicit pool size is a ceiling as well as a starting point. The controller is
        // free to grow within the config's own bounds, but not past a number a person typed
        // on the command line because their machine has that much and no more.
        let bytes = mb.saturating_mul(1024 * 1024).max(8 * 1024 * 1024);
        cfg.cache.capacity_bytes = bytes;
        cfg.capacity.max_bytes = cfg.capacity.max_bytes.min(bytes);
        cfg.capacity.host_budget_bytes = cfg.capacity.host_budget_bytes.min(bytes);
        cfg.capacity.min_bytes = cfg.capacity.min_bytes.min(bytes);
    }

    // With --real-values the pool is resident memory, so a configured size larger than the
    // machine is not a budget, it is a crash: the allocator refuses and a binary built with
    // panic=abort does not get to explain itself. Clamp to what this host can actually
    // spare, and say so, rather than starting and dying under load.
    if args.real_values {
        let (available, total) = hostmem::snapshot();
        if let Some(safe) = hostmem::safe_pool_bytes() {
            if cfg.cache.capacity_bytes > safe {
                tracing::warn!(
                    configured = %aura_server_bytes(cfg.cache.capacity_bytes),
                    allowed = %aura_server_bytes(safe),
                    available = %aura_server_bytes(available),
                    total = %aura_server_bytes(total),
                    "pool clamped to what this machine can hold: --real-values stores payloads \
                     for real, and the measured overhead is about 1.5x the logical pool"
                );
                cfg.cache.capacity_bytes = safe;
            }
            // The controller must not undo the clamp on the next tick.
            if cfg.capacity.max_bytes > safe {
                cfg.capacity.max_bytes = safe;
            }
            if cfg.capacity.host_budget_bytes > safe {
                cfg.capacity.host_budget_bytes = safe;
            }
            if cfg.capacity.min_bytes > cfg.cache.capacity_bytes {
                cfg.capacity.min_bytes = cfg.cache.capacity_bytes;
            }
            if cfg.capacity.step_bytes > cfg.cache.capacity_bytes / 2 {
                cfg.capacity.step_bytes = (cfg.cache.capacity_bytes / 4).max(8 * 1024 * 1024);
            }
        } else {
            tracing::warn!(
                "this platform will not report free memory; the pool is bounded only by the \
                 config file. Set --pool-mb explicitly if the machine is small"
            );
        }
    }
    let cfg = cfg;

    let sb = supabase::Supabase::from_env();
    match &sb {
        Some(s) => tracing::info!(project = %s.project(), "supabase configured"),
        None => tracing::info!("supabase not configured; running with local files only"),
    }

    let mut engine = Engine::new(cfg.clone(), args.seed);
    if args.models.exists() {
        match predictor::Predictor::load_dir(&args.models, cfg.predictor.online_lr) {
            Ok(p) => {
                let horizons = p.loaded_horizons();
                tracing::info!(kind = p.kind().as_str(), ?horizons, "model bundles loaded");
                engine.predictor = p;
            }
            Err(e) => tracing::warn!("no model bundles loaded: {e}"),
        }
    }

    let sim = args.scenario.as_deref().and_then(Scenario::parse).map(|s| Sim {
        generator: Generator::new(s, args.seed),
        running: true,
        speed: 1.0,
    });

    let (tx, _) = broadcast::channel(64);
    let auth_state = auth::Auth::from_env()?;
    if auth_state.is_enforced() {
        tracing::info!("auth enforced: application keys on the data path, console logins for everything else");
    } else {
        tracing::warn!(
            "auth is OPEN: every route answers without a credential. Set AURA_AUTH=enforced \
             for anything with a public address"
        );
    }

    let app = Arc::new(App {
        engine: Mutex::new(engine),
        auth: Mutex::new(auth_state),
        capacity: Mutex::new(CapacityController::new(&cfg)),
        leases: Mutex::new(Leases::default()),
        sim: Mutex::new(sim),
        cfg,
        bench: Mutex::new(None),
        real_values: args.real_values,
        probe: Mutex::new(BackendProbe { enabled: args.real_backend, ..Default::default() }),
        tx,
        started: std::time::Instant::now(),
        supabase: sb,
    });

    // Pulling the active bundle happens after the server is constructed so a slow or
    // unreachable project delays nothing: the engine serves on its online predictor until
    // the bundles land.
    if app.supabase.is_some() {
        tokio::spawn(pull_models(app.clone()));
        tokio::spawn(load_keys(app.clone()));
    }
    if args.real_backend {
        if app.supabase.is_some() {
            tracing::info!("analytics rebuild cost will be calibrated from live Supabase queries");
            tokio::spawn(probe_backend(app.clone()));
        } else {
            tracing::warn!("--real-backend needs Supabase credentials; falling back to synthetic cost");
        }
    }
    if args.real_values {
        tracing::info!("storing object payloads for real; pool size is resident memory");
    }

    tokio::spawn(drive(app.clone()));
    if app.supabase.is_some() {
        tokio::spawn(ship_audit(app.clone()));
        // A breadcrumb in the control plane saying which build came up with which pool and
        // which fidelity flags. Without it, a benchmark row six hours later has no way to
        // say what produced it.
        let announce = app.clone();
        tokio::spawn(async move {
            let detail = json!({
                "version": env!("CARGO_PKG_VERSION"),
                "capacity_bytes": announce.cfg.cache.capacity_bytes,
                "real_values": announce.real_values,
                "real_backend": announce.probe.lock().enabled,
                "predictor": announce.engine.lock().predictor.kind().as_str(),
            });
            if let Some(sb) = announce.supabase.as_ref() {
                if let Err(e) = sb.push_event("engine_started", detail).await {
                    // aura_events.kind is a foreign key onto a lookup table, so an
                    // undeclared kind is refused rather than inserted.
                    tracing::warn!(
                        "startup event not recorded ({e}); if this is a 409, run \
                         training/sql/006_audit_log.sql to declare the new event kinds"
                    );
                }
            }
        });
    }

    let cors = tower_http::cors::CorsLayer::new()
        .allow_origin(tower_http::cors::Any)
        .allow_methods(tower_http::cors::Any)
        .allow_headers(tower_http::cors::Any);

    let router = Router::new()
        .route("/healthz", get(healthz))
        .route("/metrics", get(metrics))
        .route("/v1/cache/:key", get(cache_get).put(cache_put))
        .route("/v1/cache/:key", delete(cache_delete))
        .route("/v1/cache/:key/refresh", post(cache_refresh))
        .route("/v1/cache/batch/get", post(cache_batch_get))
        .route("/v1/invalidate", post(invalidate))
        .route("/v1/version/bump", post(version_bump))
        .route("/v1/consistency", get(consistency_status))
        .route("/v1/refresh/queue", get(refresh_queue))
        .route("/v1/explain/recent", get(explain_recent))
        .route("/v1/audit", get(audit_log))
        .route("/v1/training/rows", get(training_rows))
        .route("/v1/feedback", get(feedback_stats))
        .route("/v1/explain/:key", get(explain_key))
        .route("/v1/stats", get(stats))
        .route("/v1/workload", get(workload))
        .route("/v1/policy", get(policy_get))
        .route("/v1/policy/override", post(policy_override))
        .route("/v1/capacity", get(capacity_get))
        .route("/v1/capacity/mode", post(capacity_mode))
        .route("/v1/applications", get(applications))
        .route("/v1/profiles", get(profiles_all))
        .route("/v1/profiles/knobs", get(profiles_knobs))
        .route(
            "/v1/applications/:app/profile",
            get(profile_get).put(profile_put).delete(profile_reset),
        )
        .route("/v1/nodes", get(nodes))
        .route("/v1/model/reload", post(model_reload))
        .route("/v1/supabase", get(supabase_status))
        .route("/v1/keys", get(keys_list).post(keys_mint))
        .route("/v1/keys/:id", delete(keys_revoke))
        .route("/v1/scenarios", get(scenarios))
        .route("/v1/sim/start", post(sim_start))
        .route("/v1/sim/stop", post(sim_stop))
        .route("/v1/sim/attack", post(sim_attack))
        .route("/v1/sim/speed", post(sim_speed))
        .route("/v1/sim/status", get(sim_status))
        .route("/v1/bench/run", post(bench_run))
        .route("/v1/bench/latest", get(bench_latest))
        .route("/v1/live", get(live_ws))
        .layer(tower_http::set_header::SetResponseHeaderLayer::overriding(
            axum::http::header::CACHE_CONTROL,
            HeaderValue::from_static("no-store"),
        ))
        .layer(axum::middleware::from_fn_with_state(app.clone(), gate))
        .layer(cors)
        .with_state(app);

    let addr = listen_addr(args.bind.as_deref())?;
    tracing::info!(%addr, "aura listening");
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, router).await?;
    Ok(())
}

/// One loop drives simulated traffic, the controller tick and the telemetry frame. Keeping
/// them on the same cadence means a frame never shows a half-applied capacity change.
async fn drive(app: Shared) {
    let mut ticker = tokio::time::interval(Duration::from_millis(250));
    loop {
        ticker.tick().await;
        let dt_ms = {
            let mut sim = app.sim.lock();
            match sim.as_mut() {
                Some(s) if s.running => 250.0 * s.speed,
                _ => 0.0,
            }
        };

        if dt_ms > 0.0 {
            let batch = {
                let mut sim = app.sim.lock();
                sim.as_mut().map(|s| s.generator.step(dt_ms)).unwrap_or_default()
            };
            let measured_db_ms = {
                let p = app.probe.lock();
                if p.enabled && p.samples > 0 { p.measured_ms } else { 0.0 }
            };
            let real_values = app.real_values;
            let mut eng = app.engine.lock();
            for r in batch {
                if eng.get(r.key_id, &r.context.application, r.ts_ms).is_none() {
                    let mut measured = r.context.regen;
                    let mut latency = r.regen_latency_ms;
                    // A measured round trip replaces the generator's guess for the one
                    // application that actually has a database behind it.
                    if measured_db_ms > 0.0 && r.context.application == "analytics" {
                        latency = latency - measured.db_ms + measured_db_ms;
                        measured.db_ms = measured_db_ms;
                    }
                    measured.latency_ms = latency;
                    let payload = if real_values {
                        Value::String("x".repeat(r.context.size_bytes.min(8_000_000) as usize))
                    } else {
                        Value::Null
                    };
                    eng.put(r.key_id, payload, &r.context, measured, r.ts_ms);
                }
            }
            eng.recompute_workload();
            // A real rebuild, not a clock reset. Each one goes through the origin's cost
            // model and lands on the backend ledger, because refreshing ahead of expiry is
            // cheaper than an expiry storm but it is not free — and a controller told it
            // was free would refresh everything, constantly.
            let to_refresh = eng.refresh_candidates(4);
            let at = eng.now_ms;
            for k in to_refresh {
                let payload = if real_values {
                    eng.store
                        .get(k)
                        .map(|e| Value::String("x".repeat(e.size_bytes.min(8_000_000) as usize)))
                } else {
                    None
                };
                eng.rebuild(k, payload, at);
            }
        }

        {
            let mut eng = app.engine.lock();
            // Close the feedback loop before the capacity decision, so the bandit and the
            // online model are judged on outcomes from this tick rather than the last one.
            eng.settle_feedback(256);
            let mut cap = app.capacity.lock();
            cap.maybe_apply(&mut eng, &app.cfg);
        }

        let frame = build_frame(&app);
        let _ = app.tx.send(frame.to_string());
    }
}

fn build_frame(app: &Shared) -> Value {
    let probe = *app.probe.lock();
    let eng = app.engine.lock();
    let cap = app.capacity.lock();
    let report = cap.report(&eng, &app.cfg);
    let consistency = eng.consistency_stats();
    let (lease_granted, lease_suppressed) = {
        let l = app.leases.lock();
        (l.granted, l.suppressed)
    };
    let (sim_running, sim_scenario, sim_speed, rps, vtime) = {
        let sim = app.sim.lock();
        match sim.as_ref() {
            Some(s) => (
                s.running,
                s.generator.scenario().id().to_string(),
                s.speed,
                s.generator.rps(),
                s.generator.now_ms() / 1000.0,
            ),
            None => (false, "none".to_string(), 1.0, 0.0, eng.now_ms / 1000.0),
        }
    };

    let elapsed_h = (eng.now_ms / 3_600_000.0).max(1e-6);
    let mut baselines = serde_json::Map::new();
    let mut savings = serde_json::Map::new();
    let total = eng.ledger.total();
    for s in eng.shadows.iter() {
        let shadow_total = s.total_usd();
        baselines.insert(s.policy.as_str().to_string(), json!({
            "total_usd": round4(shadow_total),
            "regen_usd": round4(s.cost_usd),
            "penalty_usd": round4(s.penalty_usd),
            "holding_usd": round4(s.holding_usd),
            "hit_rate": round4(s.hit_rate())
        }));
        let gain = if shadow_total > 0.0 {
            (shadow_total - total) / shadow_total
        } else {
            0.0
        };
        savings.insert(s.policy.as_str().to_string(), json!(round4(gain)));
    }

    let apps: Vec<Value> = eng
        .apps
        .iter()
        .map(|(name, st)| {
            let avg = if st.regen_count > 0 {
                st.regen_ms_total / st.regen_count as f64
            } else {
                0.0
            };
            json!({
                "application": name,
                "requests": st.requests,
                "hit_rate": round4(if st.requests > 0 { st.hits as f64 / st.requests as f64 } else { 0.0 }),
                "avg_object_bytes": if st.requests > 0 { st.bytes_total / st.requests } else { 0 },
                "regen_p50_ms": round2(avg),
                "cost_usd": round4(st.cost_usd),
                "cost_profile": cost_profile(name),
            })
        })
        .collect();

    json!({
        "t": now_epoch_ms(),
        "virtual_time_s": round2(vtime),
        "sim": { "running": sim_running, "scenario": sim_scenario, "speed": sim_speed, "rps": round2(rps) },
        "traffic": { "rps": round2(rps) },
        "layers": {
            "l1": { "hits": eng.store.l1_stats.hits, "misses": eng.store.l1_stats.misses,
                    "hit_rate": round4(eng.store.l1_stats.hit_rate()) },
            "l2": { "hits": eng.store.l2_stats.hits, "misses": eng.store.l2_stats.misses,
                    "hit_rate": round4(eng.store.l2_stats.hit_rate()),
                    "byte_hit_rate": round4(eng.store.l2_stats.byte_hit_rate()) },
            "backend": { "requests": eng.store.l2_stats.misses }
        },
        "latency": {
            "p50_ms": round2(eng.latency_quantile(0.5)),
            "p95_ms": round2(eng.latency_quantile(0.95)),
            "p99_ms": round2(eng.latency_quantile(0.99))
        },
        "cost": {
            "backend_usd": round4(eng.ledger.backend_usd),
            "cache_usd": round4(eng.ledger.cache_usd),
            "sla_penalty_usd": round4(eng.ledger.sla_penalty_usd),
            "total_usd": round4(total),
            "saved_vs_no_cache_usd": round4(eng.ledger.no_cache_usd - total),
            "baselines": baselines,
            "savings_vs": savings,
            "burn_rate_usd_per_hour": round2(total / elapsed_h)
        },
        "capacity": report,
        "host": {
            "available_bytes": hostmem::snapshot().0,
            "total_bytes": hostmem::snapshot().1
        },
        "workload": {
            "regime": eng.regime.as_str(),
            "confidence": round4(eng.regime_confidence),
            "features": {
                "burstiness": round4(eng.workload.burstiness),
                "entropy": round4(eng.workload.entropy),
                "working_set_growth": round4(eng.workload.working_set_growth),
                "reuse_distance_p50": round2(eng.workload.reuse_distance_p50),
                "popularity_shift": round4(eng.workload.popularity_shift),
                "scan_score": round4(eng.workload.scan_score)
            }
        },
        "policy": {
            "mixture": eng.bandit.mixture_map(),
            "bandit": app.cfg.bandit.kind,
            "ml_influence": round4(eng.predictor.confidence()),
            "predictor": eng.predictor.kind().as_str(),
            "predictor_confidence": round4(eng.predictor.confidence()),
            "bandit_regret": round4(eng.bandit.regret),
            "override": eng.override_policy.map(|p| p.as_str())
        },
        "engine": {
            "requests": eng.requests,
            "stale_serves": eng.stale_serves,
            "admissions": eng.store.admissions,
            "admissions_rejected": eng.rejections,
            "evictions": eng.store.evictions,
            "refreshes": eng.refreshes,
            "expirations": eng.store.expirations,
            "inference_calls": eng.predictor.calls,
            "resident_objects": eng.store.len(),
            "used_bytes": eng.store.used_bytes(),
            "capacity_bytes": eng.store.capacity_bytes(),
            "decision_overhead_us_p50": round2(eng.overhead_p50())
        },
        // Kept apart from `engine` on purpose. Eviction means the cache was short of space,
        // invalidation means it was wrong, expiry means time passed — three different
        // events, and a dashboard that adds them up hides the only one that is a bug.
        "consistency": {
            "tracked_keys": consistency.tracked_keys,
            "tracked_tags": consistency.tracked_tags,
            "invalidations": consistency.invalidations,
            "keys_invalidated": consistency.keys_invalidated,
            "soft_invalidations": consistency.soft_invalidations,
            "version_bumps": consistency.version_bumps,
            "stale_serves": consistency.stale_serves,
            "expired": consistency.expired,
            "evicted": consistency.evicted,
            "refresh_backlog": eng.refresh_queue.len(),
            "namespaces": eng.consistency.versions(),
            "single_flight": {
                "leases_granted": lease_granted,
                "origin_calls_suppressed": lease_suppressed
            }
        },
        "auth": {
            "enforced": app.auth.lock().is_enforced()
        },
        "fidelity": {
            // Where the requests came from, not what we wish they came from. The generator
            // only runs when a scenario was started; with it stopped, every request on this
            // engine arrived over HTTP from an application that did the work itself.
            "traffic": if sim_running { "simulated" } else { "applications" },
            "values_stored": app.real_values,
            "backend": {
                "enabled": probe.enabled,
                "measured": probe.samples > 0,
                "measured_db_ms": round2(probe.measured_ms),
                "p95_db_ms": round2(probe.p95_ms),
                "samples": probe.samples,
                "failures": probe.failures,
                "response_bytes": probe.bytes
            }
        },
        "applications": apps,
        "events": eng.events.iter().rev().take(20).collect::<Vec<_>>(),
        // The audit log rides the socket rather than being polled. Every entry carries a
        // monotonic `seq`, so the client keeps its own scrollback and drops what it has
        // already seen; a dropped frame costs nothing and a reconnect replays the tail.
        "audit": eng.audit.recent(24),
        "recent_decisions": eng.explains.iter().rev().take(12).collect::<Vec<_>>()
    })
}

/// Bytes as a person writes them, for the boot-time log lines. The engine has its own
/// formatter for telemetry; this one exists so startup does not depend on the engine being
/// constructed yet.
fn aura_server_bytes(n: u64) -> String {
    let n = n as f64;
    const KB: f64 = 1024.0;
    if n < KB * KB {
        format!("{:.0} KB", n / KB)
    } else if n < KB * KB * KB {
        format!("{:.0} MB", n / (KB * KB))
    } else {
        format!("{:.2} GB", n / (KB * KB * KB))
    }
}

fn cost_profile(app: &str) -> &'static str {
    match app {
        "analytics" => "db_heavy",
        "content" => "gpu_heavy",
        "recommendation" => "mixed",
        _ => "unknown",
    }
}


/// One gate in front of every route.
///
/// Written as a layer rather than per handler so that a route added later cannot be added
/// without an answer to "who may call this": `need_for` classifies by path and method, and
/// anything it does not recognise falls into the strictest class.
async fn gate(
    State(app): State<Shared>,
    mut req: axum::extract::Request,
    next: axum::middleware::Next,
) -> axum::response::Response {
    let method = req.method().as_str().to_string();
    let path = req.uri().path().to_string();
    let need = auth::need_for(&method, &path);

    // The socket handshake cannot carry a header from a browser, so the console passes its
    // token in the query string there. Everything else uses Authorization.
    let bearer = req
        .headers()
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .map(str::to_string)
        .or_else(|| {
            req.uri().query().and_then(|q| {
                q.split('&').find_map(|kv| kv.strip_prefix("token=").map(str::to_string))
            })
        });

    let verdict = app.auth.lock().authorise(need, bearer.as_deref());
    match verdict {
        Ok(caller) => {
            req.extensions_mut().insert(caller);
            next.run(req).await
        }
        Err(denial) => (
            StatusCode::from_u16(denial.status()).unwrap_or(StatusCode::UNAUTHORIZED),
            Json(json!({ "error": denial.message(), "path": path })),
        )
            .into_response(),
    }
}

/// Adopt the keys issued in earlier runs. Failure here is loud but not fatal: the engine
/// still serves, and newly minted keys still work; it is the old ones that stop.
async fn load_keys(app: Shared) {
    let Some(sb) = app.supabase.as_ref() else { return };
    match sb.api_keys().await {
        Ok(rows) => {
            let keys: Vec<auth::ApiKey> = rows.iter().filter_map(auth::ApiKey::from_row).collect();
            let n = keys.len();
            app.auth.lock().adopt(keys);
            if n > 0 {
                tracing::info!("adopted {n} application keys from the control plane");
            }
        }
        Err(e) => tracing::warn!(
            "could not read application keys ({e}); keys issued before this restart will \
             not be recognised. Apply training/sql/007_api_keys.sql if the table is missing"
        ),
    }
}

/// Keys this engine issued. Hashes and hints only: the secret existed once, at mint time.
async fn keys_list(State(app): State<Shared>) -> Json<Value> {
    let a = app.auth.lock();
    Json(json!({ "enforced": a.is_enforced(), "keys": a.keys() }))
}

#[derive(Deserialize)]
struct MintBody {
    application: String,
}

/// Mint a key for an application. The secret comes back exactly once; there is nowhere to
/// read it from afterwards, which is the point.
async fn keys_mint(
    State(app): State<Shared>,
    Json(body): Json<MintBody>,
) -> impl IntoResponse {
    let name = body.application.trim().to_string();
    if name.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({ "error": "an application name is required; the key is that application's identity" })),
        )
            .into_response();
    }
    let (secret, key) = {
        let mut rng = || {
            let mut b = [0u8; 8];
            getrandom_bytes(&mut b);
            u64::from_le_bytes(b)
        };
        app.auth.lock().mint(&name, &mut rng)
    };
    let now = { app.engine.lock().now_ms };
    app.engine.lock().audit.record(
        now,
        audit::AuditKind::ModelLoad,
        audit::Severity::Notice,
        &key.id,
        &name,
        format!("Issued an application key to {name}. Every request carrying it is attributed to that application."),
        vec![audit::Fact::new("key", key.hint.clone())],
        0.0,
    );
    if let Some(sb) = app.supabase.as_ref() {
        // Persisted before it is returned. A key the caller holds but the control plane
        // never saw would stop working at the next restart, silently.
        if let Err(e) = sb.insert_api_key(&key.as_row()).await {
            tracing::warn!("key {} was not persisted ({e}); it will not survive a restart", key.id);
        }
    }
    (StatusCode::CREATED, Json(json!({ "secret": secret, "key": key }))).into_response()
}

async fn keys_revoke(State(app): State<Shared>, Path(id): Path<String>) -> impl IntoResponse {
    let found = app.auth.lock().revoke(&id);
    if found {
        if let Some(sb) = app.supabase.as_ref() {
            if let Err(e) = sb.revoke_api_key(&id).await {
                tracing::warn!("key {id} revoked here but not in the control plane ({e}); it \
                                will come back at the next restart");
            }
        }
    }
    if found {
        (StatusCode::OK, Json(json!({ "revoked": id })))
    } else {
        (StatusCode::NOT_FOUND, Json(json!({ "error": "no such key" })))
    }
}

/// Randomness for key material. `getrandom` through the OS rather than the engine's seeded
/// RNG: that one is deliberately reproducible, which is exactly wrong for a secret.
fn getrandom_bytes(buf: &mut [u8]) {
    use std::io::Read;
    if let Ok(mut f) = std::fs::File::open("/dev/urandom") {
        if f.read_exact(buf).is_ok() {
            return;
        }
    }
    // Windows and anything else without /dev/urandom: mix several clock readings, which is
    // weaker than a CSPRNG and is why this path also logs.
    tracing::warn!("no OS entropy source; key material is derived from the clock");
    let mut acc = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0x9E3779B97F4A7C15);
    for b in buf.iter_mut() {
        acc ^= acc << 13;
        acc ^= acc >> 7;
        acc ^= acc << 17;
        *b = (acc & 0xff) as u8;
    }
}

async fn healthz(State(app): State<Shared>) -> Json<Value> {
    Json(json!({
        "ok": true,
        "version": env!("CARGO_PKG_VERSION"),
        "uptime_s": app.started.elapsed().as_secs()
    }))
}

async fn metrics(State(app): State<Shared>) -> String {
    let eng = app.engine.lock();
    let mut s = String::new();
    s.push_str("# TYPE aura_hit_rate gauge\n");
    s.push_str(&format!("aura_hit_rate {}\n", eng.store.l2_stats.hit_rate()));
    s.push_str("# TYPE aura_used_bytes gauge\n");
    s.push_str(&format!("aura_used_bytes {}\n", eng.store.used_bytes()));
    s.push_str("# TYPE aura_capacity_bytes gauge\n");
    s.push_str(&format!("aura_capacity_bytes {}\n", eng.store.capacity_bytes()));
    s.push_str("# TYPE aura_cost_usd_total counter\n");
    s.push_str(&format!("aura_cost_usd_total {}\n", eng.ledger.total()));
    s.push_str("# TYPE aura_evictions_total counter\n");
    s.push_str(&format!("aura_evictions_total {}\n", eng.store.evictions));
    s.push_str("# TYPE aura_requests_total counter\n");
    s.push_str(&format!("aura_requests_total {}\n", eng.requests));
    s
}

#[derive(Deserialize)]
struct AppQuery {
    #[serde(default)]
    application: Option<String>,
}

/// The read path, including the part that decides who is allowed to rebuild a miss.
///
/// A miss does not simply say "not here". It says whether *this* caller should go to the
/// origin (`rebuild: true`) or wait for someone who already is (`rebuild: false`, with a
/// hint of how long). That is what stops a flash crowd on a cold key turning into a
/// thousand identical origin calls, and it works from outside the application, which is the
/// only place a cache like this can stand.
async fn cache_get(
    State(app): State<Shared>,
    Path(key): Path<String>,
    Query(q): Query<AppQuery>,
) -> impl IntoResponse {
    let id = key_id(&key);
    let now = now_ms(&app);
    let hit = {
        let mut eng = app.engine.lock();
        let found = eng.get(id, q.application.as_deref().unwrap_or("default"), now);
        let stale = eng.consistency.is_stale(id);
        // Age before value: reading `e.value` moves the payload out of the entry, and the
        // entry cannot be asked anything afterwards.
        found.map(|e| (e.age_ms(now), stale, e.value))
    };
    match hit {
        Some((age_ms, stale, value)) => {
            (
                StatusCode::OK,
                Json(json!({
                    "hit": true,
                    "value": value,
                    "age_ms": round2(age_ms),
                    // Told plainly rather than buried, because a caller that cares about
                    // freshness needs to be able to choose, and one that does not can
                    // ignore the field entirely.
                    "stale": stale,
                    "layer": "L2",
                    "latency_us": 40
                })),
            )
        }
        None => {
            let lease_ms = app.cfg.engine.rebuild_lease_ms;
            let (rebuild, wait_ms) = app.leases.lock().acquire(id, now, lease_ms);
            (
                StatusCode::NOT_FOUND,
                Json(json!({
                    "hit": false,
                    "reason": "miss",
                    "rebuild": rebuild,
                    "retry_after_ms": round2(wait_ms),
                    "lease_ms": if rebuild { lease_ms } else { 0.0 }
                })),
            )
        }
    }
}

#[derive(Deserialize)]
struct PutBody {
    value: Value,
    context: ObjectContext,
    #[serde(default)]
    measured: Option<CostVector>,
}

async fn cache_put(
    State(app): State<Shared>,
    Path(key): Path<String>,
    caller: Option<axum::Extension<auth::Caller>>,
    Json(body): Json<PutBody>,
) -> Json<Value> {
    let id = key_id(&key);
    let now = now_ms(&app);
    // The rebuild is done; whoever was waiting on the lease may proceed.
    app.leases.lock().release(id);
    let mut eng = app.engine.lock();
    let mut body = body;
    // A key is issued to one application, so the credential names the application and the
    // body cannot. Otherwise one service could file its objects, its costs and its profile
    // under another service's name, by typing a different string.
    if let Some(axum::Extension(c)) = caller.as_ref() {
        if let Some(name) = c.application() {
            body.context.application = name.to_string();
        }
    }
    let measured = body.measured.unwrap_or(body.context.regen);
    let version = body
        .context
        .namespace
        .as_deref()
        .map(|ns| eng.consistency.version(ns));
    let (decision, evicted) = eng.put(id, body.value, &body.context, measured, now);
    Json(json!({
        "admitted": decision.action == aura_core::types::Action::Admit,
        "reason_code": decision.reason_code,
        "evicted": evicted.iter().map(|k| k.to_string()).collect::<Vec<_>>(),
        "tags": body.context.depends_on,
        "namespace_version": version,
        "used_bytes": eng.store.used_bytes()
    }))
}

async fn cache_delete(State(app): State<Shared>, Path(key): Path<String>) -> Json<Value> {
    let id = key_id(&key);
    app.leases.lock().release(id);
    let mut eng = app.engine.lock();
    let removed = eng.store.remove(id).is_some();
    // Deleting an object without forgetting what it depended on leaves a tag pointing at
    // nothing, which makes the next invalidation report work it did not do.
    eng.consistency.forget(id);
    Json(json!({ "removed": removed }))
}

/// Rebuild one object now.
///
/// With a `value`, the caller has already been to the origin and this stores the result.
/// Without one, the key is only queued: the engine will not pretend a rebuild happened by
/// resetting the clock on bytes nobody rebuilt.
async fn cache_refresh(
    State(app): State<Shared>,
    Path(key): Path<String>,
    body: Option<Json<RefreshBody>>,
) -> Json<Value> {
    let now = now_ms(&app);
    let id = key_id(&key);
    let value = body.and_then(|Json(b)| b.value);
    let mut eng = app.engine.lock();
    if !eng.store.contains(id) {
        return Json(json!({
            "rebuilt": false,
            "reason": "not resident; there is nothing to refresh"
        }));
    }
    let had_value = value.is_some();
    let rebuilt = eng.rebuild(id, value, now);
    Json(json!({
        "rebuilt": rebuilt,
        "value_replaced": had_value,
        // Said out loud, because a refresh that only moves the clock is the failure mode
        // this endpoint used to have.
        "note": if had_value {
            "value replaced and the rebuild charged to the backend ledger"
        } else {
            "rebuild charged, but no new value was supplied — send one to actually refresh it"
        }
    }))
}

#[derive(Deserialize, Default)]
struct RefreshBody {
    /// What the origin returned. Omit to charge the rebuild without replacing the payload,
    /// which is what the simulator does, where bytes are accounted rather than held.
    #[serde(default)]
    value: Option<Value>,
}

#[derive(Deserialize)]
struct InvalidateBody {
    tags: Vec<String>,
    /// `hard` removes immediately — for anything where being wrong is unacceptable.
    /// `soft` marks stale, so the next reader gets one old value while a rebuild runs.
    #[serde(default = "hard_mode")]
    mode: String,
    #[serde(default = "manual_source")]
    source: String,
}

fn hard_mode() -> String {
    "hard".into()
}
fn manual_source() -> String {
    "manual".into()
}

/// The endpoint the database trigger, the application SDK and a human all call when
/// something the cache derived from has changed.
async fn invalidate(State(app): State<Shared>, Json(body): Json<InvalidateBody>) -> impl IntoResponse {
    if body.tags.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({ "error": "no tags given; invalidation needs something to match" })),
        );
    }
    let mode = match body.mode.as_str() {
        "soft" => InvalidationMode::Soft,
        "hard" => InvalidationMode::Hard,
        other => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({ "error": "mode must be hard or soft", "got": other })),
            )
        }
    };
    let now = now_ms(&app);
    let result = app.engine.lock().invalidate(&body.tags, mode, &body.source, now);
    // A key that was just dropped is a key someone is about to rebuild, so no stale lease
    // may stand in their way.
    {
        let mut leases = app.leases.lock();
        for k in result.keys_hard.iter().chain(result.keys_soft.iter()) {
            leases.release(*k);
        }
    }
    (
        StatusCode::OK,
        Json(json!({
            "tags": result.tags,
            "mode": body.mode,
            "matched": result.matched,
            "keys_hard": result.keys_hard.len(),
            "keys_soft": result.keys_soft.len()
        })),
    )
}

#[derive(Deserialize)]
struct VersionBumpBody {
    namespace: String,
}

/// Retire a generation of objects without deleting any of them.
///
/// This is how a model redeploy is handled. Flushing instead would empty a large part of
/// the cache at once and send the whole miss stream at the origin, which is the cache
/// causing the outage it exists to prevent.
async fn version_bump(State(app): State<Shared>, Json(body): Json<VersionBumpBody>) -> Json<Value> {
    let now = now_ms(&app);
    let version = app.engine.lock().bump_version(&body.namespace, now);
    Json(json!({ "namespace": body.namespace, "version": version }))
}

async fn consistency_status(State(app): State<Shared>) -> Json<Value> {
    let eng = app.engine.lock();
    let leases = app.leases.lock();
    let stats = eng.consistency_stats();
    Json(json!({
        "tracked_keys": stats.tracked_keys,
        "tracked_tags": stats.tracked_tags,
        "namespaces": eng.consistency.versions(),
        "invalidations": stats.invalidations,
        "keys_invalidated": stats.keys_invalidated,
        "soft_invalidations": stats.soft_invalidations,
        "version_bumps": stats.version_bumps,
        "stale_serves": stats.stale_serves,
        "expired": stats.expired,
        "evicted": stats.evicted,
        "refresh_backlog": eng.refresh_queue.len(),
        "single_flight": {
            "leases_granted": leases.granted,
            "origin_calls_suppressed": leases.suppressed,
            "leases_held": leases.held.len()
        },
        "recent": eng.consistency.recent_events()
    }))
}

/// What the cache owes a rebuild on, with enough context for the application to do it.
///
/// The engine knows *which* objects are about to go bad and what they cost; only the
/// application knows how to make one. So the engine publishes the list and the application
/// PUTs the new values back. That split is what keeps this cache application-agnostic.
async fn refresh_queue(State(app): State<Shared>, Query(q): Query<LimitQuery>) -> Json<Value> {
    let eng = app.engine.lock();
    let items: Vec<Value> = eng
        .refresh_backlog(q.limit.min(500))
        .into_iter()
        .map(|(k, ctx, remaining)| {
            json!({
                "key": k.to_string(),
                "application": ctx.application,
                "object_type": ctx.object_type,
                "size_bytes": ctx.size_bytes,
                "ttl_ms": ctx.ttl_ms,
                "ttl_remaining_frac": round4(remaining),
                "last_regen": ctx.regen
            })
        })
        .collect();
    Json(json!({ "count": items.len(), "pending": eng.refresh_queue.len(), "items": items }))
}

#[derive(Deserialize)]
struct BatchGet {
    keys: Vec<String>,
    #[serde(default)]
    application: Option<String>,
}

async fn cache_batch_get(State(app): State<Shared>, Json(body): Json<BatchGet>) -> Json<Value> {
    let now = now_ms(&app);
    let application = body.application.unwrap_or_else(|| "default".into());
    let mut eng = app.engine.lock();
    let mut out = serde_json::Map::new();
    for k in body.keys {
        let id = key_id(&k);
        let v = match eng.get(id, &application, now) {
            Some(e) => json!({ "hit": true, "value": e.value, "age_ms": round2(e.age_ms(now)) }),
            None => json!({ "hit": false }),
        };
        out.insert(k, v);
    }
    Json(json!({ "results": out }))
}

#[derive(Deserialize)]
struct LimitQuery {
    #[serde(default = "default_limit")]
    limit: usize,
}

fn default_limit() -> usize {
    50
}

/// The running account of what the cache did, in sentences.
///
/// `/v1/explain/{key}` answers "why this object" for a question you already knew to ask.
/// This is the log you read when you do not: every decision that cost money or touched
/// correctness, written as prose with the numbers that produced it.
async fn audit_log(State(app): State<Shared>, Query(q): Query<AuditQuery>) -> impl IntoResponse {
    let eng = app.engine.lock();
    let limit = q.limit.min(500);
    // Filtering by kind server-side matters for the dashboard: an operator chasing a
    // correctness problem wants the invalidations, and pulling five hundred admissions to
    // find three of them is the wrong shape of request.
    let entries = match q.kind.as_deref() {
        Some(k) => eng.audit.recent_of(k, limit),
        None => eng.audit.recent(limit),
    };
    Json(json!({
        "entries": entries,
        "count": entries.len(),
        "held": eng.audit.len(),
        "empty": eng.audit.is_empty(),
        "suppressed_routine": eng.audit.suppressed,
        "pending_shipment": eng.audit.pending_shipment(),
    }))
}

#[derive(Deserialize)]
struct AuditQuery {
    #[serde(default = "default_limit")]
    limit: usize,
    /// One of the AuditKind names: admit, reject, evict, refresh, expire, invalidate,
    /// version_bump, scale_up, scale_down, scale_hold, policy_shift, model_load,
    /// regime_change, pressure.
    #[serde(default)]
    kind: Option<String>,
}

/// Drain matured decisions as labelled training rows.
///
/// These are not a log of what the cache did; they are a dataset. Each row carries the
/// exact feature vector a decision was made from and the labels that arrived up to ten
/// minutes later, so the trainer never has to reconstruct features and can never disagree
/// with the engine about what they mean.
///
/// Draining is destructive by design: a row handed out once has been consumed, which keeps
/// the buffer bounded and makes repeated polling cheap.
async fn training_rows(State(app): State<Shared>, Query(q): Query<LimitQuery>) -> impl IntoResponse {
    let limit = q.limit.min(200_000);
    let rows = app.engine.lock().drain_training_rows(limit);
    let count = rows.len();
    Json(json!({
        "rows": rows,
        "count": count,
        "feature_names": aura_core::features::FEATURE_NAMES,
        "horizons_ms": crate::feedback::HORIZONS_MS,
    }))
}

/// How well the system's own predictions have been holding up.
///
/// `calibration_error` is the number to watch: if the model says 0.70 and reality comes
/// back 0.42, it is confidently wrong and the confidence floor is the only thing keeping
/// the cache sane.
async fn feedback_stats(State(app): State<Shared>) -> impl IntoResponse {
    let eng = app.engine.lock();
    let stats = serde_json::to_value(eng.journal_stats()).unwrap_or(Value::Null);
    let (kept, refused) = (eng.calib_kept, eng.calib_refused);
    Json(json!({
        "overall": stats,
        // Split, because one number hides the asymmetry: over-confidence about what we
        // kept wastes memory, over-confidence about what we refused forces rebuilds.
        "calibration": {
            "kept":    { "predicted": round4(kept.mean_predicted()),
                         "realised":  round4(kept.mean_realised()),
                         "error":     round4(kept.error()),
                         "decisions": kept.n },
            "refused": { "predicted": round4(refused.mean_predicted()),
                         "realised":  round4(refused.mean_realised()),
                         "error":     round4(refused.error()),
                         "decisions": refused.n }
        },
        "rows_ready": eng.journal.completed_len(),
        "in_flight": eng.journal.in_flight(eng.now_ms, 20)
    }))
}

async fn explain_recent(State(app): State<Shared>, Query(q): Query<LimitQuery>) -> Json<Value> {
    let eng = app.engine.lock();
    let decisions: Vec<_> = eng.explains.iter().rev().take(q.limit).collect();
    Json(json!({ "decisions": decisions }))
}

async fn explain_key(State(app): State<Shared>, Path(key): Path<String>) -> impl IntoResponse {
    let eng = app.engine.lock();
    let id = key_id(&key);
    match eng.explain_key(id) {
        Some(r) => {
            let mut body = json!(r);
            // Correctness alongside the economics: the reasons say why the cache wants to
            // keep this object, and this says what would make it wrong to.
            if let Some(obj) = body.as_object_mut() {
                obj.insert(
                    "depends_on".into(),
                    json!(eng.consistency.tags_of(id).map(|t| t.to_vec()).unwrap_or_default()),
                );
                obj.insert("stale".into(), json!(eng.consistency.is_stale(id)));
                obj.insert("resident".into(), json!(eng.store.contains(id)));
            }
            (StatusCode::OK, Json(body))
        }
        None => (
            StatusCode::NOT_FOUND,
            Json(json!({ "present": false, "key": key })),
        ),
    }
}

async fn stats(State(app): State<Shared>) -> Json<Value> {
    Json(build_frame(&app))
}

async fn workload(State(app): State<Shared>) -> Json<Value> {
    let eng = app.engine.lock();
    Json(json!({
        "regime": eng.regime.as_str(),
        "confidence": round4(eng.regime_confidence),
        "features": {
            "burstiness": round4(eng.workload.burstiness),
            "entropy": round4(eng.workload.entropy),
            "working_set_growth": round4(eng.workload.working_set_growth),
            "reuse_distance_p50": round2(eng.workload.reuse_distance_p50),
            "popularity_shift": round4(eng.workload.popularity_shift),
            "scan_score": round4(eng.workload.scan_score)
        }
    }))
}

async fn policy_get(State(app): State<Shared>) -> Json<Value> {
    let eng = app.engine.lock();
    Json(json!({
        "mixture": eng.bandit.mixture_map(),
        "bandit": app.cfg.bandit.kind,
        "ml_influence": round4(eng.predictor.confidence()),
        "predictor": eng.predictor.kind().as_str(),
        "predictor_source": eng.predictor.source.as_str(),
        "features": eng.predictor.feature_count(),
        "holdout_auc": eng.predictor.holdout_auc(),
        "regime": eng.regime.as_str(),
        "override": eng.override_policy.map(|p| p.as_str())
    }))
}

#[derive(Deserialize)]
struct PolicyOverride {
    policy: String,
}

async fn policy_override(
    State(app): State<Shared>,
    Json(body): Json<PolicyOverride>,
) -> Json<Value> {
    let mut eng = app.engine.lock();
    eng.override_policy = if body.policy == "aura" || body.policy == "auto" {
        None
    } else {
        Policy::ALL.iter().copied().find(|p| p.as_str() == body.policy)
    };
    let active = eng.override_policy.map(|p| p.as_str()).unwrap_or("aura");
    eng.push_event("PolicyShift", &format!("operator forced {active}"));
    Json(json!({ "policy": active }))
}

async fn capacity_get(State(app): State<Shared>) -> Json<Value> {
    let eng = app.engine.lock();
    let cap = app.capacity.lock();
    Json(json!(cap.report(&eng, &app.cfg)))
}

#[derive(Deserialize)]
struct CapacityMode {
    mode: String,
    #[serde(default)]
    bytes: Option<u64>,
}

async fn capacity_mode(State(app): State<Shared>, Json(body): Json<CapacityMode>) -> Json<Value> {
    let mut cap = app.capacity.lock();
    cap.manual = body.mode == "manual";
    if let Some(b) = body.bytes {
        let mut eng = app.engine.lock();
        eng.set_capacity(b);
    }
    Json(json!({ "mode": if cap.manual { "manual" } else { "auto" } }))
}

async fn applications(State(app): State<Shared>) -> Json<Value> {
    let frame = build_frame(&app);
    Json(json!({ "profiles": frame.get("applications").cloned().unwrap_or(json!([])) }))
}

/// Every application the cache has seen, with the profile in force for it and whether that
/// profile was set by an operator or is still the default.
async fn profiles_all(State(app): State<Shared>) -> Json<Value> {
    let eng = app.engine.lock();
    let cap = eng.store.capacity_bytes().max(1);
    let mut rows: Vec<Value> = eng
        .apps
        .keys()
        .map(|name| {
            let held = *eng.resident_bytes.get(name).unwrap_or(&0);
            json!({
                "application": name,
                "customised": eng.profiles.is_customised(name),
                "resident_bytes": held,
                "pool_share": round4(held as f64 / cap as f64),
                "profile": eng.profiles.get(name),
            })
        })
        .collect();
    // An application configured before it has sent a single request still has to appear, or
    // the operator cannot pre-configure one.
    for (name, profile) in eng.profiles.customised() {
        if !eng.apps.contains_key(name) {
            rows.push(json!({
                "application": name,
                "customised": true,
                "resident_bytes": 0,
                "pool_share": 0.0,
                "profile": profile,
            }));
        }
    }
    rows.sort_by(|a, b| a["application"].as_str().cmp(&b["application"].as_str()));
    Json(json!({ "defaults": eng.profiles.defaults(), "applications": rows }))
}

/// What each knob does, served from the engine so the dashboard cannot describe a control
/// the engine does not have.
async fn profiles_knobs() -> Json<Value> {
    Json(profiles::knobs())
}

async fn profile_get(State(app): State<Shared>, Path(name): Path<String>) -> Json<Value> {
    let eng = app.engine.lock();
    Json(json!({
        "application": name,
        "customised": eng.profiles.is_customised(&name),
        "profile": eng.profiles.get(&name),
    }))
}

/// Partial update. Absent fields keep their current value, every number is clamped to a
/// range the engine can run in, and the result is the profile as it now stands rather than
/// an acknowledgement, so the dashboard renders what the engine actually accepted.
async fn profile_put(
    State(app): State<Shared>,
    Path(name): Path<String>,
    caller: Option<axum::Extension<auth::Caller>>,
    Json(body): Json<Value>,
) -> Json<Value> {
    let mut eng = app.engine.lock();
    let profile = eng.profiles.patch(&name, &body);
    let message = format!(
        "{name} now optimises for {}: horizons {:.2}/{:.2}/{:.2}, admission bar {:.2}x, \
         slow-response weight {:.1}x, refresh at {:.0}% of TTL, pool share cap {:.0}%.",
        profile.objective.as_str(),
        profile.horizon_weights[0],
        profile.horizon_weights[1],
        profile.horizon_weights[2],
        profile.admission_margin,
        profile.sla_weight,
        profile.soft_ttl_fraction * 100.0,
        profile.max_pool_share * 100.0,
    );
    let now = eng.now_ms;
    eng.audit.record(
        now,
        crate::audit::AuditKind::PolicyShift,
        crate::audit::Severity::Notice,
        &name,
        &name,
        message,
        vec![
            crate::audit::Fact::new("objective", profile.objective.as_str()),
            crate::audit::Fact::new("admission_margin", format!("{:.2}x", profile.admission_margin)),
            crate::audit::Fact::new("sla_weight", format!("{:.1}x", profile.sla_weight)),
            crate::audit::Fact::new(
                "changed_by",
                caller
                    .as_ref()
                    .map(|axum::Extension(c)| c.label())
                    .unwrap_or_else(|| "anonymous".to_string()),
            ),
        ],
        0.0,
    );
    Json(json!({ "application": name, "customised": true, "profile": profile }))
}

async fn profile_reset(State(app): State<Shared>, Path(name): Path<String>) -> Json<Value> {
    let mut eng = app.engine.lock();
    let existed = eng.profiles.reset(&name);
    Json(json!({
        "application": name,
        "customised": false,
        "was_customised": existed,
        "profile": eng.profiles.defaults(),
    }))
}

async fn nodes(State(app): State<Shared>) -> Json<Value> {
    let eng = app.engine.lock();
    Json(json!({
        "nodes": [{
            "id": "node-1",
            "capacity_bytes": eng.store.capacity_bytes(),
            "used_bytes": eng.store.used_bytes(),
            "keys": eng.store.len(),
            "ring_share": 1.0
        }]
    }))
}

#[derive(Deserialize)]
struct ModelReload {
    #[serde(default)]
    source: Option<String>,
    #[serde(default)]
    path: Option<String>,
}

async fn model_reload(State(app): State<Shared>, Json(body): Json<ModelReload>) -> Json<Value> {
    if body.source.as_deref() == Some("supabase") {
        if app.supabase.is_none() {
            return Json(json!({ "ok": false, "error": "supabase is not configured" }));
        }
        pull_models(app.clone()).await;
        let eng = app.engine.lock();
        return Json(json!({
            "ok": true,
            "source": "supabase",
            "kind": eng.predictor.kind().as_str(),
            "horizons": eng.predictor.loaded_horizons()
        }));
    }
    let dir = PathBuf::from(body.path.unwrap_or_else(|| "models".into()));
    let mut eng = app.engine.lock();
    match predictor::Predictor::load_dir(&dir, app.cfg.predictor.online_lr) {
        Ok(p) => {
            let horizons: Vec<String> = p.loaded_horizons().iter().map(|s| s.to_string()).collect();
            let kind = p.kind().as_str();
            let (name, features, auc) = (p.bundle_name(), p.feature_count(), p.holdout_auc());
            eng.predictor = p;
            eng.push_event("ModelReload", &format!("{kind} from {}", dir.display()));
            eng.audit_model(&name, features, &dir.display().to_string(), auc);
            Json(json!({
                "ok": true,
                "kind": kind,
                "horizons": horizons,
                "features": features,
                "holdout_auc": auc
            }))
        }
        Err(e) => Json(json!({ "ok": false, "error": e.to_string() })),
    }
}

async fn scenarios() -> Json<Value> {
    let list: Vec<Value> = Scenario::ALL
        .iter()
        .map(|s| {
            let spec = s.spec();
            json!({
                "id": spec.id,
                "name": spec.name,
                "description": spec.description,
                "unique_keys": spec.unique_keys,
                "base_rps": spec.base_rps,
                "attacks": spec.attacks.iter().map(|a| json!({
                    "id": a.as_str(), "description": a.description()
                })).collect::<Vec<_>>()
            })
        })
        .collect();
    Json(json!({ "scenarios": list }))
}

#[derive(Deserialize)]
struct SimStart {
    scenario: String,
    #[serde(default = "one")]
    speed: f64,
    #[serde(default = "seed42")]
    seed: u64,
}

fn one() -> f64 {
    1.0
}
fn seed42() -> u64 {
    42
}

async fn sim_start(State(app): State<Shared>, Json(body): Json<SimStart>) -> impl IntoResponse {
    let scenario = match Scenario::parse(&body.scenario) {
        Some(s) => s,
        None => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({ "error": "unknown scenario", "scenario": body.scenario })),
            )
        }
    };
    *app.sim.lock() = Some(Sim {
        generator: Generator::new(scenario, body.seed),
        running: true,
        speed: body.speed.clamp(0.1, 32.0),
    });
    app.engine
        .lock()
        .push_event("SimStart", &format!("scenario {}", scenario.id()));
    (
        StatusCode::OK,
        Json(json!({ "running": true, "scenario": scenario.id() })),
    )
}

async fn sim_stop(State(app): State<Shared>) -> Json<Value> {
    if let Some(s) = app.sim.lock().as_mut() {
        s.running = false;
    }
    Json(json!({ "running": false }))
}

#[derive(Deserialize)]
struct AttackBody {
    attack: String,
    #[serde(default = "thirty")]
    duration_s: f64,
}

fn thirty() -> f64 {
    30.0
}

async fn sim_attack(State(app): State<Shared>, Json(body): Json<AttackBody>) -> impl IntoResponse {
    let attack = match Attack::parse(&body.attack) {
        Some(a) => a,
        None => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({ "error": "unknown attack", "attack": body.attack })),
            )
        }
    };
    let mut sim = app.sim.lock();
    match sim.as_mut() {
        Some(s) => {
            s.generator.inject(attack, body.duration_s);
            drop(sim);
            app.engine.lock().push_event(
                "AttackStart",
                &format!("{} for {:.0}s", attack.as_str(), body.duration_s),
            );
            (
                StatusCode::OK,
                Json(json!({ "injected": attack.as_str(), "duration_s": body.duration_s })),
            )
        }
        None => (
            StatusCode::CONFLICT,
            Json(json!({ "error": "simulator is not running" })),
        ),
    }
}

#[derive(Deserialize)]
struct SpeedBody {
    speed: f64,
}

async fn sim_speed(State(app): State<Shared>, Json(body): Json<SpeedBody>) -> Json<Value> {
    let speed = body.speed.clamp(0.1, 32.0);
    if let Some(s) = app.sim.lock().as_mut() {
        s.speed = speed;
    }
    Json(json!({ "speed": speed }))
}

async fn sim_status(State(app): State<Shared>) -> Json<Value> {
    let sim = app.sim.lock();
    match sim.as_ref() {
        Some(s) => Json(json!({
            "running": s.running,
            "scenario": s.generator.scenario().id(),
            "speed": s.speed,
            "virtual_time_s": round2(s.generator.now_ms() / 1000.0),
            "rps": round2(s.generator.rps()),
            "emitted": s.generator.emitted(),
            "live_attacks": s.generator.live_attacks().iter().map(|a| a.as_str()).collect::<Vec<_>>()
        })),
        None => Json(json!({ "running": false })),
    }
}

#[derive(Deserialize)]
struct BenchBody {
    #[serde(default = "default_scenario")]
    scenario: String,
    #[serde(default = "default_policies")]
    policies: Vec<String>,
    #[serde(default = "default_capacity")]
    capacity_bytes: u64,
    #[serde(default = "default_requests")]
    requests: usize,
    #[serde(default = "seed42")]
    seed: u64,
}

fn default_scenario() -> String {
    "expensive_tail".into()
}
/// Everything, by default. A benchmark that quietly omits the strongest baselines is a
/// benchmark nobody should believe, and the three the brief names — LRU, LFU, GDS — are
/// the least interesting of the nine.
fn default_policies() -> Vec<String> {
    let mut v: Vec<String> = aura_core::policies::BASELINE_NAMES
        .iter()
        .map(|s| s.to_string())
        .collect();
    v.push("aura".into());
    v
}
fn default_capacity() -> u64 {
    268_435_456
}
fn default_requests() -> usize {
    60_000
}

async fn bench_run(State(app): State<Shared>, Json(body): Json<BenchBody>) -> impl IntoResponse {
    let scenario = match Scenario::parse(&body.scenario) {
        Some(s) => s,
        None => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({ "error": "unknown scenario" })),
            )
        }
    };
    let cfg = app.cfg.clone();
    let requests = body.requests.min(400_000);
    let report = tokio::task::spawn_blocking(move || {
        bench::run(
            &cfg,
            scenario,
            &body.policies,
            body.capacity_bytes,
            requests,
            body.seed,
        )
    })
    .await;
    match report {
        Ok(r) => {
            *app.bench.lock() = Some(r.clone());
            if let Some(sb) = app.supabase.clone() {
                let payload = json!(r);
                let version = env!("CARGO_PKG_VERSION").to_string();
                // Publishing must not hold up the response: the caller wants the numbers,
                // not confirmation that a third party stored them.
                tokio::spawn(async move {
                    if let Err(e) = sb.publish_benchmark(&payload, &version).await {
                        tracing::warn!("benchmark not published: {e}");
                    }
                });
            }
            (StatusCode::OK, Json(json!(r)))
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": e.to_string() })),
        ),
    }
}

async fn bench_latest(State(app): State<Shared>) -> impl IntoResponse {
    match app.bench.lock().clone() {
        Some(r) => (StatusCode::OK, Json(json!(r))),
        None => (
            StatusCode::NOT_FOUND,
            Json(json!({ "error": "no benchmark has been run" })),
        ),
    }
}

async fn live_ws(State(app): State<Shared>, ws: WebSocketUpgrade) -> impl IntoResponse {
    ws.on_upgrade(move |socket| live_socket(socket, app))
}

async fn live_socket(mut socket: WebSocket, app: Shared) {
    let mut rx = app.tx.subscribe();
    let _ = socket
        .send(Message::Text(build_frame(&app).to_string()))
        .await;
    loop {
        tokio::select! {
            frame = rx.recv() => match frame {
                Ok(text) => { if socket.send(Message::Text(text)).await.is_err() { break; } }
                Err(broadcast::error::RecvError::Lagged(_)) => continue,
                Err(_) => break,
            },
            incoming = socket.recv() => match incoming {
                Some(Ok(Message::Text(t))) => handle_client_message(&app, &t),
                Some(Ok(Message::Close(_))) | None => break,
                Some(Err(_)) => break,
                _ => {}
            }
        }
    }
}

fn handle_client_message(app: &Shared, text: &str) {
    let v: Value = match serde_json::from_str(text) {
        Ok(v) => v,
        Err(_) => return,
    };
    match v.get("type").and_then(|t| t.as_str()) {
        Some("attack") => {
            if let Some(a) = v.get("attack").and_then(|a| a.as_str()).and_then(Attack::parse) {
                let d = v.get("duration_s").and_then(|d| d.as_f64()).unwrap_or(30.0);
                if let Some(s) = app.sim.lock().as_mut() {
                    s.generator.inject(a, d);
                }
                app.engine
                    .lock()
                    .push_event("AttackStart", &format!("{} for {d:.0}s", a.as_str()));
            }
        }
        Some("speed") => {
            if let Some(sp) = v.get("speed").and_then(|s| s.as_f64()) {
                if let Some(s) = app.sim.lock().as_mut() {
                    s.speed = sp.clamp(0.1, 32.0);
                }
            }
        }
        _ => {}
    }
}

/// Fetches every active bundle from Supabase and hands them to the predictor. Failure is
/// logged and dropped: the engine already has a working predictor.
async fn pull_models(app: Shared) {
    let sb = match app.supabase.as_ref() {
        Some(s) => s.clone(),
        None => return,
    };
    let names = match sb.active_models().await {
        Ok(n) if !n.is_empty() => n,
        Ok(_) => {
            tracing::info!("supabase has no active model rows yet");
            return;
        }
        Err(e) => {
            tracing::warn!("could not list models: {e}");
            return;
        }
    };
    let mut loaded = Vec::new();
    for name in names {
        match sb.active_bundle(&name).await {
            Ok((version, bytes)) => match serde_json::from_slice::<predictor::ModelBundle>(&bytes) {
                Ok(bundle) => {
                    let mut eng = app.engine.lock();
                    let source = format!("supabase:{name}@{version}");
                    eng.predictor.load_bundle(bundle, &source);
                    let (features, auc) = (eng.predictor.feature_count(), eng.predictor.holdout_auc());
                    eng.audit_model(&name, features, &source, auc);
                    loaded.push(name);
                }
                Err(e) => tracing::warn!("bundle {name} did not parse: {e}"),
            },
            Err(e) => tracing::warn!("bundle {name} not fetched: {e}"),
        }
    }
    if !loaded.is_empty() {
        let detail = loaded.join(", ");
        app.engine.lock().push_event("ModelReload", &format!("supabase: {detail}"));
        tracing::info!(models = %detail, "models loaded from supabase");
    }
}

/// Ships the audit log to Supabase so the explanation of what the cache did outlives the
/// process that decided it.
///
/// Three things matter here. It batches, because one insert per decision would cost more
/// than the decisions. It re-queues on failure, because an audit log that silently drops
/// entries when the network hiccups is worse than no audit log — you would trust it. And it
/// never holds the engine lock across an await, because the cache must keep serving while
/// a hosted database is slow.
async fn ship_audit(app: Shared) {
    let sb = match app.supabase.as_ref() {
        Some(s) => s.clone(),
        None => return,
    };
    let mut ticker = tokio::time::interval(Duration::from_secs(2));
    let mut failures: u32 = 0;
    loop {
        ticker.tick().await;
        let batch = app.engine.lock().audit.drain_unshipped(200);
        if batch.is_empty() {
            continue;
        }
        let payload: Vec<Value> = batch
            .iter()
            .filter_map(|e| serde_json::to_value(e).ok())
            .collect();
        if let Err(e) = sb.push_audit(&payload).await {
            app.engine.lock().audit.requeue(batch);
            let text = e.to_string();
            // A missing table is not a transient failure and will never fix itself. Saying
            // so once and stopping is far more useful than the same line every ten seconds
            // for the rest of the run, which is noise a person learns to scroll past.
            if text.contains("PGRST205") || text.contains("Could not find the table") {
                tracing::warn!(
                    "audit shipping disabled: the aura_audit_log table does not exist. \
                     Run training/sql/006_audit_log.sql against the project. The in-memory \
                     log at GET /v1/audit keeps working either way."
                );
                return;
            }
            failures += 1;
            if failures % 6 == 1 {
                tracing::warn!("audit not shipped, re-queued: {e}");
            }
            // Back off rather than hammering a project that is refusing writes.
            tokio::time::sleep(Duration::from_secs(10)).await;
        } else {
            failures = 0;
        }
    }
}

/// Measures the real backend every few seconds rather than on every miss. One genuine
/// query per interval is enough to keep the cost model honest, and querying on every miss
/// would generate thousands of requests a second against a hosted database.
async fn probe_backend(app: Shared) {
    let sb = match app.supabase.as_ref() {
        Some(s) => s.clone(),
        None => return,
    };
    let mut ticker = tokio::time::interval(Duration::from_secs(3));
    let mut region = 1i32;
    let mut seen: Vec<f64> = Vec::new();
    loop {
        ticker.tick().await;
        region = region % 25 + 1;
        match sb.probe_analytics(region, 90).await {
            Ok((ms, bytes)) => {
                seen.push(ms);
                if seen.len() > 64 {
                    seen.remove(0);
                }
                let mut sorted = seen.clone();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let p95 = sorted[((sorted.len() - 1) as f64 * 0.95).round() as usize];
                let mean = seen.iter().sum::<f64>() / seen.len() as f64;
                let mut p = app.probe.lock();
                p.measured_ms = mean;
                p.p95_ms = p95;
                p.samples += 1;
                p.bytes = bytes;
            }
            Err(e) => {
                let mut p = app.probe.lock();
                p.failures += 1;
                if p.failures % 10 == 1 {
                    tracing::warn!("analytics probe failed: {e}");
                }
            }
        }
    }
}

async fn supabase_status(State(app): State<Shared>) -> Json<Value> {
    match app.supabase.as_ref() {
        Some(sb) => {
            let reachable = sb.health().await;
            let models = if reachable {
                sb.active_models().await.unwrap_or_default()
            } else {
                Vec::new()
            };
            Json(json!({
                "configured": true,
                "reachable": reachable,
                "project": sb.project(),
                "active_models": models,
                "role": "control plane only: model registry, benchmark results, events. \
The cache data path does not go through Postgres."
            }))
        }
        None => Json(json!({
            "configured": false,
            "reachable": false,
            "hint": "set SUPABASE_URL and SUPABASE_SERVICE_ROLE_SECRET_KEY, or put them in backend/.env"
        })),
    }
}

/// Keys arrive as opaque strings but the engine indexes by `u64`. A numeric key is used as
/// is so simulator and HTTP traffic land on the same slot; anything else is hashed.
fn key_id(key: &str) -> KeyId {
    if let Ok(n) = key.parse::<u64>() {
        return n;
    }
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in key.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x1000_0000_01b3);
    }
    h
}

fn now_ms(app: &Shared) -> f64 {
    let sim_time = app.sim.lock().as_ref().map(|s| s.generator.now_ms());
    match sim_time {
        Some(t) if t > 0.0 => t,
        _ => app.started.elapsed().as_secs_f64() * 1000.0,
    }
}

fn now_epoch_ms() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0)
}

fn round2(v: f64) -> f64 {
    (v * 100.0).round() / 100.0
}

fn round4(v: f64) -> f64 {
    (v * 10_000.0).round() / 10_000.0
}
