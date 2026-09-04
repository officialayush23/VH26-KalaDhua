#![forbid(unsafe_code)]

mod bench;
mod capacity;
mod engine;
mod policy;
mod predictor;
mod store;

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
use crate::engine::Engine;
use crate::policy::Policy;

#[derive(Parser, Debug)]
#[command(name = "aura", version, about = "AURA cache server")]
struct Args {
    #[arg(long, default_value = "0.0.0.0:8080")]
    bind: String,
    #[arg(long)]
    config: Option<PathBuf>,
    #[arg(long, default_value = "models")]
    models: PathBuf,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    /// Start the simulator immediately. Without this the server is a plain cache and only
    /// serves what applications put into it.
    #[arg(long)]
    scenario: Option<String>,
}

struct Sim {
    generator: Generator,
    running: bool,
    speed: f64,
}

struct App {
    engine: Mutex<Engine>,
    capacity: Mutex<CapacityController>,
    sim: Mutex<Option<Sim>>,
    cfg: Config,
    bench: Mutex<Option<BenchmarkReport>>,
    tx: broadcast::Sender<String>,
    started: std::time::Instant,
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
    let cfg = Config::load(args.config.as_deref());

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
    let app = Arc::new(App {
        engine: Mutex::new(engine),
        capacity: Mutex::new(CapacityController::new(&cfg)),
        sim: Mutex::new(sim),
        cfg,
        bench: Mutex::new(None),
        tx,
        started: std::time::Instant::now(),
    });

    tokio::spawn(drive(app.clone()));

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
        .route("/v1/explain/recent", get(explain_recent))
        .route("/v1/explain/:key", get(explain_key))
        .route("/v1/stats", get(stats))
        .route("/v1/workload", get(workload))
        .route("/v1/policy", get(policy_get))
        .route("/v1/policy/override", post(policy_override))
        .route("/v1/capacity", get(capacity_get))
        .route("/v1/capacity/mode", post(capacity_mode))
        .route("/v1/applications", get(applications))
        .route("/v1/nodes", get(nodes))
        .route("/v1/model/reload", post(model_reload))
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
        .layer(cors)
        .with_state(app);

    let addr: SocketAddr = args.bind.parse()?;
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
            let mut eng = app.engine.lock();
            for r in batch {
                if eng.get(r.key_id, &r.context.application, r.ts_ms).is_none() {
                    let mut measured = r.context.regen;
                    measured.latency_ms = r.regen_latency_ms;
                    eng.put(r.key_id, Value::Null, &r.context, measured, r.ts_ms);
                }
            }
            eng.recompute_workload();
            let to_refresh = eng.refresh_candidates(4);
            let refreshed_at = eng.now_ms;
            for k in to_refresh {
                if let Some(e) = eng.store.entry_mut(k) {
                    e.inserted_ms = refreshed_at;
                }
            }
        }

        {
            let mut eng = app.engine.lock();
            let mut cap = app.capacity.lock();
            cap.maybe_apply(&mut eng, &app.cfg);
        }

        let frame = build_frame(&app);
        let _ = app.tx.send(frame.to_string());
    }
}

fn build_frame(app: &Shared) -> Value {
    let eng = app.engine.lock();
    let cap = app.capacity.lock();
    let report = cap.report(&eng, &app.cfg);
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
        "applications": apps,
        "events": eng.events.iter().rev().take(20).collect::<Vec<_>>(),
        "recent_decisions": eng.explains.iter().rev().take(12).collect::<Vec<_>>()
    })
}

fn cost_profile(app: &str) -> &'static str {
    match app {
        "analytics" => "db_heavy",
        "content" => "gpu_heavy",
        "recommendation" => "mixed",
        _ => "unknown",
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

async fn cache_get(
    State(app): State<Shared>,
    Path(key): Path<String>,
    Query(q): Query<AppQuery>,
) -> impl IntoResponse {
    let id = key_id(&key);
    let now = now_ms(&app);
    let mut eng = app.engine.lock();
    match eng.get(id, q.application.as_deref().unwrap_or("default"), now) {
        Some(e) => (
            StatusCode::OK,
            Json(json!({
                "hit": true,
                "value": e.value,
                "age_ms": round2(e.age_ms(now)),
                "layer": "L2",
                "latency_us": 40
            })),
        ),
        None => (
            StatusCode::NOT_FOUND,
            Json(json!({ "hit": false, "reason": "miss" })),
        ),
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
    Json(body): Json<PutBody>,
) -> Json<Value> {
    let id = key_id(&key);
    let now = now_ms(&app);
    let mut eng = app.engine.lock();
    let measured = body.measured.unwrap_or(body.context.regen);
    let (decision, evicted) = eng.put(id, body.value, &body.context, measured, now);
    Json(json!({
        "admitted": decision.action == aura_core::types::Action::Admit,
        "reason_code": decision.reason_code,
        "evicted": evicted.iter().map(|k| k.to_string()).collect::<Vec<_>>(),
        "used_bytes": eng.store.used_bytes()
    }))
}

async fn cache_delete(State(app): State<Shared>, Path(key): Path<String>) -> Json<Value> {
    let mut eng = app.engine.lock();
    let removed = eng.store.remove(key_id(&key)).is_some();
    Json(json!({ "removed": removed }))
}

async fn cache_refresh(State(app): State<Shared>, Path(key): Path<String>) -> Json<Value> {
    let now = now_ms(&app);
    let mut eng = app.engine.lock();
    let id = key_id(&key);
    let queued = match eng.store.entry_mut(id) {
        Some(e) => {
            e.inserted_ms = now;
            true
        }
        None => false,
    };
    if queued {
        eng.refreshes += 1;
    }
    Json(json!({ "queued": queued }))
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

async fn explain_recent(State(app): State<Shared>, Query(q): Query<LimitQuery>) -> Json<Value> {
    let eng = app.engine.lock();
    let decisions: Vec<_> = eng.explains.iter().rev().take(q.limit).collect();
    Json(json!({ "decisions": decisions }))
}

async fn explain_key(State(app): State<Shared>, Path(key): Path<String>) -> impl IntoResponse {
    let eng = app.engine.lock();
    let id = key_id(&key);
    match eng.explain_key(id) {
        Some(r) => (StatusCode::OK, Json(json!(r))),
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
    path: Option<String>,
}

async fn model_reload(State(app): State<Shared>, Json(body): Json<ModelReload>) -> Json<Value> {
    let dir = PathBuf::from(body.path.unwrap_or_else(|| "models".into()));
    let mut eng = app.engine.lock();
    match predictor::Predictor::load_dir(&dir, app.cfg.predictor.online_lr) {
        Ok(p) => {
            let horizons: Vec<String> = p.loaded_horizons().iter().map(|s| s.to_string()).collect();
            let kind = p.kind().as_str();
            eng.predictor = p;
            eng.push_event("ModelReload", &format!("{kind} from {}", dir.display()));
            Json(json!({ "ok": true, "kind": kind, "horizons": horizons }))
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
fn default_policies() -> Vec<String> {
    vec!["lru".into(), "lfu".into(), "gdsf".into(), "aura".into()]
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
