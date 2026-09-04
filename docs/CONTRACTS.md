# AURA — Interface Contracts (frozen)

Every component in this repository is built against this file. If a contract has to
change, change it here first, then change the implementations.

System name: **AURA** — *Adaptive Utility & Runtime-Aware cache*.
Repository: `VH26-KalaDhua`.

---

## 0. Component map

| Component | Language | Path | Talks to |
|---|---|---|---|
| `aura-core` | Rust (lib) | `engine/aura-core` | — |
| `aura-sim` | Rust (lib) | `engine/aura-sim` | `aura-core` |
| `aura-server` | Rust (bin `aura`) | `engine/aura-server` | `aura-core`, `aura-sim` |
| `aura-bench` | Rust (bin `aura-bench`) | `engine/aura-bench` | `aura-core`, `aura-sim` |
| example apps | Python 3.11 / FastAPI | `apps/*` | `aura-server` HTTP, Supabase |
| training | Python 3.11 | `training/` | trace files, Supabase |
| dashboard | React 19 JSX | `frontend/universe` | `aura-server` HTTP + WS |

Ports (local + compose):

| Service | Port |
|---|---|
| `aura-server` | 8080 |
| recommendation app | 8101 |
| analytics app | 8102 |
| content app | 8103 |
| dashboard (vite dev) | 5173 |
| prometheus | 9090 |
| grafana | 3000 |

---

## 1. Core value types (JSON wire form)

### 1.1 `CostVector`
The cost of regenerating one object once. Every field is optional; missing = 0.

```json
{
  "cpu_ms": 320.0,
  "gpu_ms": 80.0,
  "db_ms": 140.0,
  "network_bytes": 1854200,
  "api_calls": 1,
  "api_cost_usd": 0.002,
  "latency_ms": 412.0
}
```

### 1.2 `ObjectContext`
What an application tells AURA about an object. This is the *only* application-specific
information the cache ever sees — it never knows what a "recommendation" is.

```json
{
  "application": "recommendation",
  "object_type": "ranking_result",
  "size_bytes": 1854200,
  "ttl_ms": 300000,
  "sla_class": "high",
  "regen": { "cpu_ms": 320.0, "db_ms": 140.0, "gpu_ms": 80.0, "api_cost_usd": 0.002 }
}
```

`sla_class` ∈ `"critical" | "high" | "normal" | "low"`.

### 1.3 `Decision`
```json
{ "action": "Admit" | "Reject" | "Keep" | "Evict" | "Refresh", "reason_code": "…" }
```

---

## 2. `aura-server` HTTP API

Base URL `http://localhost:8080`. All responses `Cache-Control: no-store`.

### 2.1 Cache data plane

```
GET    /v1/cache/{key}?application=analytics
       200 -> { "hit": true,  "value": <base64|json>, "age_ms": 812, "layer": "L2", "latency_us": 41 }
       404 -> { "hit": false, "reason": "miss" }

PUT    /v1/cache/{key}
       body: { "value": <any>, "context": ObjectContext }
       200 -> { "admitted": true, "reason_code": "density_above_threshold",
                "evicted": ["media:41", "media:99"], "used_bytes": 402653184 }

DELETE /v1/cache/{key}                       -> { "removed": true }
POST   /v1/cache/{key}/refresh               -> { "queued": true }
POST   /v1/cache/batch/get  { "keys": [...] } -> { "results": { key: {...} } }
```

`GET` is the hot path: no model inference runs on it.

### 2.2 Explainability

```
GET /v1/explain/{key}
{
  "key": "recommendation:user:1842",
  "present": true,
  "action": "Keep",
  "reuse_probability": { "h10s": 0.41, "h60s": 0.87, "h600s": 0.93 },
  "economic_value_usd": 0.00231,
  "value_density": 4.72,
  "eviction_threshold": 1.11,
  "features": { "size_bytes": 1854200, "trend": 2.84, "…": 0 },
  "contributions": [ { "feature": "trend", "weight": 0.31 }, … ],
  "reasons": ["high predicted near-term reuse", "expensive GPU+DB regeneration",
              "value density 4.3x above current eviction threshold"],
  "predictor": "gbdt", "predictor_confidence": 0.78
}

GET /v1/explain/recent?limit=50   -> { "decisions": [ ExplainRecord, … ] }
```

### 2.3 Introspection / control

```
GET  /v1/stats                    -> StatsSnapshot (§3)
GET  /v1/workload                 -> { "regime": "FlashCrowd", "confidence": 0.82, "features": {…} }
GET  /v1/policy                   -> { "mixture": { "gdsf": 0.38, "cost_aware": 0.31, … },
                                       "bandit": "thompson", "ml_influence": 0.64 }
GET  /v1/capacity                 -> CapacityReport (§3.3)
GET  /v1/applications             -> { "profiles": [ ApplicationProfile, … ] }
GET  /v1/nodes                    -> { "nodes": [ { "id": "node-1", "capacity_bytes": …,
                                                    "used_bytes": …, "keys": …, "ring_share": 0.34 } ] }
POST /v1/capacity/mode            { "mode": "auto" | "manual", "bytes": 805306368 }
POST /v1/policy/override          { "policy": "gdsf" | "lru" | … | "aura" | "auto" }
POST /v1/model/reload             { "source": "supabase" | "file", "path": "…" }
GET  /metrics                     -> Prometheus text exposition
GET  /healthz                     -> { "ok": true, "version": "…", "uptime_s": … }
```

### 2.4 Simulation control (drives the demo)

```
GET  /v1/scenarios                -> { "scenarios": [ { "id": "flash_crowd", "name": …,
                                                        "description": …, "attacks": [...] } ] }
POST /v1/sim/start                { "scenario": "flash_crowd", "speed": 1.0, "seed": 42 }
POST /v1/sim/stop
POST /v1/sim/attack               { "attack": "Scan" | "FlashCrowd" | "PopularityShift" |
                                              "CostSpike" | "ExpensiveTail" | "HotKeyEmergence" |
                                              "HotKeyDecay" | "WorkingSetExplosion" | "MixedChaos",
                                    "duration_s": 30 }
POST /v1/sim/speed                { "speed": 4.0 }
GET  /v1/sim/status               -> { "running": true, "scenario": …, "virtual_time_s": …, "rps": … }
```

### 2.5 Benchmark (comparison against baselines, run in-process)

```
POST /v1/bench/run    { "scenario": "expensive_tail", "policies": ["lru","lfu","gdsf","aura"],
                        "capacity_bytes": 536870912, "requests": 200000 }
      -> { "run_id": "…" }
GET  /v1/bench/{run_id}   -> BenchmarkReport (§3.4)
GET  /v1/bench/latest     -> BenchmarkReport
```

---

## 3. Telemetry payloads

### 3.1 WebSocket `ws://localhost:8080/v1/live`

Server pushes a frame every 250 ms (configurable). Every frame is a complete snapshot —
the client never has to accumulate state.

```json
{
  "t": 1757000000123,
  "virtual_time_s": 184.25,
  "sim": { "running": true, "scenario": "flash_crowd", "speed": 1.0, "rps": 4820.0 },
  "traffic": { "rps": 4820.0, "rps_by_app": { "recommendation": 2010.0, "analytics": 1600.0, "content": 1210.0 } },
  "layers": {
    "cdn":     { "hits": 88213, "misses": 190112, "hit_rate": 0.317 },
    "l1":      { "hits": 40122, "misses": 149990, "hit_rate": 0.211 },
    "l2":      { "hits": 121444, "misses": 28546, "hit_rate": 0.810, "byte_hit_rate": 0.742 },
    "backend": { "requests": 28546, "inflight": 41, "saturation": 0.62 }
  },
  "latency": { "p50_ms": 3.1, "p95_ms": 41.7, "p99_ms": 128.4, "mean_ms": 9.2 },
  "cost": {
    "window_s": 3600,
    "backend_usd": 8.42, "cache_usd": 3.10, "sla_penalty_usd": 0.64, "total_usd": 12.16,
    "saved_vs_no_cache_usd": 41.90,
    "baselines": { "lru": 19.34, "lfu": 18.02, "gdsf": 15.91 },
    "savings_vs": { "lru": 0.371, "lfu": 0.325, "gdsf": 0.236 },
    "burn_rate_usd_per_hour": 12.16
  },
  "capacity": {
    "logical_bytes": 536870912, "used_bytes": 501234100,
    "recommended_bytes": 805306368, "host_budget_bytes": 2147483648,
    "pressure": 0.93, "nodes": 1, "last_action": "ScaleUp",
    "mrc": [ { "bytes": 268435456, "hit_rate": 0.61 }, { "bytes": 536870912, "hit_rate": 0.72 } ]
  },
  "workload": {
    "regime": "FlashCrowd", "confidence": 0.82,
    "features": { "burstiness": 3.42, "entropy": 6.11, "working_set_growth": 0.19,
                  "reuse_distance_p50": 812.0, "popularity_shift": 0.06, "scan_score": 0.02 }
  },
  "policy": {
    "mixture": { "lru": 0.05, "lfu": 0.04, "gdsf": 0.38, "tiny_lfu": 0.12,
                 "cost_aware": 0.31, "trend_aware": 0.10 },
    "ml_influence": 0.64, "predictor_confidence": 0.78, "bandit_regret": 0.031
  },
  "engine": {
    "admissions": 91223, "admissions_rejected": 41003, "evictions": 88110,
    "refreshes": 2044, "inference_calls": 12044, "decision_overhead_us_p50": 2.9
  },
  "applications": [ ApplicationProfile, … ],
  "events": [ { "t": …, "kind": "ScaleUp" | "PolicyShift" | "AttackStart" | "Eviction" | "Refresh",
                "detail": "512 MB -> 768 MB (net +$4.10/hr)" } ],
  "recent_decisions": [ ExplainRecord, … ]
}
```

Client→server messages on the same socket:

```json
{ "type": "subscribe", "channels": ["decisions", "events"] }
{ "type": "attack", "attack": "FlashCrowd", "duration_s": 30 }
{ "type": "speed", "speed": 4.0 }
```

### 3.2 `ApplicationProfile`

```json
{
  "application": "analytics",
  "requests": 402113, "hit_rate": 0.78,
  "avg_object_bytes": 41000, "p95_object_bytes": 380000,
  "reuse_interval_p50_ms": 12000,
  "regen_p50_ms": 240.0, "regen_p95_ms": 870.0,
  "cost_profile": "db_heavy",
  "allocated_bytes": 1503238553,
  "preferred_policies": { "gdsf": 0.18, "cost_aware": 0.42, "tiny_lfu": 0.25, "trend_aware": 0.15 },
  "traffic_shape": "bursty"
}
```

### 3.3 `CapacityReport`

```json
{
  "mode": "auto",
  "logical_bytes": 536870912,
  "recommended_bytes": 805306368,
  "host_budget_bytes": 2147483648,
  "host_available_bytes": 134217728,
  "decision": "ScaleUp" | "ScaleOut" | "Hold" | "ScaleDown" | "ScaleIn",
  "marginal": [
    { "from_bytes": 536870912, "to_bytes": 805306368,
      "delta_hit_rate": 0.066, "backend_savings_usd_hr": 7.20,
      "cache_cost_usd_hr": 1.40, "net_usd_hr": 5.80, "verdict": "profitable" }
  ],
  "reason": "marginal ROI 4.1x above threshold; host has 128 MB free -> resize in place",
  "nodes": 1, "provider": "local"
}
```

### 3.4 `BenchmarkReport`

```json
{
  "run_id": "…", "scenario": "expensive_tail", "requests": 200000,
  "capacity_bytes": 536870912, "seed": 42,
  "rows": [
    { "policy": "lru",   "object_hit_rate": 0.612, "byte_hit_rate": 0.401,
      "p95_latency_ms": 88.1, "backend_requests": 77600, "total_cost_usd": 19.34,
      "regen_cost_usd": 18.70, "sla_penalty_usd": 0.64,
      "decision_overhead_us_p50": 0.2, "memory_overhead_bytes": 1048576,
      "adaptation_time_s": null },
    { "policy": "aura",  "object_hit_rate": 0.741, "…": 0 }
  ],
  "belady_upper_bound": { "object_hit_rate": 0.812, "total_cost_usd": 11.02 },
  "winner": "aura",
  "improvement_vs": { "lru": 0.371, "lfu": 0.325, "gdsf": 0.236 }
}
```

---

## 4. Trace format (Rust sim writes, Python training reads)

CSV, gzipped, one row per request. Written by `aura-bench --emit-trace` and
`aura sim --emit-trace`. Column order is fixed.

```
ts_ms,key_id,application,object_type,size_bytes,ttl_ms,sla_class,
cpu_ms,gpu_ms,db_ms,network_bytes,api_calls,api_cost_usd,regen_latency_ms,
scenario,regime
```

`key_id` is a `u64`. `regime` is the ground-truth regime label emitted by the generator
(training uses it only for stratified splits, never as a feature).

Companion `*.meta.json`:

```json
{ "scenario": "mixed_production", "seed": 42, "requests": 10000000,
  "unique_keys": 400000, "duration_s": 3600, "generator_version": 1,
  "applications": ["recommendation","analytics","content"] }
```

---

## 5. Model artifact contract

Training produces **one artifact bundle** per model. `aura-core` loads it without
any ML runtime dependency (pure Rust tree walker), and optionally via ONNX.

`model_bundle.json`:

```json
{
  "schema_version": 1,
  "name": "reuse_gbdt",
  "kind": "lightgbm_gbdt" | "linear_logistic",
  "horizon_ms": 60000,
  "version": "2026-09-04T11:20:00Z",
  "git_sha": "…",
  "feature_names": ["log_age_ms", "log_inter_arrival_ms", "freq_1m", "freq_5m", "freq_1h",
                    "ewma_fast", "ewma_slow", "trend", "acceleration", "log_size_bytes",
                    "log_regen_p50_ms", "cost_variance_ratio", "regen_cost_usd",
                    "ttl_remaining_frac", "cache_pressure", "app_id"],
  "normalization": { "mean": [...], "scale": [...] },
  "objective": "binary",
  "sigmoid_output": true,
  "trees": [
    { "split_feature": [3, 9, -1], "threshold": [12.0, 14.2, 0.0],
      "left": [1, -1, 0], "right": [2, -2, 0],
      "leaf_value": [0.41, -0.22, 0.08], "decision_type": [2, 2, 0] }
  ],
  "linear_weights": null,
  "metrics": { "auc": 0.884, "pr_auc": 0.71, "logloss": 0.402, "n_train": 8000000 }
}
```

Encoding rule for `trees` (matches LightGBM's dumped model):
- Node `i` has `split_feature[i]`, `threshold[i]`.
- `left[i]`/`right[i]`: **positive** = internal node index, **negative** = leaf index
  `-(v) - 1` into `leaf_value`. `0` in a leaf row is ignored.
- Default direction on a missing/NaN value is left.

Three bundles are produced: `reuse_gbdt_h10s`, `reuse_gbdt_h60s`, `reuse_gbdt_h600s`,
plus `reuse_linear_h60s` used for the online/cold-start path.

### 5.1 Supabase storage layout

Bucket `aura-models` (private):

```
aura-models/
  reuse_gbdt_h60s/2026-09-04T11-20-00Z/model_bundle.json
  reuse_gbdt_h60s/2026-09-04T11-20-00Z/model.onnx
  reuse_gbdt_h60s/latest -> pointer row in Postgres, not a symlink
```

Postgres table `aura_models` is the source of truth for which version is active
(§6). The server calls `POST /v1/model/reload` or reloads on boot.

---

## 6. Supabase Postgres schema (namespace `public`)

```sql
aura_models(id uuid pk, name text, kind text, horizon_ms int, version text,
            storage_path text, onnx_path text, metrics jsonb, feature_names jsonb,
            is_active bool, created_at timestamptz)

aura_benchmark_runs(id uuid pk, run_id text unique, scenario text, seed int,
                    capacity_bytes bigint, requests bigint, engine_version text,
                    created_at timestamptz, summary jsonb)

aura_benchmark_results(id bigserial pk, run_id text references …, policy text,
                       object_hit_rate double precision, byte_hit_rate double precision,
                       p95_latency_ms double precision, backend_requests bigint,
                       total_cost_usd double precision, regen_cost_usd double precision,
                       sla_penalty_usd double precision,
                       decision_overhead_us double precision, extra jsonb)

aura_traces(id uuid pk, name text, scenario text, storage_path text, rows bigint,
            unique_keys bigint, bytes bigint, meta jsonb, created_at timestamptz)

aura_events(id bigserial pk, ts timestamptz, kind text, detail jsonb)
```

Analytics-workload tables (the analytics example app runs real SQL against these):

```sql
app_regions(id serial pk, name text, country text)
app_products(id serial pk, name text, category text, unit_price numeric)
app_orders(id bigserial pk, region_id int, product_id int, qty int,
           amount numeric, created_at timestamptz)
-- index: (region_id, created_at), (product_id, created_at)
```

---

## 7. Example-application contract

Each app exposes:

```
GET  /health
GET  /work/{id}?fresh=true|false     -> the actual expensive endpoint, goes through AURA
GET  /stats                          -> { "requests": …, "cache_hits": …, "regens": …,
                                           "avg_regen_ms": …, "cost_usd": … }
POST /load  { "rps": 200, "duration_s": 60, "pattern": "zipf"|"scan"|"burst" }
```

They use `apps/common/aura_client.py`:

```python
client = AuraClient(base_url, application="analytics")
value = client.get_or_regen(key, context=ObjectContext(...), regen=lambda: expensive())
```

`get_or_regen` does: `GET /v1/cache/{key}` → on miss run `regen()`, measure the real
cost vector, `PUT /v1/cache/{key}` with it. That measured cost — not a guess — is what
AURA optimizes against.

---

## 8. Configuration

`engine/config/default.toml`, overridable by env `AURA__SECTION__KEY`.

```toml
[cache]
capacity_bytes = 536870912
shards = 16
[cache.l1]
capacity_bytes = 33554432
[cdn]
enabled = true
ttl_ms = 60000
capacity_bytes = 134217728

[pricing]                 # USD, used by the economic model
cpu_ms_usd = 0.0000000116
gpu_ms_usd = 0.000000255
db_ms_usd  = 0.0000000320
network_gb_usd = 0.09
cache_gb_hour_usd = 0.0225
sla_penalty_per_ms_over_slo_usd = 0.0000004
slo_p95_ms = 150.0

[engine]
candidate_sample = 32
controller_tick_ms = 100
admission_margin = 1.10
refresh_ttl_threshold = 0.15
ml_confidence_floor = 0.20

[capacity]
mode = "auto"
step_bytes = 268435456
min_bytes = 134217728
max_bytes = 4294967296
host_budget_bytes = 2147483648
roi_threshold = 1.25

[predictor]
kind = "gbdt"             # heuristic | linear | gbdt | onnx
bundle_path = "models/reuse_gbdt_h60s.json"
supabase_autoload = true

[bandit]
kind = "thompson"         # thompson | epsilon_greedy | exp3
exploration = 0.08
```

---

## 9. Style rules for this repository

- No generated-by / AI-authored markers anywhere: comments, commit messages, docs, or metadata.
- Rust: `#![forbid(unsafe_code)]` in every crate, `cargo fmt`, no `unwrap()` outside tests
  and `main()`.
- Python: type hints on public functions, `ruff`-clean, no notebook outputs committed.
- Frontend: `.jsx` only, no TypeScript.
- Anything that reads a secret reads it from the environment. `.env` is never committed.
