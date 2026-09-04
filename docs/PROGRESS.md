# AURA — build progress

Status as of 2026-09-04. Read `docs/CONTRACTS.md` first; it is the frozen interface and
everything here is built against it.

## Running it

Three processes. Only the first two are needed for the demo.

```bash
# 1. engine  (terminal 1)
cd engine
cargo run --release -p aura-server -- --scenario mixed_production
# listens on :8080, starts generating traffic immediately

# 2. dashboard  (terminal 2)
cd frontend/universe
npm run dev
# http://localhost:5173

# 3. example apps  (optional, terminal 3)
cd apps
pip install -r requirements.txt
python -m driver.run_universe
```

The dashboard finds the engine at `http://localhost:8080`. Override with
`VITE_AURA_URL` in `frontend/universe/.env.local` if you move it.

### Supabase

Run these two in the Supabase SQL editor, in order:

1. `training/sql/003_supabase_schema.sql` — tables, indexes, constraints, RLS.
2. `training/sql/004_supabase_seed.sql` — 120k orders / 360k order lines.

Both are idempotent. The seed takes about 20 seconds locally and longer on Supabase.
Credentials stay in `backend/.env`, which is gitignored and was not used by any tool here.

### Training

`training/notebooks/aura_training_colab.ipynb` trains the reuse models and pushes the
bundle to Supabase. It needs `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_SECRET_KEY` set as
Colab secrets. Output bundles land in `engine/models/` for the server to load with
`--models` or `POST /v1/model/reload`.

## Done

### Engine — `engine/`
- `aura-core` (pre-existing): types, config, feature builder, sketches, seeded RNG.
- `aura-sim` (new): six scenarios, nine attack shapes, seeded request generator with
  per-application cost signatures. Same generator serves the live sim and the offline
  benchmark, so both meet the identical stream.
- `aura-server` (new, binary `aura`): store with L1 admission window, decision engine,
  predictor (bundle loader plus online logistic for cold start), Thompson-sampling policy
  mixture, capacity controller, benchmark harness, full HTTP surface and the `/v1/live`
  WebSocket.

Every route in `CONTRACTS.md` §2 is implemented. `cargo check --workspace` is clean.

Verified by running it: 25k requests in the first six seconds, decision overhead 3.2 µs
p50 on the write path, zero on the read path.

### Benchmark result — `expensive_tail`, 80k requests, 128 MB, seed 42

| policy | object hit | byte hit | cost USD | p95 ms |
|---|---|---|---|---|
| lru | 0.2079 | 0.1959 | 133.48 | 6546 |
| lfu | 0.2470 | 0.2275 | 127.68 | 6401 |
| gdsf | 0.1991 | 0.2827 | 119.85 | 6358 |
| cost_aware | 0.1027 | 0.2260 | 130.99 | 6592 |
| **aura** | **0.2788** | **0.3025** | **117.49** | **6273** |

Belady ceiling 0.3953 / $69.19. AURA is 12.0% cheaper than LRU, 8.0% than LFU, 2.0% than
GDSF. On `mixed_production` it beats LRU by 9.9% and LFU by 3.0% and trails GDSF by 2.6%,
so the advantage is real but scenario-dependent — say that rather than overclaiming.

### Database — `training/sql/`
`003_supabase_schema.sql` and `004_supabase_seed.sql`. Third normal form throughout:
lookup tables for closed vocabularies, `aura_model_metrics` and `aura_model_features` as
child tables rather than JSON columns, `aura_benchmark_results` keyed on (run, policy)
rather than a wide run row, orders split from order lines. Verified against Postgres 16:
schema applies cleanly, is idempotent on re-run, seed produces the expected row counts.

### Supabase connection from the engine — `engine/aura-server/src/supabase.rs`
Control plane only, and deliberately not on the request path. Pulls the active model
bundle from Storage on boot and on `POST /v1/model/reload {"source":"supabase"}`,
publishes benchmark runs and per-policy rows after `POST /v1/bench/run`, appends events.
Reads `backend/.env` by walking up from the working directory, so it works whether the
engine is started from `engine/` or the repository root. `GET /v1/supabase` reports
configured / reachable / active models, and the dashboard shows it.

Verified: env file discovered from a nested directory, an unreachable project logs a
warning and changes nothing else, the cache keeps serving at a 66% hit rate and the
benchmark still returns while publishing fails, and an unconfigured engine reports
cleanly rather than erroring.

Schema note: the first version of `003` was more decomposed than
`training/aura_train/supabase_io.py` speaks, so every Colab push would have failed on a
missing column. The base tables now match the writer exactly, with 3NF preserved through
natural text foreign keys, and `metrics`/`feature_names` are fanned into
`aura_model_metrics`/`aura_model_features` by trigger. Verified by replaying every real
payload from `supabase_io.py` against Postgres 16 and confirming four bad-data cases are
rejected.

### Dashboard — `frontend/universe/`
Converted from TypeScript to JSX per `CONTRACTS.md` §9. GSAP animates the metric tickers,
policy mixture bars and the decision feed. Panels: cost against baselines, cache and
latency, policy mixture with regime, capacity control with the miss-ratio curve, live
decision explanations, per-application profiles, event log, scenario and attack controls,
and an in-page benchmark runner. `npm run build` passes.

## Three defects found and fixed

Worth knowing about, because each one produced a plausible-looking wrong number.

1. **Baseline shadows were fed only AURA's misses.** Their hit rates and costs were
   therefore fiction, and every savings figure derived from them was meaningless. They now
   see the full request stream. This is the one that mattered most: before the fix AURA
   appeared to *lose* to every baseline.
2. **Regime detector classified ordinary Zipf traffic as a scan.** The old score reduced
   algebraically to singles/total, which any wide window produces. Replaced with a novelty
   rate. Before the fix the engine was rejecting most admissions in normal traffic.
3. **Admission compared each object against a running average of arrivals.** That bar is
   self-referential: it rejects the bottom half of the stream regardless of value, and
   leaves capacity unused. It now compares against the object that would actually be
   evicted.

## Not done

In rough priority order.

- **`aura-bench` binary.** The benchmark runs in-process through `POST /v1/bench/run`;
  there is no standalone CLI and no `--emit-trace` yet, so the trace format in
  `CONTRACTS.md` §4 has no producer. The training pipeline currently depends on
  `training/aura_train/synthetic.py` instead of real traces.
- **Model bundles.** No trained bundle exists yet, so the server runs on the online
  logistic predictor. Everything works, but `predictor` reports `heuristic` rather than
  `gbdt` and the explainability panel shows online weights. Run the Colab notebook and
  drop the output in `engine/models/`.
- **The analytics app is not wired to Supabase in the live path.** `apps/analytics/db.py`
  and `queries.py` exist; nothing calls them from the running demo, so the miss cost is
  currently the simulator's synthetic cost rather than a real query. This is the
  cost-story upgrade: point the analytics app at Supabase and let a real 200-400 ms
  aggregate be the regeneration cost the cache learns. Note the engine's own Supabase
  connection is done — this item is about the application services, which is the correct
  place for it since the cache data path must not go through Postgres.
- **`deploy/` is empty.** No `docker-compose.yml`, no Dockerfiles, no Railway config.
- **CDN layer** in the telemetry contract is declared but not implemented; the frame
  omits it rather than reporting zeros.
- **Multi-node.** `/v1/nodes` returns a single hardcoded node. Consistent hashing and
  ScaleOut are described in the contract but not built.
- **Comment pass on Python.** The Rust tree has been trimmed (118 comment lines removed);
  `training/` and `apps/` have not.
- `training/sql/superseded/` holds `001_schema.sql` and `002_seed_analytics.sql`, which
  `003`/`004` replace. Kept for reference; delete when you are sure.

## Documentation

- `docs/CONTRACTS.md` — the frozen interface. Change it before changing code.
- `docs/ARCHITECTURE.md` — what every file does, written for someone who does not read Rust.
- `docs/RUNBOOK.md` — exact run steps, the Supabase order, the Colab walkthrough, the demo
  script and the failure modes.
- `docs/PROGRESS.md` — this file.

## Conventions being followed

No generated-by markers anywhere: no tool comments, no Co-Authored-By trailers, no
attribution of any kind in commits or metadata. Commits are local only; nothing has been pushed. `.env` is
never committed.
