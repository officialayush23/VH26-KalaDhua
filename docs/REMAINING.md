# AURA — the plan, current

Rewritten 2026-09-04, late. Every line below was checked against the code today, not
carried over from the previous version of this file. Items marked DONE were verified by
reading the implementation or by a run whose output is in the repository history.

---

## 0. What is real and what is not

Say this exactly, in front of a judge. Overclaiming is the fastest way to lose.

| Piece | State |
|---|---|
| Decision engine, admission, eviction, policy blending | Real, exercised on every request |
| Cost accounting, Belady bound, 11-policy benchmark | Real, same stream through every policy |
| Cached bytes in memory | Real with `--real-values` (measured 1.19 GB resident for an 800 MB pool) |
| Trained model | Real: three GBDT bundles load from Supabase (`h10s`, `h60s`, `h600s`) |
| Online learning from outcomes | Real: `observe()` is wired to realised reuse |
| Correctness plane: invalidation, versions, single flight, refresh-ahead | Real |
| Traffic | Simulator or engine generator. Real application traffic works but is not the default |
| Multi-node | Not built. One hardcoded node |
| CDN tier | Declared in the contract, absent from the code. The frame omits it rather than lying |

---

## DONE since the last version of this file

- **Modern baselines.** The benchmark runs eleven policies: `lru`, `fifo`, `lfu`, `gds`,
  `gdsf`, `lecar`, `tinylfu`, `s3fifo`, `sieve`, `aura`, `aura_fixed`, against a Belady
  ceiling. AURA wins five of six scenarios on cost; W-TinyLFU takes `mixed_production` by
  1%. `aura_fixed` isolates what the capacity controller is worth on its own.
- **A trained model exists and loads.** Three GBDT reuse heads pulled from Supabase on
  boot. `reuse_linear_h60s` still fails to parse (`missing field bias`) and logs a warning.
- **`observe()` is wired.** The predictor learns from realised reuse, and calibration is
  split into kept and refused, because being overconfident about what you keep wastes
  memory while being overconfident about what you refuse forces rebuilds.
- **Refresh is real.** `refresh_candidates` no longer just resets a timestamp; the backlog
  carries enough context for the application to rebuild, and the reader is served while the
  rebuild happens behind it.
- **Invalidation counted separately from eviction.** `Removal::Invalidated` is constructed
  now, so "the cache was wrong" no longer hides inside "the cache was full".
- **Capacity controller stops shrinking an idle pool.** It waits for enough traffic to
  characterise the workload and for the pool to have been squeezed at least once.
- **Real L1 in the applications.** `apps/common/l1.py` is a byte-capped, TTL-bounded,
  per-process LRU holding values, eligibility decided by freshness class. The engine's
  recency window is no longer mislabelled as a cache.
- **Supabase, both planes.** Engine pulls bundles and publishes benchmark runs; the
  analytics application queries the seeded warehouse. Verified live today: project `Aura`,
  Postgres 17, `app_orders` 120k, `app_order_items` 120k, `app_customers` 40k,
  `app_products` 4k.
- **Deployment written.** Dockerfiles for engine, applications and dashboard; compose;
  `railway.json`; `render.yaml`; `docs/DEPLOY.md`. Four blockers found and fixed: wrong
  binary name in the engine image, missing `Cargo.lock` in the dependency layer, no `PORT`
  support anywhere, and no route to Supabase from a host without IPv6.

---

## OPEN, in the order I would work

### 1. The dashboard — the thing a judge actually looks at

Current state: `frontend/universe` has panels for cost, cache and latency, policy mixture,
capacity, a decision feed, per-application profiles, an event log and a benchmark runner.
It builds. What it does not have is the shape being asked for.

Target: a control-plane console, not a metrics wall.

- **Live decision log in plain English.** Every line a sentence a human reads without a
  legend: which key, what the cache did, why, and what it cost or saved. The data is
  already there — `/v1/explain/recent` returns structured facts per decision and
  `/v1/explain/{key}` adds `depends_on`, `stale`, `resident`. Nothing renders them.
- **Plug and watch.** An application appears in the console the moment it makes its first
  call, with its own cost shape, hit rate, saved spend and the objects it owns. No
  registration step, no configuration file.
- **Per-application configuration, live.** See item 2; the console is where it is edited.
- **Allocation over time.** A step chart of pool size rising into a spike and falling after.
  The history hook already records `capacity` per frame. It is the one chart people
  remember and it is not drawn.
- **The socket already carries everything.** `consistency`, `single_flight`, split
  calibration and `in_flight` are all in the telemetry frame and none of them are shown.

### 2. Per-application profiles the operator can change

Not built, and the specification is not written down yet. The engine's config is global:
one `[engine]` block, one price table, one set of horizon weights for every application.

Two different things could be meant by "custom configs" and they are not equally sound, so
this needs deciding before anything is built:

- **Objective and policy weights per application** — SLA weight, tail-risk lambda, horizon
  weights, admission margin, TTL and freshness class, size preference, and what the
  application is optimising for (spend, p95, or origin protection). Sound, and it is the
  real product surface: the same cache serving a GPU-heavy ranker and a database-heavy
  dashboard should not weigh them identically.
- **Weights on the 24 model features** — the features are inputs to a learned reuse
  probability. Scaling them by hand changes the input distribution the GBDT was fitted on,
  so the prediction stops meaning what its calibration says it means. If this is wanted it
  has to be an explicit blend against the model output, not a multiplier on its inputs.

### 3. Two applications, better ones

Recommendation and analytics are the two that matter: CPU-heavy versus database-heavy.
Content (large objects, bandwidth-dominated, one priced third-party family with a runtime
price spike) is behind the `content` compose profile.

- **Recommendation on real data.** The compute is already real: profile fold, 6k-item
  retrieval, candidate-by-catalogue cosine blocks, ensemble shards for the expensive tail.
  The data is synthetic. Decided: build the catalogue and the co-purchase matrix from
  Supabase (4k products, 120k orders, 40k customers), embeddings from a truncated SVD of
  the co-occurrence matrix at boot, user histories from `app_order_items`. Keeps one
  universe across both applications and adds no external dataset. Synthetic stays as the
  offline fallback.
- **Analytics, more real-world.** More query shapes, more filters, a realistic dashboard
  mix where a few queries are hot and the tail is wide and expensive.

### 4. Deployment, executed

Written but never run: there is no Docker daemon on the build machine, so no image has been
built even once. Render is the target because Railway blocks builds during the day.
`docs/DEPLOY.md` is the walkthrough.

### 5. Authentication

No auth on any route. A public engine URL means anyone can `PUT` into the cache, `DELETE`
keys or drive `/v1/sim/*`. Either keep the engine private and expose only the dashboard, or
put a shared token on the write and simulation routes.

### 6. Smaller, honest gaps

- **No trace producer.** `docs/CONTRACTS.md` §4 defines a trace format; nothing writes it.
  No standalone `aura-bench` binary, no `--emit-trace`. So training cannot use recorded
  traffic from this system.
- **Multi-node.** `/v1/nodes` returns one hardcoded node. No consistent hashing, no
  ScaleOut.
- **CDN tier.** In the contract, not in the code.
- **No host memory awareness.** `host_budget_bytes` is a constant, not a reading of free RAM.
- **`reuse_linear_h60s` export bug** in the trainer.
- **Comment pass on Python.** The Rust tree was trimmed; `training/` and `apps/` were not.
- `training/sql/superseded/` can go once you are sure.
