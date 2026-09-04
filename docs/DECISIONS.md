# AURA — decisions, status and what is left

The single source of truth for what has been settled, what is actually built, and what is
still open. Update this file whenever a decision changes or a task moves.

Companion documents: `CONTRACTS.md` (frozen interfaces), `SCORING_MATH.md` (the arithmetic),
`ALGORITHMS.md` (the shapes), `REMAINING.md` (the earlier task list, superseded by §4 here).

Last updated: 2026-09-04.

---

## 1. Decisions that are final

Do not relitigate these without changing this file first.

### Product

| # | Decision | Why |
|---|---|---|
| D1 | The system is an **application-adaptive economic cache**, not a better Redis | Redis is a good general cache. Our claim is narrower and defensible: we learn what *this* application's objects are worth. |
| D2 | Two example applications: **recommendation** (compute-heavy) and **analytics** (read-heavy) | Deliberately opposite economics. One cache, one configuration, two workloads whose optimal policies differ — that divergence *is* the experiment. |
| D3 | The cache never learns what an application *means* | It sees size, measured cost, freshness class, access shape. This is what makes it plug into a third application without changes. |
| D4 | Integration is **SDK over HTTP**; Redis wire protocol is a stretch goal only | RESP compatibility throws away the cost vector, which is where all the value comes from. |

### Algorithm

| # | Decision | Why |
|---|---|---|
| D5 | The model predicts **only reuse probability**, never a cache decision | A model trained to output "evict" imitates whatever policy made its training data, and is unexplainable when wrong. |
| D6 | **Three horizons**: 10 s, 60 s, 600 s | "Will it be reused" is meaningless without a clock. Different horizons drive different decisions. |
| D7 | Economics are **deterministic**, downstream of the model | `value_density = cost × expected_reuses × tail_risk / holding_cost`. Changing a price must never require a retrain. |
| D8 | **12 portable features only** — cost features and `app_id` are removed from the model input | `app_id` is a categorical that cannot generalise to an unseen application; absolute cost and latency shift between platforms and the failure is silent. Cost stays in the deterministic layer. **This supersedes the 16-feature vector in `CONTRACTS.md` §5.** |
| D9 | Thompson sampling over **6 policy experts**, with posterior decay | No policy chosen at build time. Decay is what keeps it adapting after hour one instead of freezing. |
| D10 | Model influence floors at 20% and never fully overrides the experts | Cold start must work, and a bad model must not be able to take the cache down. |
| D11 | Candidate sampling of 32, not a sorted heap | A heap is O(log n) on every access. Sampling 32 gives a victim worse than 97% of residents and costs nothing until an eviction is needed. |
| D12 | **Correctness overrides optimisation, always** | Validity is checked before value. No score is high enough to justify serving a wrong price. |

### Learning

| # | Decision | Why |
|---|---|---|
| D13 | Rewards and labels are **delayed and realised**, never derived from the decision itself | Crediting at decision time grades its own homework. Settle at 60 s, retire at 600 s. |
| D14 | A **miss counts as reuse** for labelling purposes | The label answers "was this object wanted again". Counting only hits teaches the model that whatever the cache kept was right. |
| D15 | Training data comes from the **request stream**, not from cache decisions | Otherwise the model learns "objects the old policy cached tend to be reused", which is circular. |
| D16 | Splits are **temporal and by application**, never random | A random split leaks a key's future into training. The strong claim is held-out *application*, not held-out rows. |
| D17 | The real evaluation is **cache replay against baselines**, not AUC | A predictor with AUC 0.91 that does not lower total cost is not useful. |

### Consistency

| # | Decision | Why |
|---|---|---|
| D18 | Four mechanisms: **TTL, explicit invalidation, versioned namespaces, event-driven** | Each covers a case the others miss. TTL alone is not correctness. |
| D19 | **Dependency tags** (`row:product:1292`) with an inverted index | One write invalidates every derived object, however many, whoever built it. |
| D20 | A model or schema redeploy **bumps a namespace version**; it does not flush | Flushing sends the whole miss stream at the origin — the cache causing the outage it exists to prevent. |
| D21 | **Soft TTL** window before hard expiry, with rebuild behind a stale serve | Stops a popular key becoming a stampede the moment it expires. |
| D22 | Eviction, invalidation and expiry are **three separate events** in telemetry | Merging them hides the only one that indicates a bug. |
| D23 | The dependency graph is **not** a model feature | Correctness and optimisation must not be mixed. |

### Personalisation

| # | Decision | Why |
|---|---|---|
| D24 | Personalised keys are **epoch-versioned**: `rec:user:1842:v{model}:{epoch}` | A click changes the key, so the stale ranking is never requested again. Invalidation by construction. |
| D25 | A small fraction of requests **bypass the cache deliberately** | Keeps the RL learner fed with fresh impressions and gives a live control group measuring what the cache costs in recommendation quality. |
| D26 | The RL/hybrid recommender lives **in the application**, not in the cache | AURA must not know what SAC or PPO is. |
| D27 | Cache the **candidate set**, re-rank cheaply per request | Keeps most of the compute saving with almost none of the staleness. |

### Engineering

| # | Decision | Why |
|---|---|---|
| D28 | Rust runtime, Python for training only | The hot path is hash maps, counters and small vector maths. |
| D29 | The engine **never calls Supabase on the request path** | A Python sidecar pulls model bundles and pushes results. Control plane only. |
| D30 | `--real-values` becomes the **default** | The memory-scaling demo must run in the mode where memory is real. Measured 1.19 GB resident for an 800 MB pool. |
| D31 | Single engine replica until consistent hashing exists | Two replicas mean two independent pools and every benchmark number becomes meaningless. |
| D32 | No LLM anywhere in the decision path | Optional narration of structured reasons only. |
| D33 | One owner per directory | Two sessions writing the same tree have already overwritten each other once. |
| D34 | **Validity is checked before value** on every read, and the freshness check consults neither the model nor the economics | A cache that serves ₹40 after the database says ₹20 is not fast, it is wrong. No score is high enough to buy a wrong answer. |
| D35 | A refresh **rebuilds through the origin and is charged to the backend ledger** | Refreshing ahead of expiry is cheaper than an expiry storm, but it is not free. A controller told it was free would refresh everything, constantly. |
| D36 | Single-flight is a **lease handed to the caller**, not a rebuild the cache performs | Only the application knows how to build an object. The cache can still say who is allowed to, which is what turns a thousand simultaneous misses into one origin call. |
| D37 | Baselines run their **real implementations**, never a shared `rank()` function | W-TinyLFU, S3-FIFO and SIEVE are defined by their structure, not by a scoring expression. Beating a four-tuple caricature of them proves nothing. |

---

## 2. Architecture as decided

```text
                        USERS  (simulated: sessions, clicks, purchases)
                          │
                          ▼
              ┌────────────────────────┐
              │ APPLICATION            │
              │  recommendation        │
              │  analytics             │
              │                        │
              │  L1 in-process cache   │   <- SDK, small, seconds of TTL
              └───────────┬────────────┘
                          │ miss
                          ▼
         ╔════════════════════════════════════╗
         ║            AURA  L2                ║
         ║                                    ║
         ║  1. VALID?  ── no ──► invalidate    ║   correctness first
         ║       │ yes                        ║
         ║  2. lookup / admission             ║
         ║  3. EWMA + count-min + cost model  ║
         ║  4. reuse predictor (12 features)  ║
         ║  5. policy experts                 ║
         ║  6. Thompson selector              ║
         ║  7. economic utility               ║
         ║       ├── ADMIT / KEEP / EVICT     ║
         ║       └── REFRESH                  ║
         ║  8. capacity loop -> SCALE / HOLD  ║
         ╚═══════════════╤════════════════════╝
                         │ miss
                         ▼
              ┌────────────────────────┐
              │ ORIGIN                 │
              │  ranking model (GPU)   │
              │  Postgres / Supabase   │
              │  external priced API   │
              └───────────┬────────────┘
                          │ measured cost vector
                          ▼
                        AURA

  ── CONSISTENCY PLANE ──          ── FEEDBACK PLANE ──        ── INFRA PLANE ──
  Postgres NOTIFY                  decision journal            logical resize
  dependency tags                  settle at 60 s              scale out / in
  namespace versions               retire at 600 s             local / Railway / AWS
  soft + hard TTL                  bandit + online model
```

---

## 3. What is real right now

Verified by reading the source at commit `c9d0268`, not by reading documentation.
42 tests pass in `aura-server`, 95 across the workspace.

| Component | State | Note |
|---|---|---|
| Decision engine, admission, eviction, scoring | Real | 3.2 µs p50 on the write path, zero on reads |
| Nine baseline policies | Real | LRU, FIFO, LFU, GDS, GDSF, W-TinyLFU, S3-FIFO, SIEVE, LeCaR, plus Belady as the ceiling |
| Benchmark harness | Real | Drives the real policy implementations (D37), one identical stream, metadata overhead charged per policy |
| Cost vector and economic model | Real | Measured by the application, not estimated |
| Thompson mixture with decay | Real | Rewarded from the 60-second outcome, not from the decision |
| Delayed feedback loop | Real | `feedback.rs` wired: settle at 60 s, retire at 600 s, online model corrected from realised labels |
| Trained model | Real | LightGBM on 397,616 rows. AUC 0.9396 / 0.9251 / 0.8836, ECE 0.0123 / 0.0145 / 0.0491 |
| Feature projection | Real | Bundle feature names resolved to engine indices; an unknown name is refused, never silently dropped |
| Consistency | Real | Dependency tags, namespace versions, soft/hard TTL, three distinct removal reasons — wired into `get`, `put` and every removal path |
| Refresh | Real | Rebuilds through the origin and pays for it (D35). No longer resets the clock on bytes nobody rebuilt |
| Single-flight | Real | Lease on the miss path (D36); `origin_calls_suppressed` counted |
| Audit log | Real | Every writer wired, sentences with the numbers behind them, correctness events never sampled away, shipped to Supabase with re-queue on failure |
| Capacity controller | Real economics | Explains the ROI arithmetic in words, including the decision *not* to spend |
| Ghost cache / miss-ratio curve | Real | 1% spatial sample |
| Feature builder | Real | Parity-tested against the Python trainer |
| HTTP + WebSocket surface | Real | 30 routes; `/v1/invalidate`, `/v1/version/bump`, `/v1/consistency`, `/v1/refresh/queue` added |
| Supabase control plane | Real | Off the request path: model registry, benchmark results, events, audit log |
| Deployment | Real | Docker Compose, three images, Railway config |
| Dashboard | Partial | JSX, GSAP, live telemetry — but nothing yet renders the `consistency` block or the single-flight counters |
| **Traffic** | **Synthetic** | The engine's own generator. The two applications exist but do not yet drive the demo |
| **Timing under load** | **Not honest** | `take()` ignores `rps()`, so an offline flash crowd changes which keys arrive, not how many. p95 is isolated rebuild cost, not latency under concurrency |
| **L1** | **Real in the SDK** | `apps/common/l1.py`; the engine's own L1 is still an admission window holding keys, not values |
| **Multi-node** | **Missing** | Single replica by decision (D31) until consistent hashing exists |

## 4. Task board

Status values: `todo`, `doing`, `blocked`, `done`. Owner is a person or a session.

### Blockers — each one makes a sentence in the pitch untrue

| ID | Task | Owner | Status | Notes |
|---|---|---|---|---|
| B1 | Delayed feedback: decision journal, realised bandit reward, online predictor labels | engine | **done** | Wired in `engine.rs`; `settle_feedback` runs on the controller tick. |
| B2 | Consistency: dependency tags, namespace versions, soft/hard TTL, three removal reasons | engine | **done** | `depends_on` on PUT, `/v1/invalidate`, `/v1/version/bump`, `/v1/consistency`. Freshness checked before value. |
| B3 | Real refresh — call the origin and replace the value | engine | **done** | `Engine::rebuild` charges the backend ledger; `/v1/refresh/queue` publishes the backlog for the application to rebuild. |
| B4 | Train and publish a bundle with the portable features (D8) | training | **done** | Three GBDT bundles in `engine/models/` and in Supabase. `reuse_linear_h60s` in Supabase is a stale export and is refused at load — delete or re-export it. |

### Evidence — turns mechanisms into proof

| ID | Task | Owner | Status | Notes |
|---|---|---|---|---|
| E1 | Drive the demo from the two applications instead of the generator | apps | todo | Configuration, not code. Removes "traffic generated by the cache itself". |
| E2 | User simulator: sessions, clicks, purchases, real concurrency | apps | **doing** | Population and driver exist and have issued 5,269 real HTTP requests at 263 rps. Not yet the demo's traffic source. |
| E3 | Honest timing: Poisson arrivals respecting `rps()`, bounded worker pool, windowed metrics | engine | todo | Three defects, one fix. The offline harness still ignores spikes entirely. |
| E4 | Per-application policy mixture on screen | dashboard | todo | The experiment that proves adaptation. |
| E5 | Allocation-over-time chart | dashboard | todo | The frame already carries `capacity`. |
| E6 | Benchmark against S3-FIFO, SIEVE, W-TinyLFU | engine | **done** | All nine run by default, through their real implementations (D37). |
| E7 | Consistency and single-flight panels | dashboard | todo | The frame carries `consistency` including `origin_calls_suppressed`; nothing renders it. |

### Correctness and scale

| ID | Task | Owner | Status | Notes |
|---|---|---|---|---|
| C1 | Single-flight on the read path | engine | **done** | Lease returned on a miss (D36). |
| C2 | Postgres trigger + listener → `/v1/invalidate` | db | **doing** | `005_consistency.sql` and `apps/invalidator/listener.py` exist; the endpoint they call now exists too. Needs an end-to-end run. |
| C3 | Epoch-versioned personalised keys + deliberate bypass | apps | todo | D24, D25. |
| C4 | Real in-process L1 in the SDK; rename the admission window | apps | **doing** | `apps/common/l1.py` is real with 16 tests. The engine's own L1 is still misnamed. |
| C5 | Per-application memory floors and reallocation | engine | todo | The "move the boundary instead of buying" story is in the pitch, not the code. |
| C6 | `--real-values` by default | engine | todo | D30. Half an hour. |
| C7 | Multi-node: consistent hashing, `ScaleOut` | engine | todo | Only after single node is honest. |
| C8 | Durable audit log | engine | **done** | `006_audit_log.sql`, batched shipper, re-queue on failure. |

### Shipping

| ID | Task | Owner | Status | Notes |
|---|---|---|---|---|
| S1 | Docker Compose | deploy | **done** | Three images, `mem_limit` set so the pool is a real constraint. |
| S2 | Railway | deploy | **done** | `railway.json`; single replica (D31), `Cache-Control: no-store`. |
| S3 | Trace emitter | engine | **done** | The journal retires labelled rows; `/v1/training/rows` drains them. |

---

## 5. Open questions

1. **Which scenarios we actually win.** The benchmark now runs the real baselines rather than caricatures of them, so the margin will be smaller and the ranking may change. The brief requires beating conventional caching on at least three scenarios; until those five tables are read, that claim is unproven.
2. **Team split.** D33 says one owner per directory; the owners are not assigned.
4. **Host memory.** `host_budget_bytes` is a config constant, not a reading of real free RAM. Decide whether to read the OS or to keep it declarative and say so.
5. **Benchmark honesty.** `take()` in the offline harness ignores the spike function, so flash crowds change only which keys are requested, not the load. Fix in E3 or stop calling that scenario a flash crowd.

---

## 6. Changelog

| Date | Change |
|---|---|
| 2026-09-04 | Document created. D1–D33 recorded. `feedback.rs` and `consistency.rs` added for B1/B2. D8 supersedes the 16-feature vector in `CONTRACTS.md` §5. |
| 2026-09-04 | B1–B4, C1, C8, E6, S1–S3 closed. D34–D37 recorded. Consistency wired into the read path; refresh rebuilds instead of resetting the clock; the benchmark drives the real policy implementations. §3 rewritten against commit `c9d0268`. |
