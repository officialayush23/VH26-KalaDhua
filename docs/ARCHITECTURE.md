# AURA — what is where, and what each piece does

Written for someone who does not read Rust. You should not need to touch the Rust to run,
demo or explain this system.

---

## 1. The one-paragraph version

A cache normally keeps whatever was touched most recently. That is a bad rule when some
objects are expensive to rebuild and others are cheap: recency throws away a $0.004
database rollup to make room for a $0.000002 lookup. AURA prices every object by what
rebuilding it actually cost, predicts whether it will be asked for again, and keeps the
ones that are worth the space. The claim is measurable, so the repository measures it: the
same request stream is replayed through LRU, LFU, GDSF and AURA and the costs are compared.

---

## 2. The four processes

There are four things that can run. Only the first two are needed for the demo.

| # | Process | Language | Where | What it is |
|---|---|---|---|---|
| 1 | `aura` engine | Rust | `engine/` | The cache itself, plus the traffic simulator and the benchmark. Serves HTTP and a WebSocket on :8080. |
| 2 | dashboard | React | `frontend/universe/` | The web page. Talks to the engine only. |
| 3 | example apps | Python | `apps/` | Three fake services that use the cache the way a real application would. This is the "backend" in the ordinary sense. |
| 4 | training | Python | `training/` | Builds a dataset, trains the reuse model, uploads it to Supabase. Runs in Colab, not on your machine. |

Supabase is not a process you run. It is a hosted Postgres and file store that pieces 1, 3
and 4 talk to.

---

## 3. How the pieces talk

```
                          browser
                             │  WebSocket + HTTP
                             ▼
   ┌─────────────────────────────────────────────┐
   │  aura engine  (Rust, port 8080)             │
   │                                             │
   │   cache  ·  decision engine  ·  simulator   │
   └───────┬─────────────────────────┬───────────┘
           │ HIT: answered here      │ control plane only
           │ (no database touched)   │ model bundles in,
           │                         │ benchmark results out
           │ MISS                    ▼
           ▼                    ┌──────────┐
   ┌────────────────┐           │ Supabase │
   │ example apps   │──────────▶│ Postgres │
   │ (Python)       │  real SQL │ + Storage│
   └────────────────┘           └──────────┘
                                     ▲
                                     │ trained model bundle
                                ┌─────────┐
                                │  Colab  │
                                └─────────┘
```

**The important architectural point.** Supabase is *not* in front of the cache. If every
read went through Postgres, the cache would be pointless. Postgres is reached on a **miss**,
by the application service, which measures how long that query took and hands the measured
cost back to the cache on the write. That measured cost — not a guess, not a sleep timer —
is what the engine optimises against.

The engine's own connection to Supabase is control plane only, and never on the request
path: pull the trained model on boot, push benchmark results after a run, append events.
If Supabase is down the cache keeps working.

---

## 4. File by file

### `engine/` — the Rust cache. You run it; you do not need to read it.

**`engine/aura-core/`** — shared vocabulary and maths. No I/O, no server.

| File | What it does |
|---|---|
| `types.rs` | The core value types. `CostVector` is the important one: it holds cpu / gpu / db / network / api cost separately instead of collapsing them into one number, because two objects that both take 300 ms can have very different economics. |
| `config.rs` | Every tunable, and the prices. `Pricing::regen_cost_usd` turns a `CostVector` into dollars. These numbers are mirrored in the Python training config so both sides price identically. |
| `features.rs` | Turns a stream of accesses into the 16 numbers the model sees: age, inter-arrival, frequency at three windows, trend, acceleration, size, rebuild cost, TTL remaining, cache pressure. `FEATURE_NAMES` here must match the model bundle exactly. |
| `sketch.rs` | Small memory-cheap counters: decaying counters, a count-min sketch, running quantiles. This is how per-key statistics stay affordable at request rate. |
| `rng.rs` | Seeded random and a Zipf sampler. Seeded so every benchmark run is reproducible. |

**`engine/aura-sim/`** — generates the traffic.

| File | What it does |
|---|---|
| `scenario.rs` | The six traffic patterns (steady, flash crowd, scan, expensive tail, shifting popularity, mixed). |
| `attack.rs` | The nine disturbances the buttons on the dashboard fire. |
| `generator.rs` | Produces the actual requests. It owns the ground truth: how big each object is and what rebuilding it costs. Three applications with deliberately different cost shapes. |

**`engine/aura-server/`** — the running program. Binary is called `aura`.

| File | What it does |
|---|---|
| `main.rs` | Starts everything. Defines every HTTP route, the WebSocket, and the 250 ms loop that drives simulated traffic, the capacity controller and the telemetry frame. |
| `store.rs` | The cache storage itself. A small L1 recency window in front of the scored L2, size accounting, TTL expiry, hit/miss counters. |
| `engine.rs` | **The heart.** `get()` is the read path — a hash lookup, nothing more. `put()` is where the decision happens: price the object, predict reuse, compute value density, compare against the object that would be evicted, admit or refuse, record why. Also holds the shadow caches that replay the same stream under LRU/LFU/GDSF for the live comparison. |
| `predictor.rs` | Reuse prediction. Loads a trained bundle if there is one (a pure tree walker, no ML runtime needed), otherwise falls back to a logistic model that learns online. This is why the engine works before you have trained anything. |
| `policy.rs` | The six classical strategies as competing experts, Thompson sampling to blend them, and regime detection from six streaming statistics. |
| `capacity.rs` | Answers "is more memory worth buying" in dollars per hour, using the miss-ratio curve and the price model. |
| `bench.rs` | The offline benchmark. Replays one fixed stream through every policy plus Belady's optimal, which needs the future and so exists only offline. |
| `supabase.rs` | The control-plane client. Pull model bundles, publish benchmark runs, append events. Also loads `backend/.env`. |

### `apps/` — the Python example services. This is "the backend".

| File | What it does |
|---|---|
| `common/aura_client.py` | The client an application uses. `get_or_regen(key, context, regen)` asks the cache, and on a miss runs your expensive function, **measures what it actually cost**, and writes it back with that measurement. |
| `common/costing.py` | Turns measured work into a `CostVector`. Mirrors the Rust pricing. |
| `common/service.py` | Shared FastAPI scaffolding for the three services. |
| `common/loadgen.py` | Generates load against an app (`POST /load`). |
| `analytics/db.py`, `analytics/queries.py` | **The Supabase connection.** Real SQL against the `app_*` tables. This is what makes a miss genuinely expensive. |
| `recommendation/` | Simulated ranking service. CPU + GPU heavy. |
| `content/` | Simulated media service. Large objects, external API cost. |
| `driver/run_universe.py` | Starts all three at once. |

### `training/` — the model.

| File | What it does |
|---|---|
| `notebooks/aura_training_colab.ipynb` | **The notebook you run.** 28 cells, start to finish: install, dataset, train, evaluate, export, push to Supabase, tell the engine to reload. |
| `aura_train/features.py` | The Python half of the feature builder. Must agree with `engine/aura-core/src/features.rs`. |
| `tests/test_features_parity.py` | Proves those two agree, against golden vectors. If you change features on one side, this test fails. |
| `aura_train/synthetic.py` | Generates training data when you have no recorded traces. This is what you will use. |
| `aura_train/labels.py` | Builds the target: was this key asked for again within 10 s / 60 s / 600 s. |
| `aura_train/train_gbdt.py` | Trains the gradient-boosted model, plus ablations. |
| `aura_train/train_linear.py` | Trains the small linear model used for cold start. |
| `aura_train/evaluate.py` | AUC, PR-AUC, and a cache replay to check the model actually improves the cache rather than just scoring well. |
| `aura_train/export.py` | Writes the portable `model_bundle.json` the Rust side reads. |
| `aura_train/supabase_io.py` | Uploads the bundle and registers it in `aura_models`. |
| `sql/003_supabase_schema.sql` | **Run this first.** Tables, constraints, indexes, security. |
| `sql/004_supabase_seed.sql` | **Run this second.** Fills the analytics tables with 120k orders. |

### `docs/`

| File | What it does |
|---|---|
| `CONTRACTS.md` | The frozen interface. Every payload shape and every route. Change this before changing code. |
| `PROGRESS.md` | Running build state: what is done, what is not. |
| `ARCHITECTURE.md` | This file. |

---

## 5. Where the environment file goes

`backend/.env` — already there, already gitignored. The engine walks up from wherever it
was started looking for `backend/.env` then `.env`, so it finds it whether you run from
`engine/` or the repository root. A real shell variable always wins over the file.

Keys it reads:

```
SUPABASE_URL=https://<project>.supabase.co
SUPABASE_SERVICE_ROLE_SECRET_KEY=<service role key>
```

The Python side reads the same file. Nothing else needs configuring.

To confirm the engine picked it up, open `http://localhost:8080/v1/supabase` or look at the
Supabase panel on the dashboard.

---

## 6. What to say when someone asks how it works

1. Caches decide what to keep by recency. Recency is a proxy for value, and it is a bad one
   when rebuild costs differ by four orders of magnitude.
2. So measure the real cost. The client measures how long the rebuild actually took and
   what it consumed, and reports that. Nothing is estimated.
3. Predict whether it will be needed again, at three time horizons.
4. Keep the objects with the highest value per byte per second held. Admission compares the
   arriving object against the object that would have to be evicted for it.
5. Prove it. Replay the identical request stream through LRU, LFU and GDSF and compare
   total cost, with Belady's optimum as the ceiling.

The honest version of the result: AURA wins clearly where rebuild costs vary a lot
(`expensive_tail`, `scan_resistance`). On uniform-cost traffic it is close to GDSF and
sometimes slightly behind. Say that — it is a stronger position than a claim a judge can
break with one query.
