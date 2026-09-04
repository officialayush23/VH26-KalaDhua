# AURA vs the problem statement

VCET Hack-a-thon 2026 · Application Scaling · Adaptive, Application-Aware Cache Management.

Read this before the demo. It maps each thing the PS asks for onto the file that does it,
and says plainly what is real and what is not.

---

## The system in five sentences

1. An application asks the cache for an object by key.
2. On a **hit** the cache answers from memory and nothing downstream is touched.
3. On a **miss** the application rebuilds the object itself and tells the cache *what that
   rebuild actually cost* — CPU ms, GPU ms, DB ms, bytes, API dollars — not a guess.
4. The cache scores the object on that measured cost, how likely it is to be asked for
   again, and how much space it occupies, and keeps it only if it beats whatever would have
   to be evicted to fit it.
5. Separately, every few seconds it asks "would renting more memory pay for itself?" and
   resizes itself if the answer is yes.

Everything else is machinery serving those five steps.

---

## Requirement by requirement

### "A scoring/decision engine that ranks cached objects using a weighted, multi-factor model (not a single metric) to decide retain vs evict vs refresh"

**Where:** `engine/aura-server/src/engine.rs`, function `value_density`.

The score is not one metric. It combines:

| Signal | Where it comes from | PS signal |
|---|---|---|
| frequency at 1 m / 5 m / 1 h | decaying counters, `aura-core/src/sketch.rs` | access patterns |
| recency, inter-arrival gap | `aura-core/src/features.rs` | access patterns |
| trend and acceleration | fast vs slow EWMA ratio | *extra signal we added* |
| object size | `ObjectContext.size_bytes` | resource footprint |
| rebuild cost, split into CPU/GPU/DB/network/API | `CostVector`, priced by `Pricing::regen_cost_usd` | retrieval cost |
| cost variance (p95/p50 of rebuild time) | running quantiles per operation class | *extra signal* |
| TTL remaining | freshness | *extra signal* |
| cache pressure | occupancy at decision time | *extra signal* |
| SLA class | per-object penalty weight | *extra signal* |

Sixteen features in total, listed in `aura-core/src/features.rs` as `FEATURE_NAMES`.

**The formula**, in words: expected number of future reuses × what each avoided rebuild is
worth × a tail-risk multiplier, divided by what holding the object costs for a minute. The
result is *dollars saved per byte per second held*, which is why objects of wildly
different sizes and costs can be compared at all.

**Retain vs evict:** an arriving object is admitted only if its density beats the density
of the object that would have to be evicted for it (`victim_bar`). This is the part that
makes it scan-resistant.

**Refresh:** `refresh_candidates` picks high-value objects near TTL expiry so the expiry is
never paid for on a user request. **Honest caveat:** it currently resets the timestamp
rather than rebuilding the object. That is a real gap — see `docs/REMAINING.md` item 8.

### "Decisions must be made adaptively at runtime, not via static/hardcoded rules"

Three separate adaptive mechanisms, none of them hardcoded:

1. **Policy mixture** — `policy.rs`. Six classical strategies (LRU, LFU, GDSF, TinyLFU,
   cost-aware, trend-aware) run as competing experts. Thompson sampling assigns each a
   weight from how well it has actually been paying off, and the weights decay so a regime
   change cannot be outvoted by an hour of stale evidence. Watch the mixture bar move on
   the dashboard when you press Cost spike.
2. **Regime detection** — `engine.rs::recompute_workload`. Six streaming statistics
   (burstiness, entropy, working-set growth, reuse distance, popularity shift, novelty
   rate) classify the current traffic into Steady / FlashCrowd / Scan / Shifting /
   Expensive / Growing. The classification changes admission behaviour directly.
3. **The learned predictor** — `predictor.rs`. Reuse probability at three horizons. When no
   trained bundle is loaded, a logistic model runs instead so the engine is never blind.

**This is what "platform adapting" means in this codebase.** If someone asks where the
runtime adaptation is, show them the policy mixture bar and the regime label changing while
traffic changes, and the event log filling with `PolicyShift` lines.

### "Adaptive scaling logic that decides when additional cache capacity is actually justified (cost-benefit tradeoff), rather than scaling reactively/blindly"

**Where:** `engine/aura-server/src/capacity.rs`.

It does not scale on a utilisation threshold. Every few seconds it:

1. Builds a miss-ratio curve anchored on the measured hit rate.
2. Reads off how much hit rate the next block of memory would buy.
3. Prices that: extra hit rate × requests per hour × average cost of a miss.
4. Prices the memory: extra GB × `cache_gb_hour_usd`.
5. Scales up only if the ratio clears `roi_threshold`, scales down when the pool is
   under-used, and holds otherwise. A cooldown stops it oscillating through a burst.

The dashboard prints the actual sentence, e.g. *"marginal ROI 4.1x above the 1.25x bar;
128 MB free on host"*.

**This is the single biggest differentiator in the benchmark.** No classical policy has any
notion of resizing itself. See the numbers below.

### "A cost-awareness layer that models infrastructure cost (memory, compute, API calls) and demonstrates measurable savings versus baseline LRU/LFU"

**Where:** `aura-core/src/config.rs`, struct `Pricing`. Same constants mirrored in
`training/aura_train/config.py` so the model and the engine price identically.

```
cpu_ms_usd, gpu_ms_usd, db_ms_usd, network_gb_usd,
cache_gb_hour_usd, sla_penalty_per_ms_over_slo_usd, slo_p95_ms
```

Total cost charged to every policy = rebuild cost + SLA penalty + memory rent, where memory
rent is occupancy **integrated over time**, not a snapshot.

### "A simulated or real workload generator ... popularity spikes, cold starts, traffic bursts"

**Where:** `engine/aura-sim/`. Six scenarios and nine injectable disturbances: FlashCrowd,
Scan, PopularityShift, CostSpike, ExpensiveTail, HotKeyEmergence, HotKeyDecay,
WorkingSetExplosion, MixedChaos. Seeded, so a run repeats exactly.

### "Application-agnostic enough to plug into different backends (at least 2 distinct workload types)"

Three, not two, in `apps/`:

| Service | Type | Cost shape |
|---|---|---|
| `apps/analytics` | read-heavy, queries Postgres | DB-heavy, ~40 KB objects |
| `apps/recommendation` | compute-heavy ranking | CPU + GPU, ~120 KB objects |
| `apps/content` | media/API service | GPU + paid external API, ~600 KB objects |

The cache never learns what a "recommendation" is. It only ever sees `ObjectContext`:
application name, object type, size, TTL, SLA class, cost vector. That is what makes it
application-agnostic, and it is worth saying out loud in the demo.

### "Benchmarking: comparative report showing hit rate, latency, cost against LRU/LFU under identical, dynamically changing workload conditions"

**Where:** `engine/aura-server/src/bench.rs`. One request stream is generated once from a
fixed seed and replayed through every policy, so no policy gets luckier traffic.

Results, 50,000 requests, 128 MB starting pool, seed 42, all six policies plus Belady:

| scenario | AURA rank | vs best baseline | vs LRU |
|---|---|---|---|
| steady_zipf | 1 / 6 | +15.9% | +32.5% |
| expensive_tail | 1 / 6 | +10.2% | +19.4% |
| shifting_popularity | 1 / 6 | +10.3% | +12.2% |
| mixed_production | 1 / 6 | +8.8% | +18.2% |
| flash_crowd | 1 / 6 | +1.3% | +17.5% |
| scan_resistance | 1 / 6 | +1.1% | +4.6% |

Belady's offline optimum is reported as the ceiling. It needs the future, so it is a bound,
not a competitor.

**Be ready for this question:** *"you let AURA use more memory."* Yes — and it is charged
for every byte at the same price, integrated over time, and still wins. Choosing how much
memory to rent is the contribution, not a loophole. The baselines keep a fixed pool because
having no mechanism to resize is precisely the gap being demonstrated.

### "A dashboard showing cache hit/miss ratio, cost savings, latency improvements, and eviction/refresh decisions in real time (bonus)"

`frontend/universe`. Five tabs: Overview, Request flow (animated, live), Decisions,
Benchmark, Model & system.

---

## What is real and what is simulated

Say this before someone finds it.

| Piece | Status |
|---|---|
| Decision engine, admission, eviction, policy blend, regime detection | **Real** |
| Cost model and benchmark | **Real** |
| Cached bytes held in RAM | **Real with `--real-values`** (measured 1.19 GB resident for an 800 MB pool) |
| Analytics rebuild cost | **Real with `--real-backend`** (live Supabase query, measured) |
| Traffic | **Simulated** by default. Real when `apps/` runs |
| Trained model | **Not trained yet.** Online logistic fallback |

The PS says "a simulated **or** real workload generator", so simulated traffic is compliant.
Do not pretend otherwise; the dashboard says which mode it is in.

---

## Where the model is, and how to train it

**The notebook:** `training/notebooks/aura_training_colab.ipynb` — 28 cells.

1. Upload it to Google Colab.
2. Left sidebar, key icon, add two secrets: `SUPABASE_URL` and
   `SUPABASE_SERVICE_ROLE_SECRET_KEY`.
3. Runtime → Run all.

**The dataset:** cell 6 has `TRACE_SOURCE = "synthetic"`. It generates its own training
data via `training/aura_train/synthetic.py`. This is the correct default and the only one
that carries cost metadata, which the economic part of the model needs. Public traces
(Twitter, Wikimedia, libCacheSim — see `training/scripts/fetch_public_traces.sh`) have no
cost columns and would train the reuse half only.

**Where it plugs in:** the notebook uploads `model_bundle.json` to Supabase Storage and
registers it in `aura_models` with `is_active`. Then either press **Load from Supabase** on
the dashboard's Model tab, or:

```
curl -X POST http://localhost:8080/v1/model/reload -H "content-type: application/json" -d "{\"source\":\"supabase\"}"
```

No rebuild, no restart. The Model tab will change from `heuristic` to `gbdt`.

Skip the notebook's last cell — it posts to `localhost:8080`, which Colab cannot reach from
Google's servers.

---

## Running it

### Mode A — simulated traffic (fast, what you have been using)

```
cd engine
cargo run --release -p aura-server -- --scenario mixed_production --real-values
cd frontend/universe && npm run dev
```

### Mode B — real applications driving the cache (the stronger demo)

```
# terminal 1 — engine with NO --scenario, so the internal generator stays off
cd engine
cargo run --release -p aura-server -- --real-values --real-backend

# terminal 2 — the three services become the traffic
cd apps
pip install -r requirements.txt
python -m driver.run_universe --spawn --rps 200 --duration 300

# terminal 3
cd frontend/universe && npm run dev
```

In Mode B the traffic-source banner reads "live applications" and every rebuild cost is
measured by the service that paid it.

### Supabase

SQL editor, in order: `training/sql/003_supabase_schema.sql`, then
`training/sql/004_supabase_seed.sql`. The seed should end with 25 regions, 4,000 products,
40,000 customers, 120,000 orders, 360,000 order lines. If you see far fewer, the seed did
not finish — check with `select count(*) from app_customers;`.

---

## The five-minute demo

1. **Frame it.** "LRU keeps what was touched recently. That throws away a four-cent database
   rollup to keep a free lookup. We price every object by what rebuilding it actually cost."
2. **Request flow tab.** Dots moving. Lime returns from cache touching nothing; amber falls
   through to the application, which is the only thing that talks to the database.
3. **Press Scan.** Regime flips to `Scan`, refusals climb in the decision feed, hit rate
   holds. A plain LRU would flush its working set here.
4. **Press Cost spike.** The policy mixture bar shifts toward the cost-aware experts. That
   is the runtime adaptation, visible.
5. **Capacity panel.** Read the sentence aloud: it is a dollars-per-hour argument, not a
   utilisation rule.
6. **Benchmark tab.** Run `expensive_tail`. AURA lowest cost. Point at Belady and say it is
   the offline optimum nothing online can reach.

---

## Questions you should expect, and the honest answer

**"Is this real traffic?"** No, it is a seeded generator, which the PS permits. Mode B runs
the three real services instead.

**"Is that your actual RAM?"** With `--real-values`, yes — measured 1.19 GB resident. Without
it, it is an accounting budget.

**"Is the model trained?"** Not yet. It runs on an online logistic model. The pipeline and
notebook are complete; publishing takes one Colab run.

**"You gave AURA more memory."** It is charged for it at the same price, integrated over
time, and still wins. Deciding how much to rent is the contribution.

**"You only beat LRU, LFU and GDSF."** True. S3-FIFO and SIEVE are the obvious next
baselines and are roughly a hundred lines each — see `docs/REMAINING.md` item 5.
