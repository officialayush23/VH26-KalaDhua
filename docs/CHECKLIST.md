# The brief, item by item

Every requirement, what answers it, and where. Status is one of **done**, **partial**
(with what is missing named) or **open**. Nothing here is aspirational: if it says done,
there is a file and a route behind it, and if it is not finished the gap is written down
rather than rounded up.

---

## 1. Adaptive scaling with a cost/benefit decision — **done**

`engine/aura-server/src/capacity.rs`

The controller fits a marginal miss-ratio curve from the hit rate it has actually observed,
then prices the *next* block of memory rather than reacting to fullness:

```
roi     = savings / cache_cost
savings = delta_hit_rate x requests_per_hour x average_miss_usd
cost    = extra_gb x cache_gb_hour_usd
```

It grows when `roi >= 1.25` and the host has headroom. It shrinks only when utilisation is
under 0.55 **and** it has seen real evictions **and** at least 5,000 requests have gone
through — the guard that stops it mistaking a cold cache for an oversized one. It runs on
the 250ms controller tick with a cooldown, so it cannot oscillate.

A cache that grows whenever it is full is not scaling. It is failing to evict.

## 2. Cost-awareness layer — **done**

`engine/aura-core/src/config.rs` (`Pricing`), ledger in `engine.rs`

Priced: CPU ms, GPU ms, database ms, network bytes, direct API dollars, memory held per
GB-hour, and an SLA penalty per millisecond over the p95 objective. `total_usd` is rebuild
cost **plus** SLA penalty **plus** what the memory itself costs to hold — which is why a
policy that keeps everything can win on hit rate and still lose here.

## 3. Measurable savings against LRU / LFU / GDS — **done, with a caveat that is stated**

Two independent mechanisms:

- **Live shadow caches.** LRU, LFU and GDS see every request the engine sees, charged the
  same prices, and their running cost is on the Evidence tab beside AURA's.
- **Offline replay.** `POST /v1/bench/run` replays one fixed, seeded request stream through
  every policy — LRU, FIFO, LFU, GDS, GDSF, W-TinyLFU, S3-FIFO, SIEVE, LeCaR — plus Belady
  as the offline optimum. Real implementations, not scoring caricatures.

**The caveat, stated on the deck and repeated here:** the headline benchmark gives AURA
512MB against the baselines' 128MB. At equal memory the `aura_fixed` control loses on four
of six scenarios. The two it wins are the cost-skewed ones, which is what it was built for.

## 4. Dashboard for visualisation and benchmarking — **done**

`frontend/universe`, nine tabs. The Evidence tab is the one that answers the brief:

| Chart | The question it answers |
|---|---|
| Offered vs arriving load | A disturbance is the gap between the two lines |
| Admissions / refusals / evictions per tick | Refusals rising while evictions stay flat is the admission gate declining objects instead of throwing better ones out |
| Running cost vs LRU, LFU, GDS | The distance between the lines is the money the decision made |
| Pool size, with every scale decision marked | When it grew, when it shrank, and what it said at the time |
| Bandit posterior weights, stacked over time | The shift when the workload changes *is* the adaptation |
| Hit rate, both tiers | L1 and L2 do different jobs; one blended number hides which is working |

The engine keeps no history — it sends a complete snapshot per frame — so these series
accumulate in the browser. They fill as you watch. Nothing is backfilled.

## 5. Backend-agnostic, plug and play — **done**

The engine caches opaque values and never inspects them. An application declares what an
object cost to build (`CostVector`) and what it was built from (`depends_on` row tags); it
declares nothing else. There is no schema, no registration step, and no backend driver: the
key is the application's identity, so a service appears in the console the moment it makes
its first call. Python client in `apps/common/aura_client.py`; the wire protocol is plain
HTTP JSON, so any language can speak it.

## 6. At least two workload types — **done**

`apps/recommendation` (small, hot, latency-shaped) and `apps/analytics` (large, expensive,
scan-prone rollups over Postgres). Six generator scenarios on top: `steady_zipf`,
`flash_crowd`, `scan_resistance`, `expensive_tail`, `shifting_popularity`,
`mixed_production`.

## 7. Decisions at runtime, not preprocessed — **done**

Every candidate is scored at eviction time against the pool as it is *now*. Nothing is
ordered at insert time. `value_density = rebuild_cost x expected_reuses x tail_risk /
holding_cost`, expected reuses blended over three horizons (10s / 60s / 600s) and scaled by
the TTL remaining. Sampled eviction takes 32 random residents per pass — within a few
percent of the true minimum, without the O(n) scan a real request rate cannot afford.

Admission uses the same number: an arrival must beat the object that would have to be
evicted to make room for it (`victim_bar x 1.10`), not beat an average of arrivals.

## 8. Justified scoring model — **done** (hybrid: heuristic experts + online model + bandit)

Six eviction heuristics (LRU, LFU, GDSF, TinyLFU, cost-aware, trend-aware), each with a
Beta posterior. Thompson sampling draws from those posteriors; the resulting mixture
weights every expert's opinion. Rewards are **realised, not predicted**: a decision is
judged 60 seconds after it was made, on what actually happened. Posteriors decay at 0.995
per tick so evidence from a workload that has ended stops outvoting the one running now.

An online model contributes a reuse prediction, blended in proportion to its confidence,
which ramps over the first 5,000 samples — so a cold model cannot ruin the first minute.

## 9. Beats conventional caching on three or more scenarios — **partial**

Wins 5 of 6 at 512MB vs 128MB; 2 of 6 at equal memory. Honest framing: *cost-skewed
workloads are where a value-scored cache pays for itself, and uniform-cost workloads are
where it cannot.* See item 3.

**Open:** capping the benchmark's `max_bytes` at the baselines' capacity would make the
headline numbers memory-fair. One line, not yet applied.

## 10. Dynamic eviction driven by signals — **done**

Recency, frequency, trend, rebuild cost, object size, TTL remaining, cost variance (tail
risk), and current cache pressure. Plus three things a scoring function alone will not do:

- **Scan resistance** — a single-touch signature with a confident Scan regime is refused
  admission rather than evicting the working set to hold it.
- **Per-application floors** — a scan in analytics cannot evict recommendation below its
  share. Each eviction was correct on value density; the sum of correct decisions was a
  cache that had forgotten a tenant.
- **Dependency invalidation** — a row changes in Postgres and exactly the objects built
  from it are dropped, by surrogate tag. Not a TTL, not a flush.

## 11. Two-tier architecture, L1 per service and a global L2 — **done**

`apps/common/l1.py`, wired into `apps/common/aura_client.py`.

L1 is an in-process LRU, byte-capped, 5 second maximum TTL. L2 is the engine. They protect
different things: L1 removes a network round trip, L2 removes a rebuild and protects the
origin across the whole fleet. Only L2 can reason about value, because only L2 sees every
process's demand — so L1 is deliberately a plain LRU. Two adaptive policies competing over
the same objects would make every measurement ambiguous.

The rule that matters: **an object is only eligible for L1 if it can tolerate being wrong
for a few seconds.** Anything declared with `depends_on` row tags defaults to
`write_bound` and stays L2-only, because invalidating L2 is one message to one service
while invalidating L1 means reaching every process that might hold a copy.

## 12. Configurable load, dynamic traffic, bursts, cold start — **done**

- `POST /v1/sim/rps` sets the offered rate at runtime, separately from `/v1/sim/speed`
  (which scales virtual time and so stretches the scenario's disturbances with it). The
  console has a req/s box.
- Nine injectable disturbances including flash crowd, hot-key emergence and decay, working
  set explosion, mixed chaos.
- Cold start: the model's confidence ramps over 5,000 samples and the capacity controller
  refuses to shrink below that count.

## 13. Every decision visible with evidence — **done**

- `ActivityLog` — every admission, refusal, eviction, invalidation and scale decision as a
  sentence containing the numbers that produced it.
- `DecisionFeed` — per-object: action, rebuild cost, reuse probability, value density
  against the eviction threshold, and the reason.
- `GET /v1/explain/:key` — why one specific object is or is not resident.
- `GET /v1/audit` — the same log, filterable, persisted to Supabase.

---

# Still open

| Item | Why it matters |
|---|---|
| Memory-fair benchmark headline | The strongest single objection a judge can raise |
| `bandit.kind` is an inert knob | It is settable and changes nothing; the implementation is always Thompson sampling |
| Free-tier hosting sleeps | First request after idle takes ~50 seconds and reads as a broken demo |
| Rotate the Supabase credentials | They were pasted into a shared context during development |
| `ALGORITHMS.md`, `PS_MAPPING.md` | Still say refresh is broken and carry outdated benchmark tables |
