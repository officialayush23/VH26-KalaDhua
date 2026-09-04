# AURA — the algorithms, and where each one lives

Everything the system decides, the formula it decides it with, and the file it is in.
Written to be read out loud during a pitch.

---

## 0. Naming: what "L1" means here

This trips people up, so settle it first.

| Name people use | What it is in this repo | File |
|---|---|---|
| **The cache** (the thing being judged) | AURA's scored pool. Reported as `layers.l2` in telemetry, for historical reasons | `store.rs` `Store::entries` |
| **Admission window** | A tiny recency set holding *keys only, no values*. Used to spot one-shot keys. **Not a cache tier** | `store.rs` `Store::l1` |
| L2 / L3 / Redis | **Not built, and deliberately so.** Downstream is simulated | `aura-sim/` |

You are building one intelligent cache. Adding a Redis tier adds network plumbing and
scores nothing against a problem statement about *scoring logic*. Downstream is simulated as
cost and latency, which is the right call.

**One correction to make in the demo:** the dashboard labels the main pool "L2" and the
window "L1 window". Say "our cache" and "our admission filter" out loud, so nobody thinks
there is a missing tier.

---

## 1. The read path — `engine.rs::get`

No scoring, no model, no allocation. A hit is a hash lookup, a counter bump and a clone.

```
get(key):
    touch admission window          # records the key was seen; stores no value
    if entry expired: drop it
    if entry present:
        hits += 1; entry.hits += 1
        credit the shadow baselines with this same request
        return entry
    misses += 1
    return None
```

**Why it matters:** the expensive machinery only runs on a miss, where a backend call is
already being paid for. Measured decision overhead is ~3 µs and it is entirely on the write
path. Say this when someone asks about overhead.

---

## 2. The scoring function — the heart

**File:** `engine.rs::value_density`

```
value_density(features, reuse, size_bytes, cost_usd):

    expected_reuses = reuse[10s] · w0 · 6
                    + reuse[60s] · w1 · 2
                    + reuse[600s] · w2

    tail_risk       = 1 + min(cost_variance_ratio, 4) · λ

    value           = cost_usd · expected_reuses · tail_risk

    hold            = price_of_holding(size_bytes, for 60 s)

    return value / hold
```

Read it aloud as: **"dollars of backend work this object will save me, per dollar it costs
to keep it."** Everything else in the system exists to feed this expression.

The five factors, and the PS signal each answers:

| Term | Comes from | PS signal |
|---|---|---|
| `reuse[·]` | learned predictor, three horizons | access patterns / frequency / recency |
| `cost_usd` | measured `CostVector`, priced | retrieval cost (DB latency, API cost) |
| `hold` ∝ `size_bytes` | object size × memory price × time | resource footprint |
| `tail_risk` | p95/p50 of rebuild time for that operation class | *extra signal we added* |
| horizon weights | `engine.horizon_weights` in config | — |

**Why divide by size:** without it, a 24 MB media file with the same reuse and cost as a
40 KB query result would score identically, and the cache would fill with the expensive
thing that displaces 600 other objects.

**Why three horizons:** "will this be reused" is meaningless without a time bound. Ten
seconds decides admission, sixty decides eviction, ten minutes decides refresh.

**Why cost is a vector, not a number:** `CostVector` keeps `cpu_ms`, `gpu_ms`, `db_ms`,
`network_bytes`, `api_calls`, `api_cost_usd` separate. Two objects that both take 300 ms to
rebuild can differ by 100× in price — one burns a GPU, the other holds a DB connection.
Collapsing that into "latency" is exactly the information loss that makes LRU mis-rank.
Priced by `Pricing::regen_cost_usd` in `config.rs`.

---

## 3. Admission — should this object come in at all?

**File:** `engine.rs::put`, and `engine.rs::victim_bar`

```
put(key, ctx, measured_cost):
    cost      = price(measured_cost)          # measured, not estimated
    features  = feature_builder.transform(access_event)
    reuse     = predictor.reuse(features)
    density   = value_density(features, reuse, size, cost)

    bar       = victim_bar(size)              # 0 if the cache has room
    threshold = bar · admission_margin

    if size > capacity / 4              -> Reject "object_exceeds_quarter_capacity"
    if regime == Scan and reuse10 < .15
       and key not seen in window       -> Reject "single_touch_scan_signature"
    if cache empty or density >= thresh -> Admit
    if reuse60 > 0.75                   -> Admit  "high_near_term_reuse"
    else                                -> Reject "density_below_threshold"
```

**`victim_bar`** samples 32 resident objects at random and returns the *lowest* density
among them — an estimate of what would have to die for this object to live.

**The rule in one sentence:** admit only if the arrival is worth more than the thing it
would displace. When the cache has room the bar is zero, because refusing costs a
guaranteed backend call to avoid a hypothetical one.

**This is the scan-resistance mechanism.** A one-touch scan key has near-zero predicted
reuse, so its density is near zero, so it never clears the bar and never displaces the
working set. LRU has no such gate and gets flushed.

> **Bug worth knowing about, since it was just fixed.** Admission and eviction were being
> computed by two different formulas in two different unit scales — arrivals scored in the
> thousands, the bar scored below one. Every arrival cleared it. **Zero rejections in
> 42,614 admissions.** Both sides now end in `value_density`, and the gate refuses ~46% of
> arrivals. If someone asks how you validated the mechanism, this is a good answer: we
> instrumented it and found it silently disabled.

---

## 4. Eviction — who leaves when space is needed

**File:** `engine.rs::make_room` and `engine.rs::resident_density`

```
make_room(bytes_needed, incoming_density):
    sweep expired entries first          # free wins before costly ones
    while not enough room:
        sample 32 resident keys at random
        victim = the one with the lowest resident_density
        if victim.density <= incoming_density · 1.05:
            evict it
        else:
            stop                          # nothing here is worse than what is arriving
```

**Why sampled and not a heap:** scanning every resident object to find the true minimum is
not affordable at request rate. Sampling 32 lands within a few percent of the true minimum
— the standard result Redis's own `allkeys-lru` relies on. It keeps eviction O(1) in cache
size.

**`resident_density`** puts a resident object on the *same scale* as an arrival:

```
resident_density(key):
    features      = feature_builder.peek(key)     # read counters, do NOT record a touch
    expert_score  = Σ mixture[i] · policy[i].utility(features, age)
    expert_reuse  = expert_score / (1 + expert_score)      # squash to [0,1)
    model_reuse   = predictor.reuse_peek(features)
    share         = max(ml_confidence_floor, predictor.confidence())
    reuse         = expert_reuse · (1 - share) + model_reuse · share
    return value_density(features, reuse, size, cost) · ttl_remaining
```

Note `peek`, not `transform`. Scoring an object is not the same as using it; recording a
touch would make every eviction sweep look like traffic.

---

## 5. The six expert policies — `policy.rs::Policy::utility`

These are the classical algorithms, implemented as competing scorers rather than as the
final word. All six are computed from the same feature vector.

| Policy | Utility formula | Optimises |
|---|---|---|
| `lru` | `1 / (1 + age_s)` | recency |
| `lfu` | `freq_1h` | frequency |
| `gdsf` | `freq_5m · cost / log_size` | Greedy-Dual-Size-Frequency |
| `tiny_lfu` | `0.7·freq_1m + 0.3·freq_5m` | frequency sketch, scan-resistant |
| `cost_aware` | `cost / log_size` | price per byte |
| `trend_aware` | `ewma_fast·(1+trend⁺) + accel⁺` | momentum |

They are deliberately **not** normalised against each other. The bandit learns the scale.

---

## 6. Runtime adaptation #1 — the bandit — `policy.rs::Bandit`

Thompson sampling over the six experts.

```
mixture():                       # the weights shown on the dashboard
    for each arm i:
        mean_i = α_i / (α_i + β_i)
        w_i    = mean_i + exploration
    normalise w

credit(policy, reward):          # reward = realised value, clamped to [0,1]
    α[policy] += reward
    β[policy] += 1 - reward
    regret     = decayed running gap to the best arm

decay(0.995) every controller tick
```

**Why decay:** without it the posterior freezes. An expert that was right for an hour would
outvote fresh evidence forever, and the cache would stop adapting the moment it had seen
enough. Decay is what makes "adaptive at runtime" true rather than "adaptive for the first
five minutes".

**Reward:** on an admission, how far the object's density exceeded the bar. On a rejection,
how confidently it was refused. So the arms are graded on realised economics, not hit rate.

**Demo line:** press *Cost spike* and the mixture bar visibly shifts toward `cost_aware` and
`gdsf`. That is the adaptation, on screen, in about two seconds.

---

## 7. Runtime adaptation #2 — regime detection — `engine.rs::recompute_workload`

Six streaming statistics over the last 8,192 requests, recomputed on the controller tick.
No model — this runs constantly and must not show up in the latency budget.

| Statistic | How it is computed | What it detects |
|---|---|---|
| `burstiness` | share of traffic held by the top 16 keys | flash crowds |
| `entropy` | Shannon entropy over the key histogram | concentration |
| `working_set_growth` | change in distinct-key count vs the last window | capacity pressure |
| `reuse_distance_p50` | median gap between repeats of a key | temporal locality |
| `popularity_shift` | 1 − overlap of the top-32 with the previous top-32 | the hot set turning over |
| `scan_score` | novelty rate × (1 − repeat rate) | sequential sweeps |

`classify()` scores six regimes and picks the argmax with a confidence:
`Steady · FlashCrowd · Scan · Shifting · Expensive · Growing`.

**A bug worth mentioning if asked how you tested:** `scan_score` was originally
`singles / total`, which any wide Zipf window also produces. Ordinary traffic was being
classified as a scan, so the engine mass-rejected admissions in normal conditions. It is now
a novelty rate.

---

## 8. Runtime adaptation #3 — capacity — `capacity.rs`

The PS asks for scaling justified by cost-benefit, not by a utilisation threshold. This is
the one axis where **no baseline has any mechanism at all**, which is why it is the largest
single source of the benchmark win.

```
every controller tick:
    mrc         = miss-ratio curve anchored on the measured hit rate
    Δhit        = mrc(current + step) − mrc(current)
    savings/hr  = Δhit · requests_per_hour · average_cost_of_a_miss
    rent/hr     = extra_GB · cache_gb_hour_usd
    roi         = savings / rent

    if roi ≥ roi_threshold and host has room  -> ScaleUp
    if roi ≥ threshold but host is full       -> ScaleOut
    if pressure < 55% and above the floor     -> ScaleDown
    else                                      -> Hold

    cooldown_s must have elapsed since the last change
```

The MRC is a concave saturating fit `h(b) = b / (b + k)`, with `k` solved from the *measured*
hit rate at the current size. It is an estimate, and the dashboard says so, but it is
anchored to live data rather than invented.

**Cooldown** exists so a transient burst cannot make the controller oscillate.

**Demo line:** read the reason string aloud — *"marginal ROI 4.1× above the 1.25× bar;
128 MB free on host"*. That is a dollars-per-hour argument, not a rule of thumb.

---

## 9. Refresh — `engine.rs::refresh_candidates`

```
every tick:
    candidates = resident entries with ttl_remaining < refresh_ttl_threshold (0.15)
    rank by resident_density
    take the top 4
```

Rebuild high-value objects *before* they expire, so the expiry is never paid for on a user
request.

**Be honest about this one:** it currently resets the timestamp rather than calling the
application to rebuild the object. The selection logic is right; the rebuild is not wired.
If asked, say the ranking works and the rebuild call is the remaining piece.

---

## 10. The prediction model — `predictor.rs`

**Loaded bundle** (`model_bundle.json` from the Colab notebook): a pure Rust tree walker.
No ML runtime, no ONNX required, no Python in the serving path. Missing values go left,
matching LightGBM's dump convention. Three bundles, one per horizon.

**Cold start:** an online logistic regression that trains from realised outcomes, seeded
with priors that already know frequency and recency matter. This is why the system works on
a fresh clone with nothing trained — it degrades to a weaker predictor, not to nothing.

**Feature squashing:** `v / (1 + |v|)`. Raw features span many orders of magnitude; this
keeps one learning rate usable across all of them without a separate scaler.

**Blending:** `share = max(ml_confidence_floor, predictor.confidence())`. The model never
fully overrides the classical experts — when it is cold or wrong, the experts carry the
cache. This is the "do not make ML the whole algorithm" point from the build plan, in code.

---

## 11. How the benchmark produces the numbers — `bench.rs`

```
stream = generator.take(N)              # generated ONCE, fixed seed
for each policy:
    replay the identical stream
    charge: rebuild cost + SLA penalty + memory rent
    memory rent = occupancy integrated over time, not sampled at the end
belady = offline optimum (needs the future; a ceiling, not a competitor)
```

**Three fairness properties to state out loud:**

1. **Same stream.** One generation, replayed. No policy gets luckier traffic.
2. **Same prices.** One `Pricing` struct for every policy.
3. **Memory charged over time.** Charging the final size would let a policy hold a huge pool
   all run and pay for the instant it ended on.

**AURA is allowed to resize; the baselines are not** — because they have no mechanism to.
It pays rent on every byte it chooses to rent, and still wins. If challenged: *deciding how
much memory to rent is the contribution, not a loophole.*

### Current results — 50,000 requests, 128 MB start, seed 42

| scenario | AURA rank | vs best baseline | vs LRU |
|---|---|---|---|
| steady_zipf | 1 / 5 | **+16.7%** | +33.2% |
| mixed_production | 1 / 5 | **+7.1%** | +16.7% |
| expensive_tail | 1 / 5 | **+5.0%** | +14.8% |
| scan_resistance | 2 / 5 | −0.1% (tie) | +3.4% |

Four scenarios, three clear wins, one statistical tie. The PS asks for three. Report the tie
as a tie — claiming a 0.1% win invites someone to re-run it with a different seed.

---

## 12. File map for the algorithms

| Question | File | Function |
|---|---|---|
| What is an object worth? | `aura-server/src/engine.rs` | `value_density` |
| Should we admit it? | `aura-server/src/engine.rs` | `put`, `victim_bar` |
| Who gets evicted? | `aura-server/src/engine.rs` | `make_room`, `resident_density` |
| What should we refresh? | `aura-server/src/engine.rs` | `refresh_candidates` |
| Which policy do we trust? | `aura-server/src/policy.rs` | `Bandit`, `Policy::utility` |
| What kind of traffic is this? | `aura-server/src/engine.rs` | `recompute_workload` → `WorkloadFeatures::classify` |
| Should we buy memory? | `aura-server/src/capacity.rs` | `report`, `maybe_apply`, `mrc` |
| Will it be reused? | `aura-server/src/predictor.rs` | `reuse`, `reuse_peek`, `OnlineLogistic` |
| What are the features? | `aura-core/src/features.rs` | `FeatureBuilder::transform`, `peek` |
| What does it cost? | `aura-core/src/config.rs` | `Pricing::regen_cost_usd` |
| How do we prove it? | `aura-server/src/bench.rs` | `run`, `run_aura`, `run_classical`, `run_belady` |
| Where does traffic come from? | `aura-sim/src/generator.rs` | `Generator::step`, `take` |
| Cheap counters | `aura-core/src/sketch.rs` | `DecayCounter`, `CountMinSketch`, `QuantilePair` |

---

## 13. The 90-second version

> Traditional caches keep what was touched most recently. That is a proxy for value, and a
> bad one: it will evict a four-cent AI query result to keep a free string lookup.
>
> We score every object as **dollars of backend work it will save, per dollar it costs to
> keep**. The numerator needs three things — how likely it is to come back, at three time
> horizons; what rebuilding it actually cost, measured by the application, split across CPU,
> GPU, database and paid API; and how much room it takes.
>
> Six classical policies run underneath as competing experts, and a bandit reweights them
> from realised savings, so the cache adapts as traffic changes instead of following one
> fixed rule. A detector classifies the traffic pattern in real time and admission tightens
> when it sees a scan.
>
> Separately, the cache asks every few seconds whether renting more memory would pay for
> itself, and resizes itself when the answer is yes.
>
> We prove it by replaying one identical request stream through LRU, LFU, GDSF and TinyLFU,
> charging every one of them the same prices including rent on the memory they hold. We are
> cheapest in three of four scenarios and tied in the fourth, up to 33% under LRU. Belady's
> offline optimum is shown as the ceiling nothing online can reach.
