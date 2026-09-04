# AURA — the scoring maths, term by term

Every quantity in the score, how it is actually computed, and the constant it uses.
Companion to `ALGORITHMS.md`, which gives the shape; this gives the arithmetic.

---

## The formula

`engine.rs::value_density`

```
                cost_usd  ×  expected_reuses  ×  tail_risk
value_density = ───────────────────────────────────────────
                          holding_cost
```

Units: **dollars saved per dollar spent holding**. It is dimensionless, which is the point —
a 40 KB SQL rollup and a 24 MB video transcode land on the same axis and can be ranked
against each other.

Four terms. Each is computed below.

---

## Term 1 — `cost_usd`: what rebuilding this actually cost

`config.rs::Pricing::regen_cost_usd`

```
cost_usd = cpu_ms      × 0.0000000116        # $/ms of CPU
         + gpu_ms      × 0.000000255         # $/ms of GPU   (22× CPU)
         + db_ms       × 0.0000000320        # $/ms of DB    (2.8× CPU)
         + (network_bytes / 1e9) × 0.09      # $/GB egress
         + api_cost_usd                      # passed straight through
```

**These are not estimates the cache makes.** The application measures its own work and
reports a `CostVector` on the write. `apps/common/costing.py` builds it; `aura_client.py`
sends it.

**Why a vector and not one number.** Two objects that both take 300 ms to rebuild:

| | cpu_ms | gpu_ms | db_ms | cost |
|---|---|---|---|---|
| Recommendation ranking | 100 | 200 | 0 | $0.0000522 |
| Analytics rollup | 20 | 0 | 280 | $0.0000092 |

**5.67× apart at identical latency.** Any policy that reasons about "latency" cannot see
this. That gap is the entire premise of the problem statement, and this is where it enters
the maths.

The prices are Pricing defaults in `config.rs`, mirrored in `training/aura_train/config.py`
so the model and the engine price identically. A parity test enforces it.

---

## Term 2 — `expected_reuses`: how many rebuilds this will actually save

```
expected_reuses = P(reuse within 10 s)  × 0.50 × 6
                + P(reuse within 60 s)  × 0.35 × 2
                + P(reuse within 600 s) × 0.15
```

`horizon_weights = [0.50, 0.35, 0.15]`, `EngineConfig` default.

**Why three horizons.** "Will this be reused" is meaningless without a time bound. An object
reused in 8 seconds is worth far more than one reused in 8 minutes, because in the meantime
it occupies bytes you are paying rent on.

**Why the 6× and 2× multipliers.** They convert a *probability* into an *expected count*. If
a key has a 40% chance of being hit in the next 10 seconds, over a 60-second residency that
is roughly six independent 10-second windows — so ~2.4 expected hits, not 0.4. Without this,
short-horizon reuse would be systematically undervalued against long-horizon reuse.

**Where the three probabilities come from** — `predictor.rs`:

- **Trained bundle loaded:** a gradient-boosted tree walker, one bundle per horizon. Pure
  Rust, no ML runtime in the serving path.
- **Not trained (your current state):** an online logistic regression.

```
P(reuse) = sigmoid( bias + Σ wᵢ · squash(featureᵢ) )
squash(v) = v / (1 + |v|)          # keeps one learning rate usable across features
                                    # that span many orders of magnitude
```

Seeded with priors that already know what matters, so a cold cache is not a blind one:

```
freq_1m  +0.55     ewma_fast  +0.40     log_age_ms            −0.18
freq_5m  +0.30     trend      +0.25     log_inter_arrival_ms  −0.22
bias     −0.40
```

It updates by gradient descent on realised outcomes: `w -= lr · (p − y) · squash(f)`.

---

## Term 3 — `tail_risk`: the one that needs explaining

```
tail_risk = 1 + min(cost_variance_ratio, 4) × 0.35
```

`tail_risk_lambda = 0.35`, capped at 4 so one pathological outlier cannot dominate. Range:
**1.0 to 2.4**.

### What `cost_variance_ratio` is

`features.rs`, feature index 11:

```
cost_variance_ratio = regen_p95 / max(regen_p50, 1.0)
```

The p95 and p50 of **rebuild latency for this (application, object_type) group** — not for
this key. Grouping at the operation level is what keeps the cost model small: the engine
does not fit a regression per key, it learns the shape of each *operation class* and lets
the per-key features carry the rest.

### Why it belongs in the score

Consider two operations, both averaging 200 ms:

| Operation | p50 | p95 | ratio | tail_risk |
|---|---|---|---|---|
| Cached-friendly lookup | 190 ms | 210 ms | 1.1 | 1.39 |
| Cold analytics scan | 80 ms | 900 ms | 11 → capped 4 | **2.40** |

Both cost the same *on average*. But when the second one misses, sometimes a user waits
0.9 seconds. **A miss on a high-variance operation is a worse event than a miss on a
predictable one, even at equal mean cost**, because the pain lands in the tail — exactly
where your SLA penalty lives, and exactly what a p95 latency target measures.

`tail_risk` makes the cache willing to pay up to 2.4× more to keep an object whose rebuild
is unpredictable. It is buying insurance against the tail, not against the average.

**In one line for the pitch:** *"we weight by how erratic the rebuild is, because a miss you
can't predict the cost of is worse than a miss you can."*

### How p50 and p95 are tracked without storing samples

`sketch.rs::QuantilePair` — a Frugal-style streaming estimator, two floats per group:

```
observe(x, lr = 0.02):
    step50 = lr × max(p50, 1)
    p50 += (x > p50) ?  step50 × 0.50  :  −step50 × 0.50
    step95 = lr × max(p95, 1)
    p95 += (x > p95) ?  step95 × 0.95  :  −step95 × 0.05
    if p95 < p50: p95 = p50
```

The asymmetry is the trick. For p50 the step is symmetric (0.50 / 0.50), so it settles where
half the samples fall above. For p95 an upward move is 19× larger than a downward one
(0.95 / 0.05), so it settles where 5% fall above. **Two floats per operation class, no
histogram, no sample buffer.** Read *before* the current observation is folded in, so a
value is never compared against a quantile it already moved.

---

## Term 4 — `holding_cost`: the denominator

`config.rs::Pricing::holding_cost_usd`

```
holding_cost = (bytes / 1e9) × 0.0225 × (60000 / 3600000)
             = (bytes / 1e9) × 0.0225 × (1/60)
```

$0.0225 per GB-hour, evaluated over a 60-second window. A 1 MB object costs
**$0.000000393** to hold for a minute.

**Why divide.** Without it, a 24 MB media file with the same reuse and cost as a 40 KB query
result scores identically — while displacing ~600 of them. Dividing by size is what makes
the score a *density* rather than a total, and density is the right currency when the
constraint is space.

---

## Worked example

Analytics rollup, 40 KB, 280 ms of DB time, moderately hot:

```
cost_usd        = 280 × 0.0000000320                    = $0.00000896
reuse           = [0.42, 0.61, 0.78]
expected_reuses = 0.42×0.50×6 + 0.61×0.35×2 + 0.78×0.15 = 1.26 + 0.427 + 0.117 = 1.804
cost_variance   = 900/80 = 11.25 → capped at 4
tail_risk       = 1 + 4 × 0.35                          = 2.40
holding_cost    = (40960/1e9) × 0.0225 × (1/60)         = $0.00000001536

value_density   = (0.00000896 × 1.804 × 2.40) / 0.00000001536 ≈ 2526
```

Media variant, 4 MB, GPU-heavy, colder, predictable:

```
cost_usd        = 210 × 0.000000255 + 0.0021 (API)      = $0.00215355
expected_reuses = 0.11×0.50×6 + 0.19×0.35×2 + 0.31×0.15 = 0.33 + 0.133 + 0.047 = 0.510
tail_risk       = 1 + 1.2 × 0.35                        = 1.42
holding_cost    = (4194304/1e9) × 0.0225 × (1/60)       = $0.00000157

value_density   = (0.00215355 × 0.510 × 1.42) / 0.00000157 ≈ 991
```

**The rollup wins despite being 240× cheaper to rebuild and far less "expensive-looking",
because it is 100× smaller and reused 3.5× more.** That is the trade-off LRU cannot make and
GDSF makes only crudely.

---

## Admission — the decision, in full

`engine.rs::put`

```
1.  cost      = price(measured CostVector)         # measured, not guessed
2.  features  = feature_builder.transform(event)   # 16 numbers, ~1 µs
3.  reuse     = predictor.reuse(features)          # 3 probabilities
4.  density   = value_density(...)

5.  bar       = victim_bar(size)                   # what would have to die
6.  threshold = bar × 1.10                         # admission_margin

    if size > capacity / 4          -> Reject   one object may not own a quarter of the pool
    if regime == Scan
       and reuse[10s] < 0.15
       and key unseen in window     -> Reject   single-touch scan signature
    if cache has room               -> Admit    refusing costs a certain miss to avoid a
                                                hypothetical one
    if density ≥ threshold          -> Admit
    if reuse[60s] > 0.75            -> Admit    near-certain reuse overrides the bar
    else                            -> Reject
```

### `victim_bar` — what it would cost to make room

```
victim_bar(incoming_bytes):
    if the cache already fits it: return 0
    sample 32 resident keys uniformly at random
    return min( resident_density(k) for k in sample )
```

**The rule:** admit only if the arrival is worth more than the weakest thing it would
displace, plus a 10% margin. The margin stops churn — swapping an object for a marginally
better one costs two backend calls to gain almost nothing.

> **This gate was broken until recently.** `value_density` and the old eviction score were
> computed by different formulas in different units — arrivals scored ~5,000, the bar scored
> ~0.5. Every arrival cleared it: **42,614 admissions, 0 rejections**. Both sides now end in
> `value_density`; the gate refuses ~46% of arrivals. Worth saying in the pitch — it is a
> real instrumentation-found-a-real-bug story.

---

## Eviction — the decision, in full

`engine.rs::make_room`

```
make_room(bytes_needed, incoming_density):
    sweep expired entries first             # free wins before costly ones
    while still short of room:
        sample 32 resident keys uniformly
        victim = argmin resident_density
        if victim.density ≤ incoming_density × 1.05:
            evict victim
        else:
            stop        # nothing in this sample is worse than what is arriving
```

### Why sample 32 instead of keeping a sorted heap

A heap costs O(log n) on **every access** to maintain, on the hot path, for a cache holding
hundreds of thousands of objects. Sampling costs nothing until an eviction is actually
needed, and then O(1).

The accuracy argument: with a sample of *k*, the expected percentile of the minimum is
`1/(k+1)`. At k = 32 that is the **3rd percentile** — the victim is worse than 97% of
resident objects. Redis's own `allkeys-lru` relies on exactly this result with a default
sample of 5. Thirty-two is generous.

### `resident_density` — putting a resident object on the same scale

```
resident_density(key):
    f            = features.peek(key)          # read counters; do NOT record a touch
    expert       = Σ mixture[i] × policy[i].utility(f, age)
    expert_reuse = expert / (1 + expert)       # squash a ranking score into [0,1)
    model_reuse  = predictor.reuse_peek(f)
    share        = max(ml_confidence_floor = 0.20, predictor.confidence())
    reuse        = expert_reuse × (1 − share) + model_reuse × share
    return value_density(f, reuse, size, cost) × ttl_remaining_frac
```

Three things to notice:

1. **`peek`, not `transform`.** Scoring an object is not using it. Recording a touch would
   make every eviction sweep look like traffic and corrupt the frequency features.
2. **The experts supply a reuse estimate, not a final answer.** They rank; the economics are
   applied afterwards by the same `value_density`. This is what keeps admission and eviction
   commensurable.
3. **`ml_confidence_floor = 0.20`** means the model never contributes less than 20% and
   never fully overrides the classical policies. *Do not make ML the whole algorithm* — when
   the model is cold or wrong, the experts carry the cache.

---

## Thompson sampling, in detail

`policy.rs::Bandit`

### The problem it solves

No single eviction policy is best. LRU wins on recency-dominated traffic, LFU on stable
skew, GDSF when size and cost vary, TinyLFU under scans. Picking one at build time is the
static rule the problem statement explicitly rules out.

So run all six, and learn which to trust — continuously.

### The model

Each policy is an arm of a multi-armed bandit with a **Beta(α, β) posterior** over "how
often does trusting this arm pay off". Beta is the conjugate prior for a Bernoulli reward,
so the update is arithmetic rather than inference: no optimiser, no gradients.

```
initial:  α = 1, β = 1  for all six      # Beta(1,1) = uniform: no opinion
```

### Update — after every admission decision

```
credit(policy, reward):                  # reward ∈ [0,1]
    α[policy] += reward
    β[policy] += 1 − reward
```

Reward comes from realised economics, not hit rate:

```
on Admit:   reward = clamp( density / (threshold × 4), 0, 1 )
                     # how far it beat the bar it had to clear
on Reject:  reward = (1 − reuse[60s]) × 0.7
                     # how confidently it was refused
```

**This matters.** Grading arms on hit rate would reward keeping cheap, frequently-hit
objects. Grading on realised value rewards keeping the objects that actually save money.

### Draw — picking an arm

```
sample_arm():
    for each arm i:
        mean   = α / (α + β)
        spread = 1 / sqrt(α + β)          # posterior uncertainty
        draw   = mean + normal() × spread × 0.5
    return argmax draw
```

**This is the elegant part.** The sample width `1/√(α+β)` shrinks as evidence accumulates.

- An arm tried twice has a wide posterior — it draws high sometimes and gets explored.
- An arm tried ten thousand times has a narrow posterior — it draws near its mean.

Exploration is **automatic and self-limiting**. There is no ε to tune, no schedule to decay.
An arm stops being explored exactly when there is enough evidence about it, and starts again
the moment its performance changes.

### The mixture — what the dashboard shows

```
mixture():
    wᵢ = α/(α+β) + 0.08                   # exploration floor
    normalise so Σw = 1
```

The `+0.08` floor guarantees every policy retains some weight. Without it, an arm that
performs badly early can be driven to zero and never recover, even after the workload shifts
to favour it.

### Decay — what makes it *adaptive* rather than *trained*

```
decay(0.995) every controller tick (~2 s):
    α = 1 + (α − 1) × 0.995
    β = 1 + (β − 1) × 0.995
```

Both parameters relax toward the uniform prior. Effective memory is ~200 ticks, roughly
**7 minutes** of evidence.

**Without decay the system stops adapting.** After an hour, α and β are in the thousands and
a new observation moves the posterior by ~0.1%. An arm that was right during the morning
would outvote the afternoon indefinitely. Decay is precisely the line that makes "adapts at
runtime" true rather than "adapted once, early".

**Regret** is tracked as a decayed running gap between the best arm's mean and the chosen
arm's mean, and shown on the dashboard as evidence the bandit is converging.

### The demo moment

Press **Cost spike**. Rebuild costs jump 6–8×. Within a few seconds `cost_aware` and `gdsf`
start earning higher rewards, their α climbs, and the mixture bar visibly shifts. Nothing
was reconfigured. **That is runtime adaptation, on screen, in about two seconds.**

---

## The 16 features, and how each is computed

`features.rs::transform`. Every one is O(1) with two floats of state or fewer.

| # | Feature | Computation | Constant |
|---|---|---|---|
| 0 | `log_age_ms` | `ln(1 + Δt)` since last access | — |
| 1 | `log_inter_arrival_ms` | `ln(1 + ewma_gap)`, EWMA of gaps | α = 0.3 |
| 2 | `freq_1m` | `prev × exp(−Δt/τ)`, then `+1` | τ = 60 s |
| 3 | `freq_5m` | same | τ = 300 s |
| 4 | `freq_1h` | same | τ = 3600 s |
| 5 | `ewma_fast` | half-life decay | 5 s |
| 6 | `ewma_slow` | half-life decay | 60 s |
| 7 | `trend` | `ln((fast + ε)/(slow + ε))` | ε = 1e-6 |
| 8 | `acceleration` | `trend − prev_trend` | — |
| 9 | `log_size_bytes` | `ln(1 + bytes)` | — |
| 10 | `log_regen_p50_ms` | `ln(1 + p50)` per operation class | — |
| 11 | `cost_variance_ratio` | `p95 / max(p50, 1)` → **tail risk** | lr = 0.02 |
| 12 | `regen_cost_usd` | the priced `CostVector` | — |
| 13 | `ttl_remaining_frac` | `1 − age/ttl`, clamped | — |
| 14 | `cache_pressure` | `used_bytes / capacity` | — |
| 15 | `app_id` | stable hash of the application name | — |

### Why exponential decay rather than counters in windows

A sliding window needs a timestamp list per key. Exponential decay needs **one float**:

```
freq = prev_value × exp(−Δt / τ)   then   += 1
```

Decay is applied lazily at read time, so a key nobody touches costs nothing to maintain.
Sixteen features across hundreds of thousands of keys, and `KeyState` is 88 bytes.

### Why `trend` is a log ratio

`ln(fast/slow)` is **symmetric around zero**: doubling gives +0.69, halving gives −0.69. A
plain ratio would give 2.0 and 0.5 — the same change in opposite directions producing
wildly different magnitudes, which a linear model cannot use sensibly.

`trend` and `acceleration` are the features **no classical policy has**. LRU sees the last
access. LFU sees the count. Neither can tell a key on the way up from a key on the way down
at the same instantaneous frequency. That is what catches a flash crowd while it is forming
rather than after it has passed.

---

## Constants, all in one place

`engine/config/default.toml`, overridable by `AURA__SECTION__KEY` environment variables.

| Constant | Value | What it controls |
|---|---|---|
| `candidate_sample` | 32 | eviction sample size |
| `admission_margin` | 1.10 | how much better an arrival must be to displace |
| `ml_confidence_floor` | 0.20 | minimum model influence |
| `horizon_weights` | [0.50, 0.35, 0.15] | 10 s / 60 s / 600 s importance |
| `tail_risk_lambda` | 0.35 | how much variance is worth paying for |
| `refresh_ttl_threshold` | 0.15 | refresh when under 15% TTL left |
| `bandit.exploration` | 0.08 | mixture floor per arm |
| `roi_threshold` | 1.25 | scale up only above 1.25× return |
| `slo_p95_ms` | 150 | latency target for SLA penalties |
| `quantile_lr` | 0.02 | p50/p95 estimator step size |

None of these are tuned per scenario. The same values produce every benchmark number.
