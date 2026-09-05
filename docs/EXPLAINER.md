# AURA, explained

Everything the system decides, why it decides it that way, and the exact arithmetic behind
each decision. Written to be argued with: every formula here is copied from the code, and
the file and function it comes from is named so you can check it.

---

# 1. What the problem statement is actually asking for

The brief is "application scaling", and the naive reading is *make the cache bigger*. That
reading is wrong, and understanding why is the whole project.

## The economics

A cache exists to avoid work. The work it avoids has a price: CPU time, GPU time, database
time, bytes over the network, dollars paid to third-party APIs, and the compounding cost of
being slow when you promised not to be. The cache itself also has a price: memory is rented
by the gigabyte-hour whether or not anything in it is ever read again.

So a cache is a portfolio. Every resident object is a position: you are paying rent on it in
the expectation that it will save you a rebuild. And the question a cache should be asking,
on every admission and every eviction, is the question any portfolio manager asks:

> Is this position earning its rent, and is it earning more than the position I would have
> to sell to buy it?

Conventional caches never ask this. They cannot, because they are not told the price of
anything.

## Why this is a scaling problem and not a caching trick

At small scale the answer to a full cache is "buy more memory". At scale that stops working
for two reasons.

The first is that memory is the expensive tier. Doubling a cache doubles a fixed hourly bill
to buy a hit-rate improvement that is *sublinear* — the second gigabyte always buys less than
the first, because the head of a Zipf distribution is already resident. There is a point
where the next gigabyte costs more than the misses it prevents, and no policy that only looks
at hit rate can find that point.

The second is that a cache under memory pressure makes *choices*, and the quality of those
choices is what determines whether more memory is needed at all. A cache that evicts well at
128MB can outperform a cache that evicts badly at 512MB. Buying memory to compensate for a
bad eviction policy is the most expensive way to fix a software problem.

So "adaptive scaling" properly understood has two halves, and this system implements both:

- **Decide well inside the memory you have.** Rank objects by what they are worth, not by
  when they arrived.
- **Decide whether to buy more.** Price the next block of memory against the misses it
  would prevent, and buy it only if it pays.

---

# 2. Where LRU, LFU and GDS fall short

Each of these is a good algorithm. Each is answering a narrower question than the one that
matters, and each has a specific, nameable blind spot.

## The scene they all fail

The cache is full. A product-lookup response arrives: 48 KB, rebuilt from a primary-key
select in 2ms. To make room, something must go. The candidate is an analytics rollup: also
48 KB, rebuilt by a 412ms aggregation across three tables.

Same size. Similar access frequency. One costs **two hundred times more** to rebuild.

Here is what each policy does.

## LRU — ranks on arrival order

**Its rule:** evict whatever was touched longest ago.

**What it can see:** the order of accesses. That is all.

**Where it fails:** it evicts the rollup if the rollup was touched one second earlier. It
does not know that one object is free to rebuild and the other is not, because nobody ever
told it and its data structure has nowhere to put that information. Recency is a *proxy* for
future value, and it is a reasonable proxy right up until the objects differ in cost — at
which point it is silently, confidently wrong.

Its second failure is scans. A batch job reading a million rows once will walk the entire
resident set out of the cache, because every one of those rows is, briefly, the most recently
used thing in it. LRU has no way to distinguish "popular" from "recent", so a single
sequential pass destroys a working set that took an hour to build.

## LFU — ranks on historical frequency

**Its rule:** evict whatever has been touched least often.

**What it can see:** counts. Nothing about cost, nothing about size, and critically nothing
about *when* those counts accrued.

**Where it fails:** cache pollution. An object that was extremely popular this morning keeps
its count all afternoon and cannot be evicted, while a genuinely hot new object cannot
accumulate enough history to displace it. Every real LFU deployment needs an aging or decay
mechanism bolted on, which is an admission that raw frequency was the wrong signal.

It shares LRU's cost-blindness completely: a hundred cheap hits outrank ten expensive ones.

## GDS and GDSF — the closest to right, and still not enough

Greedy-Dual Size is the only one of the three that knows about cost. Its rank is
`cost / size`, plus an inflation term `L` that rises as objects are evicted, so that an
expensive object cannot stay resident forever on the strength of one ancient decision. GDSF
adds frequency: `frequency × cost / size`.

**This is genuinely good, and it is the baseline we take most seriously.**

**Where it still falls short — four specific places:**

1. **It uses observed frequency as a stand-in for future reuse.** Frequency is backward
   looking. A key that was hit five times in the last minute and is now dying, and a key hit
   five times in the last minute and accelerating, score identically. There is no trend term,
   no acceleration term, no notion of a reuse *horizon*.

2. **It has no notion of variance.** A rebuild that reliably takes 50ms and one that averages
   50ms with a 2-second tail are priced the same. But the second one is the one that breaks
   your p99, and it is worth more to keep resident precisely because its cost is erratic.

3. **It cannot value a hit differently for different callers.** One shared number for the
   whole system. An application that would rather pay for memory than ever be slow, and one
   that would rather be slow than pay, get identical treatment.

4. **It is fixed.** GDSF's weighting between frequency and cost is a constant chosen by
   whoever wrote it. It does not change when the workload changes. A policy tuned for a
   cost-skewed workload is the wrong policy during a flash crowd, and GDSF cannot tell the
   difference or do anything about it.

## What every one of them shares

None of them has a **feedback loop**. LRU cannot discover that it is wrong about this
workload. It has no mechanism for being wrong, because it has no prediction — only a rule.
When the workload changes, LRU keeps doing exactly what it did before, with the same
confidence, forever.

## What we do instead, mapped one to one

| Their limitation | What this system does |
|---|---|
| Ranks on a proxy (recency, frequency) | Ranks on **expected dollars saved per dollar of rent** |
| Blind to rebuild cost | Cost is measured by the application and sent with every write |
| Blind to cost variance | `cost_variance_ratio` is a feature, and tail risk multiplies value |
| Backward-looking frequency | Three forward reuse probabilities, at 10s / 60s / 600s |
| One fixed weighting | Six experts, weighted by a Beta posterior over realised outcomes |
| Cannot detect being wrong | Every decision is graded 60 seconds later, on what actually happened |
| One policy for everyone | Per-application profiles change the arithmetic per caller |
| Scans destroy the working set | Scan signature is refused admission, not admitted and then regretted |
| Nothing to say about pool size | ROI on the next block of memory decides whether to grow |

---

# 3. The exact formulas

Everything in this section is copied from the source. File and function are named.

## 3.1 The 24 features

`engine/aura-core/src/features.rs`. A fixed 24-slot `[f64; 24]`, never heap-allocated on the
hot path. Slots 0–15 come from per-key counters; slots 16–23 come from `signals.rs` and carry
per-application distributions.

| # | Name | What it is |
|---|---|---|
| 0 | `log_age_ms` | log time since insertion |
| 1 | `log_inter_arrival_ms` | log gap between the last two accesses |
| 2 | `freq_1m` | accesses in the last minute |
| 3 | `freq_5m` | accesses in the last five minutes |
| 4 | `freq_1h` | accesses in the last hour |
| 5 | `ewma_fast` | fast exponentially-weighted access rate |
| 6 | `ewma_slow` | slow exponentially-weighted access rate |
| 7 | `trend` | `ewma_fast − ewma_slow`; positive means rising |
| 8 | `acceleration` | change in trend; positive means rising *faster* |
| 9 | `log_size_bytes` | log object size |
| 10 | `log_regen_p50_ms` | log median rebuild time |
| 11 | `cost_variance_ratio` | rebuild cost spread over its mean — the tail-risk signal |
| 12 | `regen_cost_usd` | what one rebuild costs, in dollars |
| 13 | `ttl_remaining_frac` | fraction of TTL left, 1.0 fresh, 0.0 expired |
| 14 | `cache_pressure` | how full the pool is, 0.0–1.0 |
| 15 | `app_id` | which application, hashed deterministically |
| 16 | `size_percentile` | how large this object is *for this application* |
| 17 | `cost_percentile` | how expensive it is for this application |
| 18 | `cost_variance_ratio_app` | how erratic costs are for this application overall |
| 19 | `log_reuse_distance` | how many distinct keys were seen between the last two hits |
| 20 | `burstiness` | whether this key arrives in clumps or evenly |
| 21 | `novelty_rate` | fraction of recently-seen keys that were never seen before — the scan detector |
| 22 | `hour_sin` | time of day, as a continuous cycle |
| 23 | `hour_cos` | the other half of that cycle |

**Every feature is computed from strictly prior accesses.** The access being scored is never
included in its own features — otherwise the model trains on the answer, and every offline
number is a lie.

## 3.2 The score: value density

`Engine::value_density`, `engine/aura-server/src/engine.rs`.

```
expected_reuses = r₁₀ · w₀ · 6  +  r₆₀ · w₁ · 2  +  r₆₀₀ · w₂

risk            = 1 + min(cost_variance_ratio, 4) · tail_risk_lambda

sla_weight      = risk · max(profile.sla_weight, 0.05)

value           = regen_cost_usd · expected_reuses · sla_weight

holding_cost    = price_of(size_bytes held for 60s)

value_density   = value / holding_cost
```

and for something already resident, `Engine::resident_density` applies one more term:

```
resident_density = value_density · max(ttl_remaining_frac, 0.05)
```

**Reading it in words:** *how many dollars of backend work this object is expected to save,
per dollar of memory rent it will consume.* It is a pure ratio, so it is comparable across
objects of any size, any cost and any application. That is what makes a single number able to
rank the whole pool.

**Why each term is there:**

- `r₁₀, r₆₀, r₆₀₀` are reuse probabilities at 10, 60 and 600 seconds. Three horizons and not
  one, because "will be used again" is meaningless without a deadline — an object needed in
  ten minutes is worth far less right now than one needed in ten seconds.
- The multipliers **6, 2, 1** convert a probability into an expected *count*. If something has
  a 50% chance of being needed within 10 seconds, it will likely be needed several times in
  the next minute; the near horizon is worth more per unit of probability.
- `min(cost_variance_ratio, 4)` — clamped, because one pathological outlier should not make an
  object immortal.
- `max(..., 0.05)` on the SLA weight and on the TTL fraction — floors, so that an object with
  a near-dead TTL still has *some* value rather than dropping to exactly zero and becoming
  indistinguishable from garbage.
- Holding cost is priced over a fixed **60-second** window, so the denominator is a consistent
  unit of rent rather than varying with each object's TTL.

## 3.3 The reuse probabilities: how `r₁₀, r₆₀, r₆₀₀` are produced

This is the hybrid, and it is the part with the most thought in it.
`Engine::blended_reuse`.

```
expert       = Σᵢ  mixture[i] · Policyᵢ.utility(features, age)

expert_reuse = expert / (1 + max(expert, 0))        # squash into [0, 1)

share        = clamp( max(ml_confidence_floor, predictor.confidence()), 0, 1 )

rₕ           = expert_reuse · (1 − share)  +  model_rₕ · share
```

Two independent estimators, blended by how much the learned one deserves to be trusted:

- **The experts** are six cheap heuristics. They need no training and are never badly wrong,
  which is what makes the first sixty seconds of a cold cache sane.
- **The model** is a gradient-boosted tree (or a linear model) trained on retired decisions.
  It is better once it has evidence and worthless before that.
- **`share`** is the model's own confidence, ramping over its first 5,000 samples, floored at
  `ml_confidence_floor` (0.20) so the model always gets some voice.

### The six experts

`Policy::utility`, `engine/aura-server/src/policy.rs`. These are the exact expressions:

| Expert | Utility |
|---|---|
| LRU | `1 / (1 + age_ms/1000)` |
| LFU | `freq_1h` |
| GDSF | `(freq_5m · regen_cost_usd) / log_size_bytes` |
| TinyLFU | `0.7 · freq_1m + 0.3 · freq_5m` |
| CostAware | `regen_cost_usd / log_size_bytes` |
| TrendAware | `ewma_fast · (1 + max(trend, 0)) + max(acceleration, 0)` |

Note what this means: **the baselines we compete against are also inside us, as opinions.**
When LRU is the right answer for the current workload, the bandit discovers that and LRU's
weight rises. We do not beat LRU by being different from it; we beat it by knowing *when* to
be it.

## 3.4 The bandit: how `mixture[i]` is learned

`Bandit`, `engine/aura-server/src/policy.rs`.

Each expert carries a Beta posterior `Beta(αᵢ, βᵢ)` over "how often is this expert right".

**Selection (Thompson sampling):**

```
sampleᵢ = αᵢ/(αᵢ+βᵢ)  +  gaussian_jitter · 1/√(αᵢ+βᵢ)
```

The jitter shrinks as evidence accumulates — an expert with little history is sampled widely
(explore), one with a lot is sampled near its mean (exploit). That is the exploration
schedule, and nobody has to tune it: it falls out of the posterior's own shape.

**Credit — and this is the part that matters most.** The reward is not the score the engine
predicted. It is what *actually happened*, looked at 60 seconds later:

`DecisionRecord::realised_reward`, `engine/aura-server/src/feedback.rs`

| What we did | What happened | Reward |
|---|---|---|
| Admitted | it was reused | `0.55 + 0.45 · clamp(cost/holding ÷ 10, 0, 1)` |
| Rejected | it was not reused | `0.80` |
| Admitted | it was never reused | `0.15` |
| Rejected | it *was* wanted | `0.05` |

Read the ordering: **a correct refusal (0.80) scores higher than an average useful admission**,
because saying no to the right things is most of what a good cache does. The worst outcome by
far is refusing something that turned out to be wanted. And a useful admission is rewarded in
proportion to the *return* it produced — a rebuild worth ten times its rent saturates the
scale, so one enormous object cannot capture the posterior.

**Decay:** every posterior is multiplied by `0.995` per tick. Evidence has a half-life. Without
this, an expert that won for the first hour would outvote fresh evidence forever, and the
system would stop adapting exactly when a workload changed.

**Timeline:** decide at `t₀` → settle at `t₀+60s` (reward the bandit, label the model) →
retire at `t₀+600s` (emit a training row). The cache cannot grade its own homework, because it
does not hold the pen until the future has happened.

## 3.5 Admission: what gets in

`Engine::put`. In order:

```
victim_bar  = min( resident_density(k) for 32 randomly sampled resident k )

share_penalty = 1 + (share − max_pool_share) · 4      if over its share, else 1

threshold   = victim_bar · admission_margin · share_penalty
```

Then:

| Test | Outcome |
|---|---|
| `size_bytes > capacity/4` | **reject** — `object_exceeds_quarter_capacity` |
| Scan signature (below) | **reject** — `single_touch_scan_signature` |
| pool empty, or `density ≥ threshold` | **admit** — `density_above_threshold` |
| `r₆₀ > 0.75` | **admit** — `high_near_term_reuse` (a rescue for a very confident prediction) |
| otherwise | **reject** — `density_below_threshold` |

**The critical design choice:** the bar is `victim_bar` — *the object that would have to be
evicted to make room for this one* — and not a running average of arrivals. Comparing an
arrival against the mean of other arrivals rejects the bottom half of the stream no matter
how good it is, and leaves capacity unused. Comparing it against the actual victim asks the
only question that matters: **is this trade an improvement?**

The scan test:

```
scan_suspect = regime == Scan
            && regime_confidence ≥ 0.5
            && r₁₀ < 0.15
            && key not already in the admission window
```

All four conditions. The confidence floor exists because the regime detector flaps at low
confidence on ordinary traffic, and refusing admissions on a 33%-confident guess costs hit
rate for nothing.

## 3.6 Eviction: what goes

`Engine::make_room`.

```
loop while the pool cannot fit the arrival:
    floor = (capacity / resident_applications) · app_floor_fraction

    sample 32 residents at random
    partition into protected / unprotected:
        protected = belongs to a different application than the arrival
                    AND that application holds ≤ floor

    victim = weakest unprotected, else weakest protected
    evict if victim.density ≤ incoming_density · 1.05
```

**Why sampling and not a sorted structure:** finding the true minimum means scanning every
resident object, which is O(n) per eviction and unaffordable at request rate. 32 random
samples land within a few percent of the true minimum. This is the standard result behind
Redis's own sampled LRU, and the same reasoning applies at a higher scoring cost.

**Why the score is recomputed at eviction time, never at insert time:** this is the difference
between dynamic and static eviction and it deserves stating plainly. LRU's decision about an
object is fixed the moment it arrives — its position in the list is its fate, changed only by
being touched. Here, `resident_density` is called fresh on every sampled candidate, with the
object's *current* age, *current* frequency, *current* TTL remaining, and the pool's *current*
pressure. An object that looked excellent when admitted and has since gone cold is re-priced
as cold at the moment the question is asked.

## 3.7 How user tuning changes the formulas

`engine/aura-server/src/profiles.rs`, via `PUT /v1/applications/{name}/profile`.

The profile changes the arithmetic *around* the prediction and never the prediction itself —
`reuse` arrives at `value_density` exactly as the model produced it. Tuning changes what you
value, not what the system believes is true.

**The seven knobs:**

| Knob | Where it lands | Effect |
|---|---|---|
| `horizon_weights` | `w₀, w₁, w₂` in `expected_reuses` | shifts value between "needed in 10s" and "needed in 10 minutes" |
| `tail_risk_lambda` | `risk = 1 + min(cvr,4)·λ` | how much extra an erratic rebuild is worth. 0 ignores variance |
| `sla_weight` | multiplies `value` | how loudly this application says being slow costs it more |
| `admission_margin` | multiplies `threshold` | 1.0 admits anything at least as good; above 1.0 protects incumbents |
| `soft_ttl_fraction` | refresh-ahead trigger | where in an object's life the background rebuild starts |
| `max_pool_share` | `share_penalty` | share of the pool before this application's arrivals must clear a higher bar |
| `objective` | preset for all of the above | one word instead of six numbers |

**The three presets** (`preset_from`):

| Objective | horizon_weights | tail_risk_lambda | sla_weight | admission_margin | soft_ttl_fraction |
|---|---|---|---|---|---|
| `cost` (default) | from config `[0.50, 0.35, 0.15]` | config | 1.0 | 1.10 | 0.80 |
| `latency` | `[0.6, 0.3, 0.1]` | 0.6 | **2.5** | 1.10 | 0.65 |
| `origin` | `[0.35, 0.35, 0.3]` | config | 1.0 | **1.0** | 0.75 |

Read them as sentences:

- **`latency`** — weight the near horizon heavily, value erratic rebuilds highly, charge SLA
  breaches at 2.5×, and start refreshing at 65% of TTL rather than 80%. *"I would rather pay
  for memory than ever be slow."*
- **`origin`** — flatten the horizons so long-lived objects keep their value, and drop the
  admission margin to 1.0 so anything at least as good as the victim gets in. *"Protect my
  database; I care less about which particular objects survive."*
- **`cost`** — the default. Optimise dollars.

Touching any individual knob sets `objective: custom`, so the console never claims a preset
is in force when it has been edited.

There is also a **global override**: `POST /v1/policy/override` forces one named policy for
the whole engine, and `resident_density` returns that policy's raw utility before any of the
economics. It exists so you can demonstrate, live, what pure LRU would have done — and it is
the honest way to make the comparison, because it is the same engine and the same traffic.

## 3.8 The scaling decision

`CapacityController::report`, `engine/aura-server/src/capacity.rs`. Runs on the 250ms tick.

**Step 1 — fit a miss-ratio curve.** From the hit rate actually observed at the current size,
solve for the constant of a saturating curve:

```
k     = current_bytes / (1/hit_rate − 1)
h(b)  = b / (b + k)
```

This is the standard concave hit-rate-versus-size shape. One observed point pins it.

**Step 2 — price the next block.**

```
Δ            = h(current + step) − h(current)
requests/hr  = requests / elapsed_seconds × 3600
avg_miss_usd = ledger.backend_usd / misses
savings      = Δ × requests_per_hour × avg_miss_usd
cache_cost   = extra_gb × cache_gb_hour_usd
roi          = savings / cache_cost
```

**Step 3 — decide.**

| Decision | Condition |
|---|---|
| **ScaleUp** | `roi ≥ 1.25` and host has more than `2 × step` free |
| **ScaleOut** | `roi ≥ 1.25` but the host has ≤ `step` free — *more memory pays, this machine cannot supply it* |
| **ScaleDown** | `pressure < 0.55` **and** `evictions > 0` **and** `requests ≥ 5000` |
| **Hold** | everything else, including manual mode |

**The two guards on ScaleDown are the interesting part.** Low pressure on a cache that has
never evicted anything is not evidence that memory is unwanted — it is evidence that nothing
has arrived yet. Shrinking there means an idle engine gives away the pool it is about to
need, then pays to buy it back under load, which is the worst possible time. So it refuses to
shrink until it has seen real contention (`evictions > 0`) and enough traffic to characterise
the workload at all (`requests ≥ 5000`).

`host_available` is the **smaller** of what the operator budgeted and what the machine will
actually give (`hostmem::safe_pool_bytes`). Consulting only the budget is how a pool grows
past the point where the allocator can satisfy it, and then the process dies.

A cooldown between applied changes prevents oscillation.

---

# 4. Edge cases

## 4.1 Thundering herd / cache stampede — **handled, two mechanisms**

Two distinct failure modes, often conflated:

**(a) Concurrent miss on the same key.** A thousand requests arrive for a key that is not
resident. Naively, all thousand go to the origin, which falls over — and the traffic that
killed it was caused by the cache.

*Solution: single-flight rebuild leases.* `Engine::rebuild` / `Leases`. The first miss is
granted a lease and told `"rebuild": true`. Every subsequent miss inside the lease window
(`rebuild_lease_ms`, default 5000) is told `"rebuild": false` and a wait hint. **One origin
call, not a thousand.** Counted and reported as `single_flight.origin_calls_suppressed`, so
the claim is measurable rather than asserted.

**(b) Synchronised expiry.** A thousand objects written together share a TTL and all expire in
the same second, sending the entire miss stream at the origin at once.

*Solution: a soft TTL and refresh-ahead.* At `soft_ttl_fraction` of its life (default 0.80) an
object becomes *stale but servable*: the reader gets the old value immediately and a rebuild
is queued behind them. The object is replaced before it ever expires, so the expiry storm has
nothing left to storm about. `stale_serves` is counted separately, because serving stale is a
decision that should be visible.

## 4.2 Sudden spikes and flash crowds — **handled**

Traffic collapses onto one key. Three things happen together:

- The lease suppresses the duplicate origin calls (above).
- The regime detector classifies `FlashCrowd` from burstiness and entropy, and the bandit's
  posteriors shift toward the experts that suit it — `trend_aware` and `tiny_lfu` rise,
  `lfu` falls, because historical frequency is exactly the wrong signal for a key that was
  born ninety seconds ago.
- `acceleration` and `trend` are features, so the arriving hot key scores highly on its *rate
  of change* rather than needing to accumulate history first.

Injectable live from the console: `flash_crowd`, `hot_key_emergence`, `hot_key_decay`.

## 4.3 The database changed underneath us — **handled, and this is the strongest feature**

A price changes in Postgres. Everything derived from that row is now wrong.

*Solution: dependency tags (surrogate keys).* An application declares what an object was built
from:

```python
await cache.get_or_regen(
    key=f"product:{pid}",
    depends_on=[f"row:product:{pid}", "table:pricing"],
    ...)
```

On a write, the application calls `POST /v1/invalidate` with the tag. **Exactly the objects
built from that row are dropped, and nothing else.** Not a flush, not a prefix scan, not
waiting out a TTL.

Two modes, and the difference matters:

- **`hard`** — remove immediately. What a price, a balance or a permission needs. Being wrong
  is unacceptable.
- **`soft`** — mark stale; the next reader gets the old value once while a rebuild runs behind
  them. Correct for a derived rollup where a few seconds of staleness costs nothing and a
  stampede costs a lot.

The three removal reasons — evicted, invalidated, expired — are counted **separately**, never
merged into one number. A dashboard showing one figure for all three hides the only one of the
three that indicates a bug.

## 4.4 Stale data generally — **handled**

Every object has three states, not two: **fresh** → **stale but servable** (past the soft TTL)
→ **gone** (past the hard TTL). `GET` returns a `"stale": true` flag rather than burying it, so
a caller that cares about freshness can choose and one that does not can ignore it.

## 4.5 TTL — **handled**

Hard TTL from the application. Soft TTL derived as a fraction of it, per profile. A TTL of
zero means "no TTL" and the object lives until it is evicted or invalidated — legitimate for
immutable content. `ttl_remaining_frac` is both a model feature and a direct multiplier on
resident density, so an object nearing expiry is naturally cheaper to evict without needing a
special case.

## 4.6 Event-based invalidation, model redeploys — **handled**

A new recommendation model makes every cached recommendation obsolete. Flushing would empty a
large part of the cache at once and send the entire miss stream at the origin — *the cache
causing the outage it exists to prevent.*

*Solution: namespace versions.* `POST /v1/version/bump` retires a generation. New requests
carry the new version and miss cleanly; the old generation is never read again and ages out
under ordinary eviction pressure, spread over minutes instead of arriving in one spike.
**Nothing is deleted.**

## 4.7 Objects that do not exist — **handled**

A key that is absent and popular is the traffic that reaches the origin 100% of the time,
every time, forever. It is also the traffic a value-scored cache is most likely to discard,
since an absence has almost no size and almost no obvious value.

*Solution: negative caching.* An absence is stored as a marker with a short TTL (30s), counted
separately as `negative_hits`, and never handed to the caller as a value. Deliberately
**L2-only**: an absence is the cheapest thing in the cache to rebuild and the most
embarrassing to serve stale, because the object appearing is precisely the event that makes
the memory wrong.

## 4.8 Scans — **handled**

A batch job reads a million rows once. Under LRU this evicts the entire working set.

*Solution:* the `novelty_rate` signal detects that most recently-seen keys were never seen
before; the regime detector classifies `Scan`; single-touch objects with no prior admission
window entry are refused **at the door**. Refusing is strictly better than admitting and then
regretting: an object admitted and immediately evicted has already cost you the eviction of
something good.

## 4.9 One application starving another — **handled**

An analytics scan evicts every recommendation object. Every individual eviction is *correct*
on value density — a scanned row genuinely does outrank a cold recommendation — and the sum of
correct decisions is a cache that has forgotten a tenant.

*Solution: per-application floors.* `floor = capacity / resident_applications ×
app_floor_fraction` (default 0.5). At three applications each is guaranteed a sixth of the
pool and the other half is contested purely on merit. Eviction prefers unprotected candidates
and falls back to protected ones only when the whole sample is protected — **the byte budget
is an invariant, fairness is a preference.**

## 4.10 Cold start — **handled**

A cache with no history and an untrained model. The blend weights the experts at
`1 − confidence`, and confidence ramps over the first 5,000 samples, so the first minute is
governed by cheap heuristics that are never badly wrong. The capacity controller separately
refuses to shrink below 5,000 requests.

## 4.11 The cache itself failing — **handled**

The application's client carries a circuit breaker. If the engine is unreachable, the breaker
opens and calls skip straight to the origin rather than waiting on a timeout for every
request. **A cache outage must degrade to slow, never to down.**

## 4.12 Unbounded memory growth — **handled** (this was a real bug we fixed)

Per-key feature state grew without limit — the tracker for every key ever seen, forever. The
engine OOMed and restarted on Render, which is what was killing the WebSocket. Now the tracked
key count is capped proportionally to the pool (`tracked_key_ceiling`), with progressively
more aggressive age-based pruning as it approaches the ceiling.

## 4.13 What we do NOT handle — stated plainly

Being able to name these is worth more than pretending they do not exist.

| Not handled | Why, and what it would take |
|---|---|
| **Multi-node L2 coherence** | The engine is a single node. There is no cross-node invalidation protocol. `ScaleOut` is *detected and reported* — "more memory pays for itself, but this host cannot supply it" — but no second node is provisioned. Real work: consistent hashing plus a broadcast invalidation channel. |
| **L1 invalidation across processes** | We do not attempt it. That is precisely why `write_bound` objects are excluded from L1 by default: reaching every process reliably is a hard problem, and pretending otherwise is how caches serve stale prices. The 5-second L1 TTL ceiling bounds the blast radius instead. |
| **Persistence across restart** | The pool is in memory. A restart is a cold cache. Deliberate — durability is the origin's job. |
| **Compression or deduplication** | Objects are stored as given. |
| **Write-through / write-behind** | Read-through only. The application owns its writes. |
| **`bandit.kind` config knob** | Settable, and does nothing: the implementation is always Thompson sampling. It should be removed. |
| **Memory-fair headline benchmark** | The headline gives AURA 512MB against the baselines' 128MB. At equal memory the control loses on 4 of 6 scenarios — winning the two cost-skewed ones, which is what it was built for. |

---

# 5. The story to tell

Two applications, chosen because they stress the cache in *opposite* directions. That contrast
is the demo — one application would only ever prove that the cache works for that application.

## The two applications

**Recommendation** — small objects, hot keys, latency-shaped. A few kilobytes each, rebuilt in
tens of milliseconds by a scoring model. High reuse, cheap to rebuild. The failure that hurts
here is *slowness*: a recommendation strip that takes 400ms is a recommendation nobody sees.
Profile: `latency`.

**Analytics** — large objects, cold keys, expensive. Rollups over Postgres: tens of kilobytes,
400ms and up, aggregating across tables. Low reuse, brutally expensive to rebuild, and prone
to scans when someone opens a dashboard that touches every region. The failure that hurts here
is *cost*: each miss is real database load. Profile: `cost`.

## The five beats

**Beat 1 — the collision.** Both applications share one pool. Watch the Evidence tab. A
conventional cache would give these two identical treatment, because it has no way to know
they are different — same bytes, same API, same everything as far as LRU can see. Ours knows
that an analytics rollup costs two hundred times what a product lookup costs, because
analytics *told* it, with every write.

*Show:* the Decisions feed. One line per object: rebuild cost, reuse probability, value
density against the threshold, and the reason. **Every decision, with its reasoning, in
English.**

**Beat 2 — pressure, and the difference it makes.** Turn the load up (`req/s` box). The pool
fills. Now every admission is a trade. Watch the cost chart: LRU, LFU and GDS are shadow
caches running on this same traffic, charged these same prices, right now. The gap between the
lines is the money the decision made.

*Say:* "These aren't numbers from a paper. Those three policies are seeing every request this
one sees."

**Beat 3 — the world stops being polite.** Fire three disturbances, and *predict each one out
loud before you press the button*:

- **Flash crowd** — a thousand requests for one key that isn't there. "One origin call, not a
  thousand." Point at `origin_calls_suppressed`.
- **Price change** — a row changes in Postgres. "Exactly the objects built from that row are
  dropped. Not a flush." Point at `keys_invalidated` against `resident_objects`.
- **Analytics scan** — "Two things stop this. It's refused at the door as a scan signature, and
  even the parts that do get in cannot evict recommendation below its floor." Point at the
  admissions/refusals chart: **refusals spike, evictions stay flat.**

**Beat 4 — it is learning, and you can see it.** The bandit chart. Six bands, six experts,
shifting as the workload changes. "Nothing about this was configured. The reward is what
actually happened sixty seconds after each decision, not what we predicted would happen. The
cache cannot mark its own homework."

Then the closer: "LRU is one of those six bands. When LRU is right for the workload, we *are*
LRU. We don't beat it by being different — we beat it by knowing when to be it."

**Beat 5 — the honest one.** Do not let a judge find this first:

> "Our headline benchmark gives us more memory than the baselines. At equal memory our control
> loses on four of six scenarios — and wins the two cost-skewed ones, which is exactly what
> this was built for. A value-scored cache pays for itself when rebuild costs vary. When every
> object costs the same to rebuild, it cannot, and it shouldn't pretend to."

That is the strongest thing you can say in the room. Everyone else's numbers are all upside.

## The single sentence, if you only get one

> Conventional caches rank objects by *when* they arrived. This one ranks them by *what they
> would cost to rebuild*, learns which ranking is working from what actually happens, and buys
> memory only when the next gigabyte would save more than it costs.
