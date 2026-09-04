# The reuse model — what it predicts, what it is fed, and what that bought

## What it predicts

Three binary classifiers, one per horizon:

> Given what I know about this object right now, will it be requested again within
> **10 / 60 / 600 seconds**?

Nothing about value, cost or eviction. Those are computed deterministically downstream, in
the engine. That separation is what lets a price change take effect instantly with no
retrain, and what lets the model work on an application it has never seen.

## What it is fed — 20 features, and why each survives

### Base (12) — behavioural, taken from the engine's own feature vector

| Feature | What it says |
|---|---|
| `log_age_ms`, `log_inter_arrival_ms` | recency, and this key's typical gap |
| `freq_1m`, `freq_5m`, `freq_1h` | decayed frequency at three timescales |
| `ewma_fast`, `ewma_slow`, `trend`, `acceleration` | whether interest is building or dying |
| `log_size_bytes` | absolute footprint |
| `ttl_remaining_frac` | freshness of the resident copy |
| `cache_pressure` | how full the cache was at that moment |

### Extra (8) — signals a frequency counter cannot express

| Feature | Why a frequency counter cannot see it |
|---|---|
| `size_percentile` | relative footprint *within its own application* |
| `cost_percentile` | **how cost re-enters the model without breaking portability** |
| `cost_variance_ratio` | `p95/p50` — a miss you cannot price is worse than one you can |
| `log_reuse_distance` | distinct keys since last access. What LRU approximates by construction. Two keys with identical frequency can have completely different reuse distance |
| `burstiness` | CV of this key's gaps: steady-every-2s vs twenty-then-nothing |
| `novelty_rate` | fraction of recent requests that were first-time keys — near 1.0 during a scan |
| `hour_sin`, `hour_cos` | time of day as a phase, so 23:59 and 00:01 are adjacent |

### Removed as platform-bound

`regen_cost_usd`, `log_regen_p50_ms`, `app_id`.

Absolute dollars and milliseconds shift by orders of magnitude between deployments. Every
tree split learned on them becomes meaningless when they do — silently, with no error.
`app_id` is worse: a categorical that cannot generalise to an application it never saw.

**Cost is not gone, it is re-expressed.** `cost_percentile` says "90th-percentile expensive
for this application", which means the same thing whether the currency is GPU-seconds or
database time.

## What the extra features actually bought

Measured, not assumed. Same data, same split, same seed — the only difference is the feature
set. Synthetic bootstrap, 138,726 rows, three applications.

### In-distribution (temporal split) — they buy nothing

| Horizon | 12 features | 20 features |
|---|---|---|
| h10s | 0.9306 | 0.9299 |
| h60s | 0.9265 | 0.9266 |
| h600s | 0.8873 | 0.8850 |

### Transfer (train without `recommendation`, test only on it) — they buy real gains

| Horizon | 12 features | 20 features | Δ |
|---|---|---|---|
| h10s | 0.9524 | **0.9551** | +0.003 |
| h60s | 0.9324 | **0.9400** | +0.008 |
| h600s | 0.8797 | **0.9069** | **+0.027** |

That is the result the design predicted and the one worth reporting: the extra signals do
not help when the model has already seen the workload, and help substantially when it has
not — most at the long horizon, where frequency counters are weakest.

## Two corrections found while measuring

**The recency baseline was broken, and it was flattering us.** A key's *first* access has
`log_age_ms == 0`, which naively ranks as "just used" when it means "never seen before".
Ranking those first made the LRU baseline look terrible and produced a reported lift of
**9.5×**. With the correction, the honest lift over the best baseline is around **0.4%** on
in-distribution data. The first number was fiction and would not have survived a question.

**Frequency is very hard to beat at predicting reuse.** On a stationary Zipf workload,
frequency is nearly all the signal there is, and a boosted tree fed frequency counters will
rediscover frequency. The model's value is not that it predicts reuse better than LFU — it
is that its output feeds an economic layer that knows what a miss *costs*, which no
frequency counter has any way to express.

## Where the data comes from

Tried in order, falling through automatically:

1. **The running engine** — `GET /v1/training/rows` drains the decision journal, which holds
   the exact feature vector each decision was made from plus the labels that arrived later.
   Best source by a wide margin: no possibility of drift between training and serving, and
   it is what makes retraining continuous.
2. **The simulator's request log** — `apps/runs/requests.jsonl`, replayed through the same
   feature builder.
3. **Synthetic traces** — seven regimes, for bootstrapping from nothing.

## How labels are made

One backward pass. For the access to key `k` at time `t`, the label for horizon `h` is 1 if
`k` appears again in `(t, t + h]`.

Rows whose horizon runs past the end of the trace are **censored and dropped**. We cannot
know whether reuse happened, and labelling them 0 would teach the model that the end of
every trace is a cold region.

## How it is split

Never randomly. A random split puts an access and its own future in both sets and inflates
every metric.

- **Temporal**: 70 / 15 / 15 by time.
- **Held-out application**: train without one application, test only on it. This is the
  stronger claim and the one the transfer table above uses.

## Commands

```bash
cd training

python -m portable features                    # show the feature set and where each comes from
python -m portable bootstrap                   # synthetic -> first bundles, from nothing
python -m portable train                       # engine journal -> request log -> synthetic
python -m portable train --hold-out-application recommendation
python -m portable train --no-extras           # measure what the extra signals buy
python -m portable watch --every-s 600         # retrain forever, publishing each round
```

`watch` is the continuous loop: collect, build, train, evaluate, export, reload the running
engine. Ten minutes is the floor that makes sense — the journal settles decisions at 60 s and
retires them at 600 s, so retraining faster just refits the same rows.

Bundles land in `engine/models/` and are picked up by
`POST /v1/model/reload {"source":"file"}` without a restart.
