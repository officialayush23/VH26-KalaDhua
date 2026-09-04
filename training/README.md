# AURA training

Cache traces in, model bundles out.

This directory owns everything between a trace file and the `model_bundle.json`
that `aura-core` loads: feature construction, labelling, the regime-stratified
split, the gradient-boosted reuse model, the online logistic model used during
cold start, evaluation, and publication to Supabase.

Two things here are interface, not implementation, and cannot be changed on one
side only:

* **The feature vector.** Sixteen features, fixed order, defined in
  [`aura_train/features.py`](aura_train/features.py) and pinned by
  [`tests/golden/feature_vectors.json`](tests/golden/feature_vectors.json). The
  Rust engine computes the same sixteen numbers on the hot path. If the two
  drift apart, the model is being fed a different workload than it was trained
  on, and nothing will look obviously broken -- the probabilities will just be
  wrong.
* **The bundle format.** Contract section 5, produced by
  [`aura_train/export.py`](aura_train/export.py), which also contains the
  reference tree walker the Rust one must agree with.

---

## Quick start

```bash
cd training
pip install -r requirements.txt

# No Rust toolchain and no traces? Generate a synthetic corpus first.
python -m aura_train.cli synth

# Everything: dataset -> train -> evaluate -> export
python -m aura_train.cli all --ablations

# Check what came out, then publish it
python scripts/verify_bundle.py models/*.json
python scripts/push_model.py models/reuse_gbdt_h60s.json --activate
```

The pipeline runs with nothing but numpy, pandas, scikit-learn and matplotlib.
LightGBM, pyarrow, ONNX and the Supabase SDK are all optional, and each one has
a documented fallback rather than a hard failure -- see
[Optional dependencies](#optional-dependencies).

---

## Commands

There is no Makefile (this directory is driven from the repository root), but
these are the commands you actually type:

| Command | What it does |
|---|---|
| `python -m aura_train.cli synth` | Write one synthetic trace per regime into `data/traces` |
| `python -m aura_train.cli inspect` | List discovered traces and their detected format |
| `python -m aura_train.cli build-dataset` | Traces → feature/label shards in `data/dataset`, prints the class-balance table |
| `python -m aura_train.cli train` | Train all three horizons plus the linear model into `models/` |
| `python -m aura_train.cli train --ablations` | …and run the feature-group ablation harness |
| `python -m aura_train.cli evaluate` | Per-regime metrics, calibration, importance, counterfactual replay → `reports/` |
| `python -m aura_train.cli export` | `models/*.json` (+ ONNX), with the parity assertion |
| `python -m aura_train.cli push --activate` | Upload to Supabase Storage, insert `aura_models`, flip `is_active` |
| `python -m aura_train.cli pull --name reuse_gbdt_h60s` | Download the active bundle |
| `python -m aura_train.cli all` | build-dataset → train → evaluate → export |
| `python scripts/verify_bundle.py models/*.json` | Schema + numeric + parity validation |
| `python scripts/push_model.py models/*.json --activate` | Verify, publish, optionally reload the engine |
| `python tests/test_features_parity.py` | Run the golden-vector tests without pytest |
| `pytest tests` | Same, with pytest |
| `ruff check . && ruff format --check .` | Lint |
| `psql "$SUPABASE_DIRECT_CONNECTION_URL" -f sql/001_schema.sql` | Apply the schema |
| `psql "$SUPABASE_DIRECT_CONNECTION_URL" -f sql/002_seed_analytics.sql` | Seed ~200k analytics orders |

Paths are overridable per run (`--trace-dir`, `--dataset-dir`, `--model-dir`,
`--report-dir`) or from the environment (`AURA_TRAIN_TRACE_DIR`,
`AURA_TRAIN_DATASET_DIR`, `AURA_TRAIN_MODEL_DIR`, `AURA_TRAIN_REPORT_DIR`,
`AURA_TRAIN_SHARD_ROWS`, `AURA_TRAIN_MAX_ROWS_PER_TRACE`, `AURA_TRAIN_SEED`).

---

## The feature vector

Sixteen features, in this order. This table is the contract with the Rust
implementation; keep it in sync with
[`aura_train/features.py`](aura_train/features.py) and with
`engine/aura-core`'s feature builder.

Notation: `t` is the timestamp of the current access, `t_prev` the previous
access to the *same key*, `dt = t - t_prev` in milliseconds, `dt_s = dt / 1000`.

| # | Name | Definition | Notes |
|---|---|---|---|
| 0 | `log_age_ms` | `ln(1 + dt)` | `0` on a key's first access — there is no measurable age yet, and fabricating one biases every cold key |
| 1 | `log_inter_arrival_ms` | `ln(1 + g)` where `g ← 0.3·dt + 0.7·g`, seeded with the first observed gap | EWMA of inter-arrival gaps, `alpha = 0.3` |
| 2 | `freq_1m` | decayed counter, `c ← c·exp(−dt/60000)`, emitted *before* `c += 1` | counts strictly prior accesses |
| 3 | `freq_5m` | same with `exp(−dt/300000)` | |
| 4 | `freq_1h` | same with `exp(−dt/3600000)` | |
| 5 | `ewma_fast` | `f ← f·2^(−dt_s/5)`, emitted before `f += 1` | half-life 5 s |
| 6 | `ewma_slow` | `s ← s·2^(−dt_s/60)`, emitted before `s += 1` | half-life 60 s |
| 7 | `trend` | `ln((f + ε)/(s + ε))`, `ε = 1e-6` | positive = accelerating, negative = cooling |
| 8 | `acceleration` | `trend − trend_prev` for this key | `0` on the first access |
| 9 | `log_size_bytes` | `ln(1 + size_bytes)` | |
| 10 | `log_regen_p50_ms` | `ln(1 + p50)` where `p50` is a running median of regeneration latency for `(application, object_type)` | read before the current observation is folded in |
| 11 | `cost_variance_ratio` | `p95 / max(p50, 1)` | how heavy the regeneration-cost tail is |
| 12 | `regen_cost_usd` | `cpu_ms·1.16e-8 + gpu_ms·2.55e-7 + db_ms·3.20e-8 + (network_bytes/1e9)·0.09 + api_cost_usd` | prices from contract section 8 `[pricing]` |
| 13 | `ttl_remaining_frac` | `clamp(1 − (t − fill_ts)/ttl_ms, 0, 1)` if a copy is resident and unexpired, else `0` | |
| 14 | `cache_pressure` | `used_bytes / capacity_bytes` immediately *before* this access | |
| 15 | `app_id` | `0` recommendation, `1` analytics, `2` content; anything else `3 + FNV-1a-64(name) mod 1021` | never Python's `hash()`, which is salted per process |

Three details that are easy to get wrong and expensive to get wrong:

**Decay-read-update ordering.** Every per-key counter is decayed to `t`, *read*,
and only then incremented for the current access. So `freq_5m` at an access is
the decayed count of *previous* accesses, not including this one. Both
implementations must use this order or the two will differ by exactly one unit
on every row.

**Quantile estimation.** `p50` and `p95` are streaming SGD estimators, not exact
quantiles, because the Rust side cannot keep a histogram per object type on the
hot path:

```
lr    = 0.02 · max(estimate, 1.0)
est  += lr · q          if observation >  est
est  -= lr · (1 − q)    if observation <= est
```

with `q = 0.5` and `q = 0.95`, both seeded with the first observation, and `p95`
clamped to be at least `p50`.

**Frequency sketches.** Python uses exact per-key decayed counters. The engine
uses a decaying count-min sketch, which is the same arithmetic over a shared
array of counters, so its `freq_*` values are *upper bounds* — hash collisions
can only inflate a count. In practice the overestimate is small (a few percent
at the sketch widths the engine uses) and it is systematic rather than random,
so the model tolerates it; but it is a real difference between training and
serving, and it is why the golden fixture uses exact counters and the Rust test
asserts against them with the sketch disabled.

**Feature groups** used by the ablation harness:

| Group | Features |
|---|---|
| `recency` | `log_age_ms`, `log_inter_arrival_ms` |
| `frequency` | `freq_1m`, `freq_5m`, `freq_1h` |
| `trend` | `ewma_fast`, `ewma_slow`, `trend`, `acceleration` |
| `cost` | `log_regen_p50_ms`, `cost_variance_ratio`, `regen_cost_usd` |
| `context` | `log_size_bytes`, `ttl_remaining_frac`, `cache_pressure`, `app_id` |

---

## Labels

For an access to key `k` at time `t` and a horizon `h`:

```
label_h = 1  if k is accessed again at some t' with 0 < t' − t <= h
```

Computed by one reverse pass over the trace ([`labels.py`](aura_train/labels.py)).
Three horizons are trained: **10 s**, **60 s** (primary) and **600 s**.

**Leakage.** Features are built in a forward pass that never looks beyond `t`;
labels are built in a separate reverse pass. The two never share state, and
`test_features_do_not_look_ahead` asserts that appending future events does not
change any earlier feature vector.

**Right censoring.** If `t + h` is past the end of the trace, a zero label means
"we could not observe a reuse", not "there was none". Those rows are flagged
`censored_h*` and dropped. Keeping them would make the tail of every trace look
like a wall of negatives and teach the model that late traffic is worthless — a
mistake that is invisible in aggregate metrics and very visible in a live cache.

---

## Splitting: by regime, never at random

A random split leaks. Two accesses to the same key seconds apart end up on
opposite sides of the split, the model memorises the key, and the AUC looks
excellent right up until production. So:

* **Train**: `steady`, `zipf_shift_moderate`, `analytics_stable` — the first 80%
  of each regime, by time.
* **Validate**: the trailing 20% of those same regimes, by time. Early stopping
  and the linear model's calibration use this and nothing else.
* **Test**: `flash_crowd`, `scan`, `expensive_tail`, `cost_spike` — regimes the
  model has never seen in any form.

`evaluate.py` reports the test metrics **per regime**, as
`reports/per_regime_h60s.csv` and `reports/per_regime_h60s.png`. That table is
the project's actual claim: not "our model gets 0.93 AUC" but "our model holds
up on workload shapes it was never trained on". A pooled number would hide the
one thing worth knowing.

---

## The counterfactual replay

AUC does not pay the bill. `evaluate.py` also replays the held-out trace through
a cache simulator once per policy — LRU, LFU, GDSF and AURA — and charges each
policy the `regen_cost_usd` of every miss it allows. All four use the same
sampled-eviction machinery (worst of 32 random residents, matching the engine's
`candidate_sample = 32`), so the comparison isolates the value function rather
than the eviction mechanics. AURA's value function is

```
value = P(reuse within h) · regen_cost_usd / size_bytes
```

which is the only place the model's output enters. The result lands in
`reports/replay_h60s.csv` and `reports/replay_h60s.png` as cost in USD per
regime, plus `saving_vs_lru`.

This is where a model that ranks reuse well but ignores cost gets caught: it can
beat GDSF on hit rate and still lose on money, because it filled the cache with
cheap objects.

---

## Cold start

A freshly deployed AURA has no trained model and no telemetry. Three predictors
cover the gap, and the engine blends them by confidence
([`train_linear.py`](aura_train/train_linear.py) implements all three so the
blend can be simulated and tested offline):

1. **Heuristic.** `0.6·ln(1 + freq_5m) − 0.25·log_age_ms + 1.2`, through a
   sigmoid. No parameters, no data, always available. This is what the cache
   runs on in its first milliseconds.
2. **Linear.** `reuse_linear_h60s`, a 16-weight logistic model over standardised
   features. It ships as a hand-specified prior (`cold_start_model()`) that
   encodes what every cache paper already knows — recent and frequent keys come
   back, big keys are worth less per byte, expensive keys are worth more to
   keep — and it improves online: `partial_fit` applies one SGD step per batch
   of observed reuse outcomes, so the same artifact becomes a fitted model
   within a few thousand requests. Fitting it offline warm-starts from that same
   prior, so an under-trained model degrades towards the heuristic rather than
   towards noise.
3. **GBDT.** `reuse_gbdt_h{10,60,600}s`, the accurate model, available once a
   bundle has been trained and loaded.

The blend:

```
p = heuristic
if linear_confidence >= floor:  p = (1 − w_lin) · p + w_lin · linear
if gbdt_confidence   >= floor:  p = (1 − w_gbdt) · p + w_gbdt · gbdt
```

with `floor = [engine] ml_confidence_floor = 0.20`. Confidence comes from sample
count and observed calibration error, and a predictor below the floor
contributes exactly nothing. That floor is what makes a model rollout safe: a
newly loaded bundle cannot take over the cache before it has demonstrated it
predicts the traffic it is actually seeing.

The linear bundle's `normalization.mean` / `normalization.scale` are real
statistics (the GBDT bundle's are identity, since trees are scale-invariant), so
the engine applies `(x − mean) / scale` uniformly before either predictor.

---

## Where the data comes from

Three sources, and they are not interchangeable. Public traces have request
streams but no cost metadata; ours have both. So:

| Source | Trains the reuse head | Trains the economic head | Why |
|---|---|---|---|
| Our simulator traces | yes | **yes** | the only source with a cost vector per object |
| Public research traces | yes | no | timestamps, keys and sizes only; `cpu_ms`, `gpu_ms`, `db_ms`, `api_cost_usd` and the regeneration-latency quantiles are all zero, so `regen_cost_usd` degenerates to the network-egress term derived from object size and `cost_variance_ratio` is identically zero |
| Live server telemetry | yes | yes | measured cost from real regenerations, but only for keys the cache actually missed |

### Source 1 — our own simulator traces (primary)

Written by the Rust engine, in the contract section 4 CSV format. This is the
only source where `regen_cost_usd`, `cost_variance_ratio` and
`log_regen_p50_ms` carry real signal, so every model that the engine uses for
economic decisions is trained on these.

```bash
cd engine
cargo build --release

# One scenario at a time
./target/release/aura-bench --scenario steady               --requests 5000000 \
    --emit-trace ../training/data/traces/steady.csv.gz
./target/release/aura-bench --scenario zipf_shift_moderate  --requests 5000000 \
    --emit-trace ../training/data/traces/zipf_shift_moderate.csv.gz
./target/release/aura-bench --scenario analytics_stable     --requests 5000000 \
    --emit-trace ../training/data/traces/analytics_stable.csv.gz
./target/release/aura-bench --scenario flash_crowd          --requests 2000000 \
    --emit-trace ../training/data/traces/flash_crowd.csv.gz
./target/release/aura-bench --scenario scan                 --requests 2000000 \
    --emit-trace ../training/data/traces/scan.csv.gz
./target/release/aura-bench --scenario expensive_tail       --requests 2000000 \
    --emit-trace ../training/data/traces/expensive_tail.csv.gz
./target/release/aura-bench --scenario cost_spike           --requests 2000000 \
    --emit-trace ../training/data/traces/cost_spike.csv.gz

# Or from a running server, capturing whatever the demo is doing
./target/release/aura sim --scenario mixed_production --seed 42 \
    --emit-trace ../training/data/traces/mixed_production.csv.gz
```

Each trace writes a companion `*.meta.json`. About 24 M rows total at those
settings, roughly 1.2 GB gzipped, and the dataset build turns that into ~2.5 GB
of Parquet shards. If you have less disk, `--max-rows-per-trace 2000000` is
enough to train a usable model.

If you have no Rust toolchain, `python -m aura_train.cli synth` writes the same
seven regimes from a pure-Python generator. It is a weaker workload than the
Rust simulator — the point of it is that the pipeline, the notebook and the
tests never dead-end.

### Source 2 — open cache traces

`scripts/fetch_public_traces.sh` documents each of these and downloads the ones
that can be fetched non-interactively. **None of them carry cost metadata**, so
they are used to train the reuse head only. Concretely, for a public trace
`cpu_ms`, `gpu_ms`, `db_ms`, `api_cost_usd` and `regen_latency_ms` are all zero,
which makes `log_regen_p50_ms` and `cost_variance_ratio` identically zero and
reduces `regen_cost_usd` to the egress term implied by object size. The right
way to use them is as a pre-training corpus for the reuse head, or as an
out-of-distribution check on a model trained on our traces — not as a source of
economic signal.

| Trace | Format | Size | Notes |
|---|---|---|---|
| **Twitter cache traces** (`cluster*.sort.zst`, Yang et al., OSDI '20) | CSV: `timestamp,key,key_size,value_size,client_id,operation,ttl` | ~2.8 TB for all 54 clusters; a single cluster is 100 MB–30 GB | The best public analogue to an application cache. `cluster17`, `cluster44` are convenient sizes. Read with `read_public_csv`. |
| **libCacheSim / cache-dataset `oracleGeneral`** | binary, 24-byte records: `u32 time, u64 obj_id, u32 size, i64 next_access_vtime` | tens of MB to hundreds of GB | The most convenient way to get MSR, CDN, Wikipedia, Meta and Twitter traces in one format. Our reader **ignores** `next_access_vtime` — it is Belady's oracle and using it as a feature would be leakage. It is legitimate for the engine's Belady upper bound. Read with `read_oracle_general`. |
| **Wikimedia CDN request samples** | gzipped TSV: relative time, hashed URL, response size | a few hundred MB per day | Sampled 1:100 or 1:1000. Reuse *structure* survives sampling; absolute rates do not, so `freq_*` and `ewma_*` are learned on a compressed timescale. Say so if you publish numbers from it. |
| **Tencent block storage** (SNIA IOTTA) | CSV: time, volume, offset, length, r/w | ~10 TB full set; single volumes a few hundred MB | Block level. Fix a block size and use `(volume, offset/block)` as the key. Registration and citation required. |
| **Alibaba block traces** | same shape | ~440 GB compressed, 1000 volumes, 30 days | Apache-2.0. Together with Tencent, the best available stress test for the `scan` regime. |
| **SEC EDGAR access logs** | daily zipped CSV, includes a `size` column | 200–400 MB/day | Genuinely bursty around filing deadlines: a real flash crowd rather than a simulated one, and it has object sizes. |
| **IBM Cloud Object Storage** (SNIA IOTTA) | object-level GET/PUT with sizes | 98 buckets, 7 days | Closest public workload to what AURA targets. |
| **MSR Cambridge block traces** | CSV | small | Old and small, but every cache simulator in the literature has published hit rates for it — the standard way to check that our reader and replay are not lying. |

All of these land in `data/traces/public/` and are picked up automatically;
`aura_train.traces.detect_format` distinguishes them from our own CSVs by
header.

### Source 3 — replayed live telemetry (the online path)

The server records the features it computed and, once the horizon has elapsed,
the outcome it observed. That stream is the only data that reflects *this*
deployment's traffic, and it is what the online logistic model consumes through
`partial_fit`. Two ways to use it:

```bash
# Pull the traces the running server has archived to Supabase Storage
python -c "
from pathlib import Path
from aura_train.supabase_io import Session
s = Session()
for row in s.select('aura_traces', order='created_at', limit=10):
    print(row['name'], row['rows'], row['storage_path'])
"

# Then retrain on them exactly like any other trace
python -m aura_train.cli all --trace-dir data/traces/live
```

The caveat that matters: telemetry is **censored by the cache's own decisions**.
An object the cache evicted and never saw again might have been reused; you only
observe reuse for objects that were still around to be hit. This is standard
selection bias in cache learning, and the honest mitigations are (a) the engine
samples a small fraction of admissions as a control group that bypasses the
policy, and (b) the offline traces from source 1 are unbiased because the
simulator sees every request regardless of what the cache did. Do not train
exclusively on live telemetry.

---

## Model artifacts

Four bundles per run, all contract section 5:

```
models/
  reuse_gbdt_h10s.json     reuse_gbdt_h10s.onnx
  reuse_gbdt_h60s.json     reuse_gbdt_h60s.onnx     <- [predictor] bundle_path default
  reuse_gbdt_h600s.json    reuse_gbdt_h600s.onnx
  reuse_linear_h60s.json                            <- cold start / online path
```

The JSON bundle is authoritative. ONNX is a convenience for serving the same
model from another runtime; the engine's default path is the pure-Rust tree
walker, which has no ML dependency at all.

`export.py` asserts parity before writing anything: the bundle is scored through
the reference walker and compared against the trainer's own prediction over 1000
held-out rows, and a disagreement above `1e-6` fails the export. In practice the
observed disagreement is ~1e-16, i.e. floating-point noise. `scripts/verify_bundle.py`
re-runs that check plus schema and numeric validation, and `push_model.py`
refuses to publish a bundle that fails it.

### What the Rust side must match

1. **Feature order and definitions** — the table above, verified against
   `tests/golden/feature_vectors.json`.
2. **`app_id`** — the reserved ids `0/1/2`, and FNV-1a-64 `mod 1021` plus 3 for
   everything else.
3. **The pricing constants** — contract section 8 `[pricing]`, used for
   `regen_cost_usd`.
4. **The tree walker** — `feature <= threshold` goes left; `left[i]`/`right[i]`
   negative means leaf `-(v) - 1`; missing/NaN goes left; a single-node tree has
   `decision_type[0] == 0` and returns `leaf_value[0]`. `export.predict_bundle`
   is the executable specification.
5. **Normalisation** — apply `(x − mean) / scale` before both the tree walker
   and the linear model. It is identity for GBDT bundles, real for linear ones.
6. **Sigmoid** — sum the leaf values, then `1 / (1 + exp(−raw))` when
   `sigmoid_output` is true.

---

## Supabase

```bash
export SUPABASE_URL=https://<project>.supabase.co
export SUPABASE_SERVICE_ROLE_SECRET_KEY=<service role key>
export SUPABASE_DIRECT_CONNECTION_URL=postgresql://...

psql "$SUPABASE_DIRECT_CONNECTION_URL" -f sql/001_schema.sql
psql "$SUPABASE_DIRECT_CONNECTION_URL" -f sql/002_seed_analytics.sql

python scripts/push_model.py models/reuse_gbdt_h60s.json --activate
curl -X POST http://localhost:8080/v1/model/reload \
     -H "Content-Type: application/json" -d '{"source":"supabase"}'
```

Nothing in this directory reads a credential from anywhere but the environment.
`supabase_io.py` creates the `aura-models` bucket if it does not exist, and the
schema enforces at most one active version per model name with a partial unique
index — the invariant the engine depends on when it boots.

---

## Optional dependencies

Every one of these degrades to something that still runs, because a training
pipeline that only works on one machine is not a training pipeline.

| Missing | What happens |
|---|---|
| `lightgbm` | Trains with scikit-learn's `HistGradientBoostingClassifier` instead. Same algorithm family, and `export.py` converts both to the identical bundle encoding — the Rust walker cannot tell them apart. The numbers differ slightly; the log line says which backend ran, and so does `metrics.backend` in the bundle. |
| `pyarrow` | Dataset shards are written as gzipped CSV instead of Parquet. Slower to read, functionally identical. |
| `onnxmltools` / `skl2onnx` | ONNX export is skipped with a warning. The JSON bundle is unaffected and is what the engine loads. |
| `supabase` | `supabase_io` falls back to REST over `requests`, speaking the same PostgREST and Storage APIs. |
| `pytest` | `python tests/test_features_parity.py` runs the same tests with a built-in runner. |

Force a backend explicitly with `--backend lightgbm` or `--backend sklearn_hist`.

---

## Layout

```
training/
  aura_train/
    config.py        run configuration, pricing constants, env overrides
    features.py      the 16-feature builder  <- keep in sync with Rust
    labels.py        future-reuse labels, censoring
    traces.py        readers: our CSV.gz, oracleGeneral, public CSV variants
    synthetic.py     pure-Python trace generator (7 regimes)
    dataset.py       streaming shard build, regime-stratified split
    train_gbdt.py    LightGBM / HistGradientBoosting + ablation harness
    train_linear.py  online logistic model, cold-start prior, confidence blend
    export.py        model_bundle.json + ONNX + the reference tree walker
    evaluate.py      per-regime metrics, calibration, importance, cost replay
    supabase_io.py   Storage + aura_models / aura_traces / aura_benchmark_*
    cli.py           the entry point
  sql/               schema and analytics seed data
  scripts/           trace fetching, bundle verification, publication
  notebooks/         the Colab notebook
  tests/             golden-vector parity tests and the shared fixture
```
