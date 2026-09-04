# AURA example applications

Three services that sit in front of the AURA cache, plus the client SDK they
share. They exist to demonstrate one claim: **the cache is
application-agnostic, and it is the applications - not the cache - that know
what a miss costs.**

Every application does the same three things:

1. ask AURA for an object,
2. on a miss, rebuild it while *measuring* CPU, accelerator time, database
   service time, bytes and third-party spend,
3. hand that measurement back with the object, as an `ObjectContext`
   (contract section 1.2).

The cache never learns what a "ranking" or a "dashboard query" is. It sees a
key, a size, a TTL, an SLA class and a cost vector. That is the entire
interface, and it is why a fourth application takes about forty lines to add
(see [Plugging in a fourth application](#plugging-in-a-fourth-application)).

---

## Layout

```
apps/
├── common/
│   ├── aura_client.py    the SDK: get_or_regen, explain, circuit breaker, stats
│   ├── costing.py        CostVector, ObjectContext, CostMeter, the pricing table
│   ├── loadgen.py        zipf / scan / burst / popularity-shift generators
│   ├── service.py        shared HTTP scaffolding: routes, JSON logs, request ids
│   ├── settings.py       pydantic-settings, environment driven
│   └── telemetry.py      per-app counters behind /stats and /metrics
├── recommendation/       port 8101  - CPU and accelerator heavy, large objects
├── analytics/            port 8102  - database heavy, small objects
├── content/              port 8103  - bandwidth heavy, priced third-party objects
├── driver/run_universe.py            mixed workload with a live table
└── tests/
    ├── fake_aura.py                  a stand-in aura-server (contract section 2)
    └── test_client_against_fake.py   SDK behaviour: hit, miss, admit, reject, outage
```

---

## What each application models, and what it costs

| | recommendation (8101) | analytics (8102) | content (8103) |
|---|---|---|---|
| object | personalised ranking | dashboard aggregate | media / document blob |
| size | 0.5 - 4 MB | 0.4 - 500 KB | 100 KB - 20 MB |
| dominant cost | `cpu_ms` + `gpu_ms` | `db_ms` | `network_bytes`, sometimes `api_cost_usd` |
| measured regen (this machine) | p50 ≈ 90 ms, p95 ≈ 1.9 s | p50 ≈ 30 ms, p95 ≈ 600 ms | p50 ≈ 1 ms, priced objects 20 - 600 ms |
| TTL | 300 s | 120 s - 1800 s per query | 90 s - 3600 s per object type |
| SLA class | `high` | `critical` / `high` / `normal` | `high` / `normal` / `low` |
| traffic shape | personalised long tail, fast popularity shift | strong temporal locality, hour bursts | flash popularity, short-lived interest |
| what it punishes | size-blind admission | cost-blind eviction (tiny objects look free) | frequency-only ranking (a 20 MB object is not one unit) |

The three are deliberately in tension. An analytics result is 30 KB and costs
half a second of database time to rebuild: the cheapest thing in the cache to
keep and one of the most expensive to lose. A content video segment is 12 MB and
costs a millisecond of CPU: the most expensive thing to keep and nearly free to
lose - *unless* it is one of the priced objects, in which case losing it costs
real money. No single scalar ranks those correctly, which is the point.

### The regeneration work is real

Nothing sleeps to fake a cost.

* **recommendation** runs item-item cosine similarity over a 6 000 x 384 float32
  embedding matrix: a candidate block of 2 000 items scored against the full
  catalogue, chunked so peak memory stays bounded, then a diversity-aware
  ranking pass in Python. `cpu_ms` is a `time.process_time` delta; `gpu_ms` is
  the measured retrieval stage, the part a production system would run on an
  accelerator; `db_ms` is the feature-store lookup.
* **analytics** runs real SQL - joins across `app_orders`, `app_products` and
  `app_regions`, `GROUP BY`, and window functions (`rank`, `percent_rank`,
  `lag`, `first_value`, `ntile`, moving-average frames). `db_ms` is the measured
  service time of the statement.
* **content** generates blobs from a seeded, salted block and buys the
  `syndicated_article` object type from a simulated priced provider with
  heavy-tailed latency.

### The expensive tail

Every application exposes `POST /expensive-tail`. When it is on, a stable ~5% of
keys - chosen by a hash of the key, so membership never correlates with
frequency - costs far more to regenerate while keeping the *same size* and the
*same access rate*:

* recommendation runs 30 ensemble shards instead of one (measured ≈ 90 ms → ≈
  1.9 s, roughly 15-25x depending on the key);
* analytics switches to `cohort_retention`, which has no date predicate and
  windows over the whole order book (measured ≈ 25 ms → ≈ 530 ms, roughly 20x);
* content buys the priced document 30 times instead of once, so
  `api_cost_usd` scales with the multiplier.

This is the workload a frequency-only policy cannot see: the keys look identical
from the outside, and only the reported cost vector tells them apart.

---

## Running it

Python 3.11. From `apps/`:

```bash
pip install --break-system-packages -r requirements.txt
```

The applications talk to `aura-server` on `http://localhost:8080`. Until the
engine exists, `tests/fake_aura.py` is a working stand-in that implements
contract section 2:

```bash
python -m tests.fake_aura --port 8080          # optional stand-in cache
python -m recommendation.main                  # 8101
python -m analytics.main                       # 8102
python -m content.main                         # 8103
```

Everything is configured from the environment (prefix `AURA_APPS_`):

```bash
export AURA_APPS_AURA_BASE_URL=http://localhost:8080
export AURA_APPS_LOG_LEVEL=INFO
export SUPABASE_DIRECT_CONNECTION_URL=postgresql://...    # optional
```

**The applications never require the cache to be up.** With no server on 8080
they log the outage once, open a circuit breaker, and serve every request from
their own origin path. Killing `aura-server` mid-demo degrades the hit rate and
nothing else.

### Analytics database

If `SUPABASE_DIRECT_CONNECTION_URL` is set, the app opens an asyncpg pool with a
server-side `statement_timeout` and runs the Postgres dialect of every query. If
it is unset - or if Postgres cannot be reached - it falls back to a local SQLite
fixture at `AURA_APPS_SQLITE_PATH` (default `/tmp/aura_analytics.sqlite3`),
seeded on first boot with 12 regions, 800 products and 300 000 orders whose
region distribution is deliberately skewed. The schema matches contract section
6, and the SQLite dialect of each query keeps the same joins, grouping and
window functions. Seeding takes about 15 seconds once.

### The demo path

```bash
python -m driver.run_universe --duration 120 --rps 20 --expensive-tail --price-spike
```

It drives a mixed workload (zipf on analytics, popularity-shift on
recommendation, burst on content), prints a live table every few seconds, and -
with `--price-spike` - raises the content provider's price mid-run so the cost
column jumps without the traffic changing. `--spawn` starts the three services
itself.

```
application         reqs   hit%  regens   p50ms    p95ms   avg KB    spent $    saved $     cache
-------------------------------------------------------------------------------------------------
recommendation       161   18.6     131   944.6   1859.3   2204.0   0.029957   0.006347    closed
analytics            401   80.5      78    42.8    597.1     33.1   0.000563   0.002387    closed
content              241   66.8      80     1.8     83.8   2354.3   0.060196   0.108511    closed
-------------------------------------------------------------------------------------------------
total                                                               0.090716   0.117245
```

`spent` is what regeneration actually cost, priced with the engine's own table;
`saved` is what the hits avoided. On this run the cache is already paying for
itself, and the three applications reach that point by completely different
routes.

---

## The SDK

```python
from common.aura_client import AuraClient

async with AuraClient("http://localhost:8080", application="analytics", default_sla="high") as client:
    value = await client.get_or_regen(
        key="analytics:revenue:mumbai:30d",
        object_type="dashboard_query",
        ttl_ms=600_000,
        regen=load_revenue,          # sync or async; may take a CostMeter
    )
```

What `get_or_regen` does:

1. `GET /v1/cache/{key}?application=...`. On a hit it records the hit, credits
   the saving with the cost the key last measured, and returns.
2. On a miss it runs `regen()` inside a `CostMeter`: wall latency,
   `time.process_time` delta for `cpu_ms`, whatever the callable reports for
   `db_ms` / `gpu_ms` / API spend, and the serialised size of the value.
3. `PUT /v1/cache/{key}` with the value and the measured `ObjectContext`.
4. `admitted: false` is a normal answer, not an error - the value is returned
   either way.

`regen` may be a coroutine or a plain function, may take zero arguments or a
single `CostMeter`, and may return either the value or a `(value, CostVector)`
pair. A non-zero dimension it reports overrides what the SDK measured; anything
it leaves at zero keeps the measurement.

Other API:

| call | purpose |
|---|---|
| `get` / `put` / `delete` / `refresh` / `batch_get` | the data plane |
| `explain(key)` | `GET /v1/explain/{key}` - why this object was kept or evicted |
| `explain_recent(limit)` | the engine's latest decisions |
| `stats()` | hits, misses, admissions, rejections, breaker state, USD spent and saved |
| `SyncAuraClient` | blocking facade for scripts; runs its own loop on a thread |

Resilience: one httpx connection pool with explicit connect and read timeouts, a
circuit breaker that opens after N consecutive failures and half-opens after a
cooldown, a single log line per outage rather than one per request, and no code
path where a cache problem can fail an application request.

---

## Endpoints

Every application exposes the contract section 7 surface plus a few extras.

```bash
# health
curl -s localhost:8102/health

# the expensive endpoint, through AURA
curl -s "localhost:8101/work/1842"
curl -s "localhost:8101/work/1842?fresh=true"        # bypass the cache
curl -s "localhost:8101/work/1842?value=true"        # include the full 2 MB payload
curl -s "localhost:8102/work/17?rows=true"           # include all result rows
curl -s "localhost:8103/work/900"

# counters
curl -s localhost:8102/stats
curl -s localhost:8102/metrics                       # Prometheus text

# synthetic load: zipf | scan | burst | popularity_shift
curl -s -XPOST localhost:8102/load \
     -d '{"rps":200,"duration_s":60,"pattern":"zipf","key_space":240}'

# the pathological workload
curl -s -XPOST localhost:8101/expensive-tail \
     -d '{"enabled":true,"fraction":0.05,"multiplier":30}'
curl -s localhost:8101/expensive-tail

# why did AURA keep or evict my object?
curl -s localhost:8101/explain/1842

# what this application is, in cost terms
curl -s localhost:8103/profile

# analytics: the exact SQL a key id will run
curl -s "localhost:8102/sql/17"
curl -s "localhost:8102/sql/17?expensive=true"

# content: change the third-party price at runtime (the cost-spike lever)
curl -s localhost:8103/price
curl -s -XPOST localhost:8103/price \
     -d '{"price_usd":0.025,"median_latency_ms":120,"reason":"provider_surge"}'
```

Every response carries an `x-request-id` (echoed from the request when
supplied) and `cache-control: no-store`. Logs are one JSON object per line,
with the request id attached.

---

## Plugging in a fourth application

Nothing in `common/` knows about the three that exist. A new application needs a
key, an object type, a TTL, an SLA class, and a callable that rebuilds the
object. Create `apps/search/main.py`:

```python
from common.costing import CostMeter, CostVector
from common.service import AppService, build_app, configure_logging

class SearchService(AppService):
    def __init__(self) -> None:
        super().__init__(application="search", default_sla="normal", work=self.produce)

    def cache_key(self, key_id):
        return f"search:query:{key_id}"

    async def produce(self, key_id, fresh, options=None):
        async def regen(meter: CostMeter):
            hits, index_ms = await run_query(key_id)      # your work
            meter.add_db_ms(index_ms)
            return {"hits": hits}, CostVector(db_ms=index_ms)

        outcome = await self.client.get_or_regen_detailed(
            self.cache_key(key_id),
            object_type="search_result",
            ttl_ms=60_000,
            regen=regen,
            force_fresh=fresh,
        )
        self.account(object_type="search_result", outcome=outcome, key_id=key_id)
        return {"served_from": outcome.served_from, "hits": outcome.value["hits"]}

app = build_app(SearchService(), [])
```

That is the whole integration. `/health`, `/work/{id}`, `/stats`, `/metrics`,
`/load`, `/expensive-tail` and `/explain/{id}` come from `AppService`; cost
measurement, admission handling, the circuit breaker and the Prometheus series
come from the SDK. The cache needs no change, no new object type registered, no
new policy: it starts pricing `search` objects against `analytics` objects the
first time one is admitted.

Add the port to `docs/CONTRACTS.md` section 0 and a target to
`driver/run_universe.py` if it should join the mixed workload.

---

## Tests

```bash
pytest tests/ -q
```

`tests/fake_aura.py` implements contract section 2 and is served over real HTTP
by uvicorn, so the SDK exercises its actual transport. The suite asserts:
regeneration happens exactly once on a miss and the PUT carries the *measured*
cost; a hit skips regeneration and is credited with the saving; `admitted:
false` is not an error; an unreachable cache still serves every request, opens
the breaker and stops calling out; the breaker recovers; binary objects round
trip byte for byte with `size_bytes` reporting the object rather than its base64
inflation; and `explain`, `batch_get`, `delete`, `refresh` and the synchronous
facade behave.

```bash
ruff check .
```

---

## Measurement caveats

Worth knowing when reading the numbers:

* `cpu_ms` comes from `time.process_time`, which counts the whole process. A
  regeneration that awaits while other requests compute will absorb some of
  their CPU; the applications therefore measure CPU-bound sections directly
  (`CostMeter.section()`) where it matters. Under saturation, treat `cpu_ms` as
  an upper bound.
* On the SQLite fallback the database runs *inside* the process, so its CPU
  appears in both `cpu_ms` and `db_ms`. Against Postgres the two are disjoint.
* The recommendation model uses multi-threaded BLAS, so `cpu_ms` can legitimately
  exceed `latency_ms`.
