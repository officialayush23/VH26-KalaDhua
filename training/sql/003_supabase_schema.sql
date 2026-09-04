-- AURA — Supabase schema.
-- Idempotent. Safe to run repeatedly in the Supabase SQL editor.
--
-- Two concerns live here and are kept apart on purpose:
--   aura_*  control plane. Model registry, benchmark results, traces, events.
--   app_*   the analytics workload. Real tables the analytics service queries on a
--           cache miss, so regeneration cost is measured against a real database
--           rather than a sleep timer.
--
-- Normalisation: third normal form throughout. Every non-key column depends on the
-- whole key and nothing but the key. Repeating groups (metrics, feature lists, per-policy
-- results) are separate tables, not arrays or JSON blobs, except where the payload is
-- genuinely schemaless and never joined on.

begin;

create extension if not exists "pgcrypto";

-- ============================================================================
-- 1. Control plane: model registry
-- ============================================================================

-- 3NF note: kind and objective are small closed vocabularies, so they are lookup
-- tables rather than free text repeated on every row.
create table if not exists aura_model_kinds (
    kind_id     smallint primary key,
    code        text        not null unique,
    description text        not null
);

insert into aura_model_kinds (kind_id, code, description) values
    (1, 'lightgbm_gbdt',   'Gradient boosted trees dumped to the portable bundle format'),
    (2, 'linear_logistic', 'Logistic regression used for the cold-start and online path'),
    (3, 'onnx',            'ONNX graph loaded through the optional runtime')
on conflict (kind_id) do nothing;

create table if not exists aura_models (
    model_id      uuid        primary key default gen_random_uuid(),
    name          text        not null,
    kind_id       smallint    not null references aura_model_kinds (kind_id),
    horizon_ms    integer     not null check (horizon_ms > 0),
    version       text        not null,
    git_sha       text,
    storage_path  text        not null,
    onnx_path     text,
    is_active     boolean     not null default false,
    created_at    timestamptz not null default now(),
    constraint aura_models_name_version_key unique (name, version)
);

-- At most one active model per (name, horizon). Enforced by the database rather than
-- by whichever client happens to be promoting a model.
create unique index if not exists aura_models_one_active
    on aura_models (name, horizon_ms)
    where is_active;

create index if not exists aura_models_active_lookup
    on aura_models (name, horizon_ms, created_at desc);

-- Metrics are a repeating group over a model. In 3NF that is a child table, which also
-- means "show me AUC across every model version" is a plain query.
create table if not exists aura_model_metrics (
    model_id    uuid   not null references aura_models (model_id) on delete cascade,
    metric_name text   not null,
    metric_value double precision not null,
    primary key (model_id, metric_name)
);

-- Feature order is part of the model contract: position matters, so it is stored as an
-- ordinal, not as a set.
create table if not exists aura_model_features (
    model_id     uuid     not null references aura_models (model_id) on delete cascade,
    position     smallint not null check (position >= 0),
    feature_name text     not null,
    mean         double precision,
    scale        double precision,
    primary key (model_id, position),
    constraint aura_model_features_unique_name unique (model_id, feature_name)
);

-- ============================================================================
-- 2. Control plane: benchmarks
-- ============================================================================

create table if not exists aura_scenarios (
    scenario_id smallint primary key,
    code        text     not null unique,
    name        text     not null,
    description text     not null
);

insert into aura_scenarios (scenario_id, code, name, description) values
    (1, 'steady_zipf',         'Steady Zipf',          'Stationary skewed traffic; the control case'),
    (2, 'flash_crowd',         'Flash crowd',          'Traffic collapses onto a few keys, then releases'),
    (3, 'scan_resistance',     'Scan resistance',      'A sweep of one-touch keys tries to flush the working set'),
    (4, 'expensive_tail',      'Expensive tail',       'The rarest objects are the costliest to rebuild'),
    (5, 'shifting_popularity', 'Shifting popularity',  'The hot set is swapped out underneath the cache'),
    (6, 'mixed_production',    'Mixed production',     'Three applications plus overlapping disturbances')
on conflict (scenario_id) do nothing;

create table if not exists aura_policies (
    policy_id smallint primary key,
    code      text     not null unique,
    family    text     not null,
    description text   not null
);

insert into aura_policies (policy_id, code, family, description) values
    (1, 'lru',         'baseline', 'Least recently used'),
    (2, 'lfu',         'baseline', 'Least frequently used'),
    (3, 'gdsf',        'baseline', 'Greedy dual size frequency'),
    (4, 'tiny_lfu',    'baseline', 'Frequency sketch admission filter'),
    (5, 'cost_aware',  'baseline', 'Regeneration cost per byte'),
    (6, 'trend_aware', 'baseline', 'Short-horizon trend and acceleration'),
    (7, 'aura',        'engine',   'Learned mixture with economic admission control'),
    (8, 'belady',      'bound',    'Offline optimal; an upper bound, not a competitor')
on conflict (policy_id) do nothing;

create table if not exists aura_benchmark_runs (
    run_id          uuid        primary key default gen_random_uuid(),
    run_key         text        not null unique,
    scenario_id     smallint    not null references aura_scenarios (scenario_id),
    seed            bigint      not null,
    capacity_bytes  bigint      not null check (capacity_bytes > 0),
    requests        bigint      not null check (requests > 0),
    engine_version  text        not null,
    model_id        uuid        references aura_models (model_id),
    created_at      timestamptz not null default now()
);

create index if not exists aura_benchmark_runs_scenario
    on aura_benchmark_runs (scenario_id, created_at desc);

-- One row per (run, policy). The per-policy numbers are attributes of that pair, which
-- is exactly what makes this 3NF rather than a wide run table with lru_cost, lfu_cost…
create table if not exists aura_benchmark_results (
    run_id                   uuid     not null references aura_benchmark_runs (run_id) on delete cascade,
    policy_id                smallint not null references aura_policies (policy_id),
    object_hit_rate          double precision not null check (object_hit_rate between 0 and 1),
    byte_hit_rate            double precision not null check (byte_hit_rate between 0 and 1),
    p95_latency_ms           double precision not null check (p95_latency_ms >= 0),
    backend_requests         bigint           not null check (backend_requests >= 0),
    total_cost_usd           numeric(14, 6)   not null check (total_cost_usd >= 0),
    regen_cost_usd           numeric(14, 6)   not null check (regen_cost_usd >= 0),
    sla_penalty_usd          numeric(14, 6)   not null default 0 check (sla_penalty_usd >= 0),
    holding_cost_usd         numeric(14, 6)   not null default 0 check (holding_cost_usd >= 0),
    decision_overhead_us_p50 double precision not null default 0,
    memory_overhead_bytes    bigint           not null default 0,
    primary key (run_id, policy_id)
);

-- ============================================================================
-- 3. Control plane: traces and events
-- ============================================================================

create table if not exists aura_traces (
    trace_id     uuid        primary key default gen_random_uuid(),
    name         text        not null unique,
    scenario_id  smallint    not null references aura_scenarios (scenario_id),
    storage_path text        not null,
    row_count    bigint      not null check (row_count >= 0),
    unique_keys  bigint      not null check (unique_keys >= 0),
    size_bytes   bigint      not null check (size_bytes >= 0),
    seed         bigint      not null,
    generator_version integer not null default 1,
    created_at   timestamptz not null default now()
);

create table if not exists aura_event_kinds (
    kind_id smallint primary key,
    code    text     not null unique
);

insert into aura_event_kinds (kind_id, code) values
    (1, 'ScaleUp'), (2, 'ScaleDown'), (3, 'ScaleOut'), (4, 'PolicyShift'),
    (5, 'AttackStart'), (6, 'Eviction'), (7, 'Refresh'), (8, 'ModelReload'),
    (9, 'SimStart')
on conflict (kind_id) do nothing;

create table if not exists aura_events (
    event_id  bigserial   primary key,
    occurred_at timestamptz not null default now(),
    kind_id   smallint    not null references aura_event_kinds (kind_id),
    -- Free-form by nature: the payload differs per kind and is never joined on, so it
    -- stays as a document rather than being forced into columns.
    detail    jsonb       not null default '{}'::jsonb
);

create index if not exists aura_events_recent on aura_events (occurred_at desc);
create index if not exists aura_events_kind on aura_events (kind_id, occurred_at desc);

-- ============================================================================
-- 4. Analytics workload
--    These are the tables the analytics service actually queries on a cache miss.
--    The regeneration cost the cache learns comes from real query execution here.
-- ============================================================================

create table if not exists app_countries (
    country_id  smallint primary key,
    iso_code    char(2)  not null unique,
    name        text     not null unique
);

create table if not exists app_regions (
    region_id   serial   primary key,
    country_id  smallint not null references app_countries (country_id),
    name        text     not null,
    constraint app_regions_unique_name_per_country unique (country_id, name)
);

create table if not exists app_categories (
    category_id smallint primary key,
    name        text     not null unique
);

create table if not exists app_products (
    product_id  serial       primary key,
    category_id smallint     not null references app_categories (category_id),
    sku         text         not null unique,
    name        text         not null,
    unit_price  numeric(10, 2) not null check (unit_price >= 0)
);

create table if not exists app_customers (
    customer_id bigserial   primary key,
    region_id   integer     not null references app_regions (region_id),
    external_ref text       not null unique,
    created_at  timestamptz not null default now()
);

-- Orders carry no product or price columns. Line items do. That split is what keeps
-- this in third normal form: order-level facts depend on the order, line-level facts
-- depend on the line.
create table if not exists app_orders (
    order_id    bigserial   primary key,
    customer_id bigint      not null references app_customers (customer_id),
    region_id   integer     not null references app_regions (region_id),
    placed_at   timestamptz not null default now(),
    status      text        not null default 'complete'
                            check (status in ('pending', 'complete', 'cancelled'))
);

create table if not exists app_order_items (
    order_id    bigint       not null references app_orders (order_id) on delete cascade,
    line_no     smallint     not null check (line_no > 0),
    product_id  integer      not null references app_products (product_id),
    quantity    integer      not null check (quantity > 0),
    -- Price is duplicated from the product on purpose: it is the price *at the time of
    -- sale*. That is a different fact from the current catalogue price, so it is not a
    -- normalisation violation.
    unit_price  numeric(10, 2) not null check (unit_price >= 0),
    primary key (order_id, line_no)
);

-- The analytics queries filter by region and time and group by product and category.
-- These indexes are what make the miss path expensive-but-real rather than pathological.
create index if not exists app_orders_region_time  on app_orders (region_id, placed_at desc);
create index if not exists app_orders_customer     on app_orders (customer_id, placed_at desc);
create index if not exists app_order_items_product on app_order_items (product_id);
create index if not exists app_products_category   on app_products (category_id);
create index if not exists app_customers_region    on app_customers (region_id);

-- Derived amount, exposed as a view so no table stores a value computable from its own
-- columns. Storing line_total would be a 3NF violation.
create or replace view app_order_totals as
select
    o.order_id,
    o.region_id,
    o.customer_id,
    o.placed_at,
    sum(i.quantity * i.unit_price) as order_total,
    count(*)                       as line_count
from app_orders o
join app_order_items i on i.order_id = o.order_id
where o.status = 'complete'
group by o.order_id, o.region_id, o.customer_id, o.placed_at;

-- ============================================================================
-- 5. Row level security
--    Anon may read reference and result data. Everything is written by the service
--    role, which bypasses RLS. No policy grants anon a write.
-- ============================================================================

alter table aura_models             enable row level security;
alter table aura_model_metrics      enable row level security;
alter table aura_model_features     enable row level security;
alter table aura_benchmark_runs     enable row level security;
alter table aura_benchmark_results  enable row level security;
alter table aura_traces             enable row level security;
alter table aura_events             enable row level security;

-- The anon and authenticated roles exist on Supabase but not on a plain Postgres, so
-- the same script can be applied to a local Docker instance without editing.
do $$
declare
    t text;
    roles text;
begin
    select string_agg(quote_ident(rolname), ', ')
      into roles
      from pg_roles
     where rolname in ('anon', 'authenticated');

    if roles is null then
        raise notice 'skipping read policies: neither anon nor authenticated exists';
        return;
    end if;

    foreach t in array array[
        'aura_models', 'aura_model_metrics', 'aura_model_features',
        'aura_benchmark_runs', 'aura_benchmark_results', 'aura_traces', 'aura_events'
    ] loop
        execute format('drop policy if exists %I on %I', t || '_read', t);
        execute format(
            'create policy %I on %I for select to %s using (true)',
            t || '_read', t, roles);
    end loop;
end $$;

commit;
