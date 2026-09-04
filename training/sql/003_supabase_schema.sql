-- AURA — Supabase schema. Run this FIRST, before 004_supabase_seed.sql.
-- Idempotent: safe to paste into the Supabase SQL editor more than once.
--
-- Two independent concerns live here:
--
--   aura_*   control plane. Model registry, benchmark results, traces, events.
--            Written by training/aura_train/supabase_io.py.
--   app_*    the analytics workload. Real tables the analytics service queries on a
--            cache miss, so the regeneration cost the cache learns comes from a real
--            query rather than a sleep timer.
--
-- Normalisation. Third normal form, using natural text keys where a writer already
-- speaks in text: aura_models.kind, aura_benchmark_runs.scenario and
-- aura_benchmark_results.policy are foreign keys onto lookup tables, so a typo is
-- rejected by the database instead of becoming a new category.
--
-- Two columns stay jsonb because their writers send documents and nothing joins on
-- them: aura_models.metrics and aura_models.feature_names. They are also decomposed
-- into aura_model_metrics and aura_model_features by trigger, so the relational form
-- exists for querying without the writer having to change.

begin;

create extension if not exists "pgcrypto";

-- ============================================================================
-- 1. Lookup tables
--    Closed vocabularies. Foreign keys point at the text code, which is what the
--    Python and Rust sides already exchange.
-- ============================================================================

create table if not exists aura_model_kinds (
    code        text primary key,
    description text not null
);

insert into aura_model_kinds (code, description) values
    ('lightgbm_gbdt',   'Gradient boosted trees dumped to the portable bundle format'),
    ('linear_logistic', 'Logistic regression for the cold-start and online path'),
    ('onnx',            'ONNX graph loaded through the optional runtime')
on conflict (code) do update set description = excluded.description;

create table if not exists aura_scenarios (
    code        text primary key,
    name        text not null,
    description text not null
);

insert into aura_scenarios (code, name, description) values
    ('steady_zipf',         'Steady Zipf',         'Stationary skewed traffic; the control case'),
    ('flash_crowd',         'Flash crowd',         'Traffic collapses onto a few keys, then releases'),
    ('scan_resistance',     'Scan resistance',     'A sweep of one-touch keys tries to flush the working set'),
    ('expensive_tail',      'Expensive tail',      'The rarest objects are the costliest to rebuild'),
    ('shifting_popularity', 'Shifting popularity', 'The hot set is swapped out underneath the cache'),
    ('mixed_production',    'Mixed production',    'Three applications plus overlapping disturbances'),
    ('unknown',             'Unknown',             'Placeholder for a run whose scenario was not recorded')
on conflict (code) do update set name = excluded.name, description = excluded.description;

create table if not exists aura_policies (
    code        text primary key,
    family      text not null,
    description text not null
);

insert into aura_policies (code, family, description) values
    ('lru',         'baseline', 'Least recently used'),
    ('lfu',         'baseline', 'Least frequently used'),
    ('gdsf',        'baseline', 'Greedy dual size frequency'),
    ('tiny_lfu',    'baseline', 'Frequency sketch admission filter'),
    ('cost_aware',  'baseline', 'Regeneration cost per byte'),
    ('trend_aware', 'baseline', 'Short-horizon trend and acceleration'),
    ('aura',        'engine',   'Learned mixture with economic admission control'),
    ('belady',      'bound',    'Offline optimal; an upper bound, not a competitor')
on conflict (code) do update set family = excluded.family, description = excluded.description;

create table if not exists aura_event_kinds (
    code text primary key
);

insert into aura_event_kinds (code) values
    ('ScaleUp'), ('ScaleDown'), ('ScaleOut'), ('PolicyShift'), ('AttackStart'),
    ('Eviction'), ('Refresh'), ('ModelReload'), ('SimStart'),
    ('TrainingRun'), ('DatasetBuilt'), ('ModelPublished')
on conflict (code) do nothing;

-- ============================================================================
-- 2. Model registry
--    Column names and types match register_model() in supabase_io.py exactly.
-- ============================================================================

create table if not exists aura_models (
    id            uuid        primary key default gen_random_uuid(),
    name          text        not null,
    kind          text        not null references aura_model_kinds (code),
    horizon_ms    integer     not null check (horizon_ms > 0),
    version       text        not null,
    storage_path  text        not null,
    onnx_path     text,
    metrics       jsonb       not null default '{}'::jsonb,
    feature_names jsonb       not null default '[]'::jsonb,
    is_active     boolean     not null default false,
    created_at    timestamptz not null default now(),
    constraint aura_models_name_version_key unique (name, version)
);

-- At most one active version per model name. set_active() deactivates then activates,
-- and this index is what guarantees the intermediate state can never persist.
create unique index if not exists aura_models_one_active
    on aura_models (name)
    where is_active;

create index if not exists aura_models_lookup
    on aura_models (name, horizon_ms, created_at desc);

-- Decomposed forms of the two jsonb columns. Populated by trigger, never written
-- directly, so the pipeline keeps working while the relational shape stays available.
create table if not exists aura_model_metrics (
    model_id     uuid not null references aura_models (id) on delete cascade,
    metric_name  text not null,
    metric_value double precision not null,
    primary key (model_id, metric_name)
);

create table if not exists aura_model_features (
    model_id     uuid     not null references aura_models (id) on delete cascade,
    position     smallint not null check (position >= 0),
    feature_name text     not null,
    primary key (model_id, position),
    constraint aura_model_features_unique_name unique (model_id, feature_name)
);

create or replace function aura_fan_out_model() returns trigger
language plpgsql as $$
begin
    delete from aura_model_metrics  where model_id = new.id;
    delete from aura_model_features where model_id = new.id;

    insert into aura_model_metrics (model_id, metric_name, metric_value)
    select new.id, key, value::text::double precision
    from jsonb_each(coalesce(new.metrics, '{}'::jsonb))
    where jsonb_typeof(value) = 'number';

    insert into aura_model_features (model_id, position, feature_name)
    select new.id, (ord - 1)::smallint, feat #>> '{}'
    from jsonb_array_elements(coalesce(new.feature_names, '[]'::jsonb))
         with ordinality as t(feat, ord);

    return new;
end $$;

drop trigger if exists aura_models_fan_out on aura_models;
create trigger aura_models_fan_out
    after insert or update of metrics, feature_names on aura_models
    for each row execute function aura_fan_out_model();

-- ============================================================================
-- 3. Benchmarks
--    push_benchmark_run() upserts on run_id, so run_id carries the unique constraint
--    and is what the results table points at.
-- ============================================================================

create table if not exists aura_benchmark_runs (
    id             uuid        primary key default gen_random_uuid(),
    run_id         text        not null unique,
    scenario       text        not null references aura_scenarios (code),
    seed           bigint      not null default 0,
    capacity_bytes bigint      not null default 0 check (capacity_bytes >= 0),
    requests       bigint      not null default 0 check (requests >= 0),
    engine_version text        not null default 'unknown',
    summary        jsonb       not null default '{}'::jsonb,
    created_at     timestamptz not null default now()
);

create index if not exists aura_benchmark_runs_scenario
    on aura_benchmark_runs (scenario, created_at desc);

-- One row per (run, policy). That composite key is what keeps this in third normal
-- form rather than a wide run table carrying lru_cost, lfu_cost, gdsf_cost columns.
create table if not exists aura_benchmark_results (
    run_id               text not null references aura_benchmark_runs (run_id) on delete cascade,
    policy               text not null references aura_policies (code),
    object_hit_rate      double precision check (object_hit_rate between 0 and 1),
    byte_hit_rate        double precision check (byte_hit_rate between 0 and 1),
    p95_latency_ms       double precision check (p95_latency_ms >= 0),
    backend_requests     bigint           check (backend_requests >= 0),
    total_cost_usd       double precision check (total_cost_usd >= 0),
    regen_cost_usd       double precision check (regen_cost_usd >= 0),
    sla_penalty_usd      double precision default 0 check (sla_penalty_usd >= 0),
    decision_overhead_us double precision,
    extra                jsonb not null default '{}'::jsonb,
    primary key (run_id, policy)
);

-- ============================================================================
-- 4. Traces and events
-- ============================================================================

create table if not exists aura_traces (
    id           uuid        primary key default gen_random_uuid(),
    name         text        not null unique,
    scenario     text        not null references aura_scenarios (code),
    storage_path text        not null,
    rows         bigint      not null default 0 check (rows >= 0),
    unique_keys  bigint      not null default 0 check (unique_keys >= 0),
    bytes        bigint      not null default 0 check (bytes >= 0),
    meta         jsonb       not null default '{}'::jsonb,
    created_at   timestamptz not null default now()
);

create table if not exists aura_events (
    id     bigserial   primary key,
    ts     timestamptz not null default now(),
    kind   text        not null references aura_event_kinds (code),
    detail jsonb       not null default '{}'::jsonb
);

create index if not exists aura_events_recent on aura_events (ts desc);
create index if not exists aura_events_kind   on aura_events (kind, ts desc);

-- Convenience view for the results page: one row per policy with the run's context
-- already joined on, so the dashboard does not have to assemble it.
create or replace view aura_benchmark_report as
select
    r.run_id,
    r.scenario,
    s.name as scenario_name,
    r.seed,
    r.capacity_bytes,
    r.requests,
    r.engine_version,
    r.created_at,
    b.policy,
    p.family as policy_family,
    b.object_hit_rate,
    b.byte_hit_rate,
    b.p95_latency_ms,
    b.backend_requests,
    b.total_cost_usd,
    b.regen_cost_usd,
    b.sla_penalty_usd,
    b.decision_overhead_us
from aura_benchmark_runs r
join aura_benchmark_results b on b.run_id = r.run_id
join aura_scenarios s on s.code = r.scenario
join aura_policies  p on p.code = b.policy;

-- ============================================================================
-- 5. Analytics workload
--    The analytics service queries these on a cache miss. Volume and index shape
--    are what make that query cost a realistic few hundred milliseconds.
-- ============================================================================

create table if not exists app_countries (
    country_id smallint primary key,
    iso_code   char(2)  not null unique,
    name       text     not null unique
);

create table if not exists app_regions (
    region_id  serial   primary key,
    country_id smallint not null references app_countries (country_id),
    name       text     not null,
    constraint app_regions_unique_name_per_country unique (country_id, name)
);

create table if not exists app_categories (
    category_id smallint primary key,
    name        text     not null unique
);

create table if not exists app_products (
    product_id  serial         primary key,
    category_id smallint       not null references app_categories (category_id),
    sku         text           not null unique,
    name        text           not null,
    unit_price  numeric(10, 2) not null check (unit_price >= 0)
);

create table if not exists app_customers (
    customer_id  bigserial   primary key,
    region_id    integer     not null references app_regions (region_id),
    external_ref text        not null unique,
    created_at   timestamptz not null default now()
);

-- Orders carry no product or price columns; line items do. That split is the 3NF
-- point: order-level facts depend on the order, line-level facts on the line.
create table if not exists app_orders (
    order_id    bigserial   primary key,
    customer_id bigint      not null references app_customers (customer_id),
    region_id   integer     not null references app_regions (region_id),
    placed_at   timestamptz not null default now(),
    status      text        not null default 'complete'
                            check (status in ('pending', 'complete', 'cancelled'))
);

create table if not exists app_order_items (
    order_id   bigint         not null references app_orders (order_id) on delete cascade,
    line_no    smallint       not null check (line_no > 0),
    product_id integer        not null references app_products (product_id),
    quantity   integer        not null check (quantity > 0),
    -- The price at the time of sale, which is a different fact from the current
    -- catalogue price. Duplicating it here is correct, not a normalisation slip.
    unit_price numeric(10, 2) not null check (unit_price >= 0),
    primary key (order_id, line_no)
);

create index if not exists app_orders_region_time  on app_orders (region_id, placed_at desc);
create index if not exists app_orders_customer     on app_orders (customer_id, placed_at desc);
create index if not exists app_order_items_product on app_order_items (product_id);
create index if not exists app_products_category   on app_products (category_id);
create index if not exists app_customers_region    on app_customers (region_id);

-- Line and order totals are computable from their own columns, so they are a view.
-- Storing them would be the 3NF violation.
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
-- 6. Storage buckets
--    upload_bundle() and upload_trace() call ensure_bucket() first, but creating
--    them here means a fresh project is ready before the notebook runs.
-- ============================================================================

do $$
begin
    if exists (select 1 from information_schema.tables
               where table_schema = 'storage' and table_name = 'buckets') then
        insert into storage.buckets (id, name, public)
        values ('aura-models', 'aura-models', false),
               ('aura-traces', 'aura-traces', false)
        on conflict (id) do nothing;
    else
        raise notice 'no storage schema here; buckets are created by the client instead';
    end if;
end $$;

-- ============================================================================
-- 7. Row level security
--    anon and authenticated may read. Writes come from the service role, which
--    bypasses RLS entirely, so no policy grants anon a write.
-- ============================================================================

alter table aura_models            enable row level security;
alter table aura_model_metrics     enable row level security;
alter table aura_model_features    enable row level security;
alter table aura_benchmark_runs    enable row level security;
alter table aura_benchmark_results enable row level security;
alter table aura_traces            enable row level security;
alter table aura_events            enable row level security;
alter table aura_scenarios         enable row level security;
alter table aura_policies          enable row level security;
alter table aura_model_kinds       enable row level security;
alter table aura_event_kinds       enable row level security;

-- The anon and authenticated roles exist on Supabase but not on a plain Postgres, so
-- the same file also applies to a local Docker instance without editing.
do $$
declare
    t     text;
    roles text;
begin
    select string_agg(quote_ident(rolname), ', ')
      into roles
      from pg_roles
     where rolname in ('anon', 'authenticated');

    if roles is null then
        raise notice 'skipping read policies: neither anon nor authenticated exists here';
        return;
    end if;

    foreach t in array array[
        'aura_models', 'aura_model_metrics', 'aura_model_features',
        'aura_benchmark_runs', 'aura_benchmark_results', 'aura_traces', 'aura_events',
        'aura_scenarios', 'aura_policies', 'aura_model_kinds', 'aura_event_kinds'
    ] loop
        execute format('drop policy if exists %I on %I', t || '_read', t);
        execute format(
            'create policy %I on %I for select to %s using (true)',
            t || '_read', t, roles);
    end loop;
end $$;

commit;
