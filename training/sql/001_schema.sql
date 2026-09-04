-- AURA Supabase schema (contract section 6).
--
-- Apply with either:
--   psql "$SUPABASE_DIRECT_CONNECTION_URL" -f sql/001_schema.sql
-- or by pasting it into the Supabase SQL editor.
--
-- Everything is idempotent, so re-running it after a contract change is safe.
--
-- Security model: every table below is written by the training pipeline and the
-- engine, both of which authenticate with the service role key and therefore
-- bypass RLS entirely. RLS is still enabled on every table, because a Supabase
-- project exposes PostgREST publicly and a table without RLS is a table anyone
-- with the anon key can read. The policies grant read-only access to
-- authenticated dashboard users and nothing at all to anon.

begin;

create extension if not exists "pgcrypto";

-- ---------------------------------------------------------------------------
-- Models
-- ---------------------------------------------------------------------------

create table if not exists public.aura_models (
    id            uuid primary key default gen_random_uuid(),
    name          text        not null,
    kind          text        not null,
    horizon_ms    integer     not null,
    version       text        not null,
    storage_path  text        not null,
    onnx_path     text,
    metrics       jsonb       not null default '{}'::jsonb,
    feature_names jsonb       not null default '[]'::jsonb,
    is_active     boolean     not null default false,
    created_at    timestamptz not null default now(),
    constraint aura_models_name_version_key unique (name, version)
);

create index if not exists aura_models_name_created_idx
    on public.aura_models (name, created_at desc);

-- At most one active version per model name. This is the invariant the engine
-- relies on when it boots and asks "which bundle am I supposed to run?".
create unique index if not exists aura_models_one_active_per_name
    on public.aura_models (name) where is_active;

-- ---------------------------------------------------------------------------
-- Benchmarks
-- ---------------------------------------------------------------------------

create table if not exists public.aura_benchmark_runs (
    id             uuid primary key default gen_random_uuid(),
    run_id         text        not null unique,
    scenario       text        not null,
    seed           integer     not null default 0,
    capacity_bytes bigint      not null default 0,
    requests       bigint      not null default 0,
    engine_version text,
    created_at     timestamptz not null default now(),
    summary        jsonb       not null default '{}'::jsonb
);

create index if not exists aura_benchmark_runs_created_idx
    on public.aura_benchmark_runs (created_at desc);

create table if not exists public.aura_benchmark_results (
    id                   bigserial primary key,
    run_id               text not null
        references public.aura_benchmark_runs (run_id) on delete cascade,
    policy               text not null,
    object_hit_rate      double precision,
    byte_hit_rate        double precision,
    p95_latency_ms       double precision,
    backend_requests     bigint,
    total_cost_usd       double precision,
    regen_cost_usd       double precision,
    sla_penalty_usd      double precision,
    decision_overhead_us double precision,
    extra                jsonb not null default '{}'::jsonb
);

create index if not exists aura_benchmark_results_run_idx
    on public.aura_benchmark_results (run_id, policy);

-- ---------------------------------------------------------------------------
-- Traces and events
-- ---------------------------------------------------------------------------

create table if not exists public.aura_traces (
    id           uuid primary key default gen_random_uuid(),
    name         text        not null,
    scenario     text        not null,
    storage_path text        not null,
    rows         bigint      not null default 0,
    unique_keys  bigint      not null default 0,
    bytes        bigint      not null default 0,
    meta         jsonb       not null default '{}'::jsonb,
    created_at   timestamptz not null default now()
);

create index if not exists aura_traces_scenario_idx
    on public.aura_traces (scenario, created_at desc);

create table if not exists public.aura_events (
    id     bigserial primary key,
    ts     timestamptz not null default now(),
    kind   text        not null,
    detail jsonb       not null default '{}'::jsonb
);

create index if not exists aura_events_ts_idx on public.aura_events (ts desc);
create index if not exists aura_events_kind_idx on public.aura_events (kind, ts desc);

-- ---------------------------------------------------------------------------
-- Analytics workload tables (the analytics example app runs real SQL on these)
-- ---------------------------------------------------------------------------

create table if not exists public.app_regions (
    id      serial primary key,
    name    text not null,
    country text not null
);

create table if not exists public.app_products (
    id         serial primary key,
    name       text    not null,
    category   text    not null,
    unit_price numeric(12, 2) not null
);

create table if not exists public.app_orders (
    id         bigserial primary key,
    region_id  int not null references public.app_regions (id),
    product_id int not null references public.app_products (id),
    qty        int not null,
    amount     numeric(14, 2) not null,
    created_at timestamptz not null default now()
);

-- The two indexes the analytics app's GROUP BY queries actually need. Without
-- them the app is measuring sequential scan speed rather than cache behaviour.
create index if not exists app_orders_region_created_idx
    on public.app_orders (region_id, created_at);
create index if not exists app_orders_product_created_idx
    on public.app_orders (product_id, created_at);

-- ---------------------------------------------------------------------------
-- Row level security
-- ---------------------------------------------------------------------------

alter table public.aura_models            enable row level security;
alter table public.aura_benchmark_runs    enable row level security;
alter table public.aura_benchmark_results enable row level security;
alter table public.aura_traces            enable row level security;
alter table public.aura_events            enable row level security;
alter table public.app_regions            enable row level security;
alter table public.app_products           enable row level security;
alter table public.app_orders             enable row level security;

do $$
declare
    tbl text;
begin
    foreach tbl in array array[
        'aura_models', 'aura_benchmark_runs', 'aura_benchmark_results',
        'aura_traces', 'aura_events', 'app_regions', 'app_products', 'app_orders'
    ] loop
        execute format('drop policy if exists %I on public.%I', tbl || '_read', tbl);
        execute format(
            'create policy %I on public.%I for select to authenticated using (true)',
            tbl || '_read', tbl
        );
    end loop;
end;
$$;

-- Writes are service-role only, which bypasses RLS, so there are deliberately
-- no insert/update/delete policies here. If you ever need a client to write,
-- add a narrow policy for that one table rather than widening these.

commit;
