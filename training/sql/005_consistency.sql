-- AURA consistency plane: turning a database write into a cache invalidation.
--
-- This file answers the question the whole project was missing: someone opens the SQL
-- editor, changes a price from 40 to 20, and the cache is now serving a number that is
-- wrong. TTL does not fix that -- it only bounds how long we stay wrong.
--
-- The mechanism is deliberately boring, because correctness should be:
--
--   1. Every row that cached objects can be derived from carries a `version` column.
--      Any change bumps it. That gives a cheap way to ask "is what I hold still current"
--      without comparing whole rows.
--
--   2. A trigger emits a NOTIFY carrying a dependency *tag* -- `row:product:1292` --
--      rather than a cache key. The database has no idea what keys exist, and should not.
--      The engine's inverted index turns one tag into every object built from that row.
--
--   3. A listener process (apps/invalidator) forwards the tag to POST /v1/invalidate.
--      It is deliberately outside the request path: an invalidation arriving 50 ms late
--      is fine, a GET blocked on a database round trip is not.
--
-- Run order: 003_supabase_schema.sql, 004_supabase_seed.sql, then this file.
-- Idempotent: safe to run more than once.

begin;

-- ---------------------------------------------------------------------------
-- 1. Version columns
-- ---------------------------------------------------------------------------
-- `version` increments on every update. `updated_at` is for humans and for the
-- version-comparison safety net described in section 5.

alter table if exists app_products
    add column if not exists version    bigint      not null default 1,
    add column if not exists updated_at timestamptz not null default now();

alter table if exists app_regions
    add column if not exists version    bigint      not null default 1,
    add column if not exists updated_at timestamptz not null default now();

alter table if exists app_orders
    add column if not exists version    bigint      not null default 1,
    add column if not exists updated_at timestamptz not null default now();

-- ---------------------------------------------------------------------------
-- 2. The tag emitter
-- ---------------------------------------------------------------------------
-- One generic trigger for every table. The tag format is `row:<entity>:<id>`, matching
-- what applications pass in `depends_on` when they write to the cache. TG_ARGV[0] is the
-- entity name, so `app_products` emits `row:product:1292` rather than
-- `row:app_products:1292` -- the application's vocabulary, not the schema's.
--
-- Payload is JSON so the listener can decide between a hard and a soft invalidation
-- without a second lookup:
--
--   {"tag": "row:product:1292", "entity": "product", "id": "1292",
--    "op": "UPDATE", "version": 18, "mode": "hard", "table": "app_products"}
--
-- NOTIFY payloads are capped at 8000 bytes, and these are ~150, so there is no risk of
-- truncation. Sending the tag rather than the row is also what keeps that true.

create or replace function aura_notify_change() returns trigger
language plpgsql
as $$
declare
    entity   text := tg_argv[0];
    mode     text := coalesce(tg_argv[1], 'hard');
    rec      record;
    row_id   text;
    row_ver  bigint;
    payload  json;
begin
    if tg_op = 'DELETE' then
        rec := old;
    else
        rec := new;
    end if;

    row_id  := rec.id::text;
    begin
        row_ver := rec.version;
    exception when others then
        row_ver := null;
    end;

    payload := json_build_object(
        'tag',     'row:' || entity || ':' || row_id,
        'entity',  entity,
        'id',      row_id,
        'op',      tg_op,
        'version', row_ver,
        'mode',    mode,
        'table',   tg_table_name,
        'at',      extract(epoch from clock_timestamp())
    );

    -- One channel for everything. A single listener is simpler to reason about than one
    -- connection per table, and the volume here is orders of magnitude below what a
    -- Postgres notification channel handles comfortably.
    perform pg_notify('aura_invalidate', payload::text);
    return rec;
end;
$$;

-- Bump `version` and `updated_at` on every update. Kept separate from the notifier so a
-- table can have one without the other.
create or replace function aura_bump_version() returns trigger
language plpgsql
as $$
begin
    new.version    := coalesce(old.version, 0) + 1;
    new.updated_at := now();
    return new;
end;
$$;

-- ---------------------------------------------------------------------------
-- 3. Wire the triggers
-- ---------------------------------------------------------------------------
-- Prices and regions are `hard`: being wrong about them is a correctness bug, so the
-- cached object is removed outright.
--
-- Orders are `soft`: dashboards are derived, tolerant, and enormously more numerous. A
-- hard invalidation on every order insert would evict rollups faster than they could be
-- rebuilt, and each rebuild is the expensive query we are trying to avoid. Soft marks
-- them stale: the next reader gets the old answer once while a rebuild runs behind it.
--
-- That difference is not a detail. It is the whole argument for having two modes.

drop trigger if exists app_products_version on app_products;
create trigger app_products_version
    before update on app_products
    for each row execute function aura_bump_version();

drop trigger if exists app_products_notify on app_products;
create trigger app_products_notify
    after insert or update or delete on app_products
    for each row execute function aura_notify_change('product', 'hard');

drop trigger if exists app_regions_version on app_regions;
create trigger app_regions_version
    before update on app_regions
    for each row execute function aura_bump_version();

drop trigger if exists app_regions_notify on app_regions;
create trigger app_regions_notify
    after insert or update or delete on app_regions
    for each row execute function aura_notify_change('region', 'hard');

drop trigger if exists app_orders_notify on app_orders;
create trigger app_orders_notify
    after insert or update or delete on app_orders
    for each row execute function aura_notify_change('order', 'soft');

-- ---------------------------------------------------------------------------
-- 4. Namespace versions
-- ---------------------------------------------------------------------------
-- Retiring a generation of objects without deleting any of them. When the recommendation
-- model is redeployed, bumping this row moves every new key to `:v8`; the `:v7` generation
-- becomes unreachable and ages out under ordinary eviction pressure.
--
-- Deleting them instead would empty a large fraction of the cache at once and send the
-- entire miss stream at the ranking service -- the cache causing the outage it exists to
-- prevent.

create table if not exists aura_namespace_versions (
    namespace   text        primary key,
    version     bigint      not null default 1,
    reason      text,
    updated_at  timestamptz not null default now()
);

insert into aura_namespace_versions (namespace, version, reason)
values
    ('recommendation', 1, 'initial'),
    ('analytics',      1, 'initial')
on conflict (namespace) do nothing;

create or replace function aura_bump_namespace(ns text, why text default null)
returns bigint
language plpgsql
as $$
declare
    next_version bigint;
begin
    insert into aura_namespace_versions as v (namespace, version, reason, updated_at)
    values (ns, 2, why, now())
    on conflict (namespace) do update
        set version    = v.version + 1,
            reason     = coalesce(excluded.reason, v.reason),
            updated_at = now()
    returning version into next_version;

    perform pg_notify(
        'aura_invalidate',
        json_build_object(
            'tag',       'namespace:' || ns,
            'entity',    'namespace',
            'id',        ns,
            'op',        'VERSION',
            'version',   next_version,
            'mode',      'version',
            'at',        extract(epoch from clock_timestamp())
        )::text
    );
    return next_version;
end;
$$;

-- ---------------------------------------------------------------------------
-- 5. The safety net
-- ---------------------------------------------------------------------------
-- Triggers cover writes that reach this database. They do not cover a restore, a
-- replication catch-up, a listener that was down, or a bulk load with triggers disabled.
--
-- For the handful of objects where being wrong is unacceptable, the application can check
-- the version it cached against the current one in a single cheap query rather than
-- refetching the row. Use this sparingly: it is a network round trip, which is exactly
-- what the cache exists to remove.

create or replace function aura_current_versions(entity text, ids bigint[])
returns table (id bigint, version bigint)
language plpgsql
as $$
begin
    if entity = 'product' then
        return query select p.id, p.version from app_products p where p.id = any(ids);
    elsif entity = 'region' then
        return query select r.id, r.version from app_regions r where r.id = any(ids);
    else
        raise exception 'aura_current_versions: unknown entity %', entity;
    end if;
end;
$$;

-- ---------------------------------------------------------------------------
-- 6. An audit trail
-- ---------------------------------------------------------------------------
-- What was invalidated, when, and why. The dashboard's consistency panel reads this, and
-- it is the difference between claiming the mechanism works and showing that it did.

create table if not exists aura_invalidation_log (
    id           bigserial   primary key,
    tag          text        not null,
    entity       text,
    row_id       text,
    op           text,
    mode         text,
    keys_matched integer,
    source       text        not null default 'postgres',
    created_at   timestamptz not null default now()
);

create index if not exists aura_invalidation_log_created_idx
    on aura_invalidation_log (created_at desc);
create index if not exists aura_invalidation_log_tag_idx
    on aura_invalidation_log (tag, created_at desc);

commit;

-- ---------------------------------------------------------------------------
-- Verifying it, by hand
-- ---------------------------------------------------------------------------
-- In one psql session:
--
--     listen aura_invalidate;
--
-- In another, or in the Supabase SQL editor:
--
--     update app_products set unit_price = 20 where id = 1292;
--
-- The first session receives:
--
--     Asynchronous notification "aura_invalidate" with payload
--     {"tag":"row:product:1292","entity":"product","id":"1292","op":"UPDATE",
--      "version":2,"mode":"hard","table":"app_products",...}
--
-- With apps/invalidator running, that same payload reaches POST /v1/invalidate and every
-- cached object that declared `row:product:1292` as a dependency is dropped -- including
-- rollups and rankings built from it, which no TTL would have caught in time.
--
-- To retire a model generation:
--
--     select aura_bump_namespace('recommendation', 'ranker v8 deployed');
