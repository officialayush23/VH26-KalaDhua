-- Seed data for the analytics example app.
--
--   psql "$SUPABASE_DIRECT_CONNECTION_URL" -f sql/002_seed_analytics.sql
--
-- The analytics app's /work/{id} endpoint runs real GROUP BY aggregations over
-- app_orders. For the demo to mean anything those queries have to be genuinely
-- expensive -- a few hundred milliseconds of database time, not a few hundred
-- microseconds -- because the whole premise of AURA is that it decides what to
-- cache by how much regenerating it costs. ~200k orders over 18 months across
-- 12 regions and 60 products puts a regional monthly rollup in the right range
-- on a small Supabase instance.
--
-- Re-running this truncates and reseeds, so the row count stays deterministic.

begin;

truncate table public.app_orders restart identity;
truncate table public.app_products restart identity cascade;
truncate table public.app_regions restart identity cascade;

insert into public.app_regions (name, country) values
    ('North East',    'US'),
    ('South East',    'US'),
    ('Midwest',       'US'),
    ('Pacific',       'US'),
    ('Ontario',       'CA'),
    ('Quebec',        'CA'),
    ('Greater London','GB'),
    ('Bavaria',       'DE'),
    ('Ile-de-France', 'FR'),
    ('Maharashtra',   'IN'),
    ('Karnataka',     'IN'),
    ('New South Wales','AU');

-- 60 products across 6 categories, prices spanning two orders of magnitude so
-- that revenue aggregates are skewed the way real ones are.
insert into public.app_products (name, category, unit_price)
select
    format('%s-%s', category, lpad(n::text, 3, '0')) as name,
    category,
    round((4.5 * exp(ln(220.0 / 4.5) * ((n % 10)::numeric / 9)))::numeric, 2) as unit_price
from generate_series(1, 60) as n
cross join lateral (
    select (array['audio', 'video', 'storage', 'network', 'compute', 'support'])[
        1 + ((n - 1) / 10)
    ] as category
) c;

-- 200,000 orders. Three deliberate structures, because a uniform random table
-- makes every cache key equally valuable and the demo then proves nothing:
--   * order volume grows over the 18 months and spikes at quarter ends,
--   * region popularity is Zipf-ish, so a few regional rollups are hot,
--   * product popularity is skewed independently of region.
insert into public.app_orders (region_id, product_id, qty, amount, created_at)
select
    region_id,
    product_id,
    qty,
    round(qty * p.unit_price, 2) as amount,
    created_at
from (
    select
        -- Zipf-ish over 12 regions: floor(12 ^ u) lands most rows on region 1-3.
        least(12, greatest(1, floor(power(12.0, random()))::int))          as region_id,
        least(60, greatest(1, floor(power(60.0, random()))::int))          as product_id,
        1 + floor(random() * 9)::int                                       as qty,
        (
            timestamptz '2025-01-01 00:00:00+00'
            + (interval '18 months' * power(random(), 0.75))
            + (interval '1 day' * case when random() < 0.08 then 0 else random() * 3 end)
        )                                                                  as created_at
    from generate_series(1, 200000)
) o
join public.app_products p on p.id = o.product_id;

analyze public.app_orders;
analyze public.app_products;
analyze public.app_regions;

commit;

-- Sanity output. The analytics app's most expensive query is roughly this one,
-- so if it comes back instantly the seed did not do its job.
select
    (select count(*) from public.app_orders)   as orders,
    (select count(*) from public.app_products) as products,
    (select count(*) from public.app_regions)  as regions,
    (select min(created_at) from public.app_orders) as first_order,
    (select max(created_at) from public.app_orders) as last_order;

explain analyze
select
    r.name,
    date_trunc('month', o.created_at) as month,
    sum(o.amount)                     as revenue,
    count(*)                          as orders
from public.app_orders o
join public.app_regions r on r.id = o.region_id
where o.created_at >= now() - interval '12 months'
group by 1, 2
order by 2, 3 desc;
