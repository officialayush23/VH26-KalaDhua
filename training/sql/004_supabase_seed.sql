-- AURA — analytics workload seed.
-- Run after supabase_schema.sql. Idempotent: truncates the app_* tables first.
--
-- Volume is the point. The analytics service has to run a query that genuinely costs
-- a few hundred milliseconds, otherwise the regeneration cost the cache learns is noise.
-- Defaults below produce roughly 400k order lines, which puts a regional rollup in the
-- 200-600 ms range on a small Supabase instance.

begin;

truncate app_order_items, app_orders, app_customers, app_products,
         app_regions, app_categories, app_countries restart identity cascade;

insert into app_countries (country_id, iso_code, name) values
    (1, 'IN', 'India'), (2, 'US', 'United States'), (3, 'GB', 'United Kingdom'),
    (4, 'DE', 'Germany'), (5, 'SG', 'Singapore');

insert into app_regions (country_id, name)
select c.country_id, r.name
from app_countries c
cross join (values ('North'), ('South'), ('East'), ('West'), ('Central')) as r(name);

insert into app_categories (category_id, name) values
    (1, 'apparel'), (2, 'footwear'), (3, 'accessories'),
    (4, 'electronics'), (5, 'home'), (6, 'outdoor');

insert into app_products (category_id, sku, name, unit_price)
select
    ((n % 6) + 1)::smallint,
    'SKU-' || lpad(n::text, 6, '0'),
    'Product ' || n,
    round((15 + (n % 400) * 1.75)::numeric, 2)
from generate_series(1, 4000) as n;

insert into app_customers (region_id, external_ref)
select
    1 + (n % (select count(*) from app_regions))::int,
    'CUST-' || lpad(n::text, 7, '0')
from generate_series(1, 40000) as n;

-- Orders spread across 180 days so time-window queries have something to scan.
insert into app_orders (customer_id, region_id, placed_at, status)
select
    c.customer_id,
    c.region_id,
    now() - (random() * interval '180 days'),
    case when random() < 0.04 then 'cancelled' else 'complete' end
from app_customers c
cross join generate_series(1, 3) as rep;

-- Between one and five lines per order, skewed toward the low end.
insert into app_order_items (order_id, line_no, product_id, quantity, unit_price)
select
    o.order_id,
    ln::smallint,
    p.product_id,
    1 + floor(random() * 4)::int,
    p.unit_price
from app_orders o
cross join lateral generate_series(1, 1 + floor(random() * 4)::int) as ln
join app_products p
  on p.product_id = 1 + ((o.order_id * 7 + ln * 13) % 4000);

analyze app_countries;
analyze app_regions;
analyze app_categories;
analyze app_products;
analyze app_customers;
analyze app_orders;
analyze app_order_items;

commit;

-- Sanity check. Expect roughly: 25 regions, 4000 products, 40000 customers,
-- 120000 orders, ~300000 order lines.
select 'regions'     as table_name, count(*) from app_regions
union all select 'products',   count(*) from app_products
union all select 'customers',  count(*) from app_customers
union all select 'orders',     count(*) from app_orders
union all select 'order_items', count(*) from app_order_items;
