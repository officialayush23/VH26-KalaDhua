"""The dashboard queries.

Real SQL: multi-table joins, GROUP BY, and window functions over
`app_orders`, `app_products` and `app_regions`. Each query is written twice -
once for Postgres, once for SQLite - because the fallback fixture must run the
same shape of work, not a simplified version of it.

Nothing here is an ORM. The statement that executes is the statement in this
file, which is also what makes the measured `db_ms` meaningful.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from analytics.db import Backend, QueryResult

# Windows a dashboard actually offers.
WINDOW_DAYS = (1, 7, 30, 90, 365)
REGION_COUNT = 12


@dataclass(frozen=True)
class Query:
    """One dashboard query in both dialects."""

    name: str
    object_type: str
    sql_postgres: str
    sql_sqlite: str
    ttl_ms: int
    sla_class: str

    def sql_for(self, dialect: str) -> str:
        """SQL text for the active backend."""
        return self.sql_postgres if dialect == "postgres" else self.sql_sqlite


REVENUE_BY_REGION = Query(
    name="revenue_by_region",
    object_type="dashboard_query",
    ttl_ms=600_000,
    sla_class="high",
    sql_postgres="""
        WITH windowed AS (
            SELECT r.id AS region_id, r.name AS region, r.country,
                   p.category AS category,
                   sum(o.amount) AS revenue,
                   sum(o.qty) AS units,
                   count(*) AS orders
            FROM app_orders o
            JOIN app_regions r ON r.id = o.region_id
            JOIN app_products p ON p.id = o.product_id
            WHERE o.created_at >= now() - make_interval(days => $1)
            GROUP BY r.id, r.name, r.country, p.category
        )
        SELECT region, country, category,
               round(revenue::numeric, 2) AS revenue,
               units, orders,
               rank() OVER (PARTITION BY region ORDER BY revenue DESC) AS category_rank,
               round((revenue / NULLIF(sum(revenue) OVER (PARTITION BY region), 0) * 100)::numeric, 3)
                   AS pct_of_region,
               round(sum(revenue) OVER (PARTITION BY region ORDER BY revenue DESC
                                        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)::numeric, 2)
                   AS running_revenue
        FROM windowed
        ORDER BY region, category_rank
        LIMIT $2
    """,
    sql_sqlite="""
        WITH windowed AS (
            SELECT r.id AS region_id, r.name AS region, r.country,
                   p.category AS category,
                   sum(o.amount) AS revenue,
                   sum(o.qty) AS units,
                   count(*) AS orders
            FROM app_orders o
            JOIN app_regions r ON r.id = o.region_id
            JOIN app_products p ON p.id = o.product_id
            WHERE o.created_at >= datetime('now', ?)
            GROUP BY r.id, r.name, r.country, p.category
        )
        SELECT region, country, category,
               round(revenue, 2) AS revenue,
               units, orders,
               rank() OVER (PARTITION BY region ORDER BY revenue DESC) AS category_rank,
               round(revenue / NULLIF(sum(revenue) OVER (PARTITION BY region), 0) * 100, 3)
                   AS pct_of_region,
               round(sum(revenue) OVER (PARTITION BY region ORDER BY revenue DESC
                                        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW), 2)
                   AS running_revenue
        FROM windowed
        ORDER BY region, category_rank
        LIMIT ?
    """,
)


TOP_PRODUCTS = Query(
    name="top_products",
    object_type="dashboard_query",
    ttl_ms=300_000,
    sla_class="high",
    sql_postgres="""
        WITH sales AS (
            SELECT p.id, p.name, p.category, p.unit_price,
                   sum(o.amount) AS revenue,
                   sum(o.qty) AS units,
                   count(DISTINCT o.region_id) AS regions
            FROM app_orders o
            JOIN app_products p ON p.id = o.product_id
            WHERE o.region_id = $1
              AND o.created_at >= now() - make_interval(days => $2)
            GROUP BY p.id, p.name, p.category, p.unit_price
        )
        SELECT name, category, unit_price,
               round(revenue::numeric, 2) AS revenue, units, regions,
               row_number() OVER (ORDER BY revenue DESC) AS overall_rank,
               rank() OVER (PARTITION BY category ORDER BY revenue DESC) AS category_rank,
               round(percent_rank() OVER (ORDER BY revenue)::numeric, 4) AS revenue_percentile,
               round((revenue - lag(revenue) OVER (ORDER BY revenue DESC))::numeric, 2) AS gap_to_previous
        FROM sales
        ORDER BY revenue DESC
        LIMIT $3
    """,
    sql_sqlite="""
        WITH sales AS (
            SELECT p.id, p.name, p.category, p.unit_price,
                   sum(o.amount) AS revenue,
                   sum(o.qty) AS units,
                   count(DISTINCT o.region_id) AS regions
            FROM app_orders o
            JOIN app_products p ON p.id = o.product_id
            WHERE o.region_id = ?
              AND o.created_at >= datetime('now', ?)
            GROUP BY p.id, p.name, p.category, p.unit_price
        )
        SELECT name, category, unit_price,
               round(revenue, 2) AS revenue, units, regions,
               row_number() OVER (ORDER BY revenue DESC) AS overall_rank,
               rank() OVER (PARTITION BY category ORDER BY revenue DESC) AS category_rank,
               round(percent_rank() OVER (ORDER BY revenue), 4) AS revenue_percentile,
               round(revenue - lag(revenue) OVER (ORDER BY revenue DESC), 2) AS gap_to_previous
        FROM sales
        ORDER BY revenue DESC
        LIMIT ?
    """,
)


DAILY_TREND = Query(
    name="daily_trend",
    object_type="timeseries",
    ttl_ms=120_000,
    sla_class="critical",
    sql_postgres="""
        WITH daily AS (
            SELECT date_trunc('day', o.created_at)::date AS day,
                   sum(o.amount) AS revenue,
                   count(*) AS orders,
                   count(DISTINCT o.product_id) AS distinct_products
            FROM app_orders o
            WHERE o.region_id = $1
              AND o.created_at >= now() - make_interval(days => $2)
            GROUP BY 1
        )
        SELECT day,
               round(revenue::numeric, 2) AS revenue, orders, distinct_products,
               round(avg(revenue) OVER (ORDER BY day ROWS BETWEEN 6 PRECEDING AND CURRENT ROW)::numeric, 2)
                   AS revenue_ma7,
               round((revenue - lag(revenue) OVER (ORDER BY day))::numeric, 2) AS delta,
               round(sum(revenue) OVER (ORDER BY day)::numeric, 2) AS cumulative_revenue
        FROM daily
        ORDER BY day
        LIMIT $3
    """,
    sql_sqlite="""
        WITH daily AS (
            SELECT date(o.created_at) AS day,
                   sum(o.amount) AS revenue,
                   count(*) AS orders,
                   count(DISTINCT o.product_id) AS distinct_products
            FROM app_orders o
            WHERE o.region_id = ?
              AND o.created_at >= datetime('now', ?)
            GROUP BY 1
        )
        SELECT day,
               round(revenue, 2) AS revenue, orders, distinct_products,
               round(avg(revenue) OVER (ORDER BY day ROWS BETWEEN 6 PRECEDING AND CURRENT ROW), 2)
                   AS revenue_ma7,
               round(revenue - lag(revenue) OVER (ORDER BY day), 2) AS delta,
               round(sum(revenue) OVER (ORDER BY day), 2) AS cumulative_revenue
        FROM daily
        ORDER BY day
        LIMIT ?
    """,
)


CATEGORY_MATRIX = Query(
    name="category_matrix",
    object_type="dashboard_query",
    ttl_ms=900_000,
    sla_class="normal",
    sql_postgres="""
        WITH cell AS (
            SELECT r.name AS region, p.category AS category,
                   sum(o.amount) AS revenue,
                   avg(o.amount) AS avg_order,
                   count(*) AS orders
            FROM app_orders o
            JOIN app_regions r ON r.id = o.region_id
            JOIN app_products p ON p.id = o.product_id
            WHERE o.created_at >= now() - make_interval(days => $1)
            GROUP BY r.name, p.category
        )
        SELECT region, category,
               round(revenue::numeric, 2) AS revenue,
               round(avg_order::numeric, 2) AS avg_order,
               orders,
               round((revenue / NULLIF(sum(revenue) OVER (PARTITION BY category), 0))::numeric, 5)
                   AS region_share_of_category,
               round((revenue / NULLIF(sum(revenue) OVER (PARTITION BY region), 0))::numeric, 5)
                   AS category_share_of_region,
               dense_rank() OVER (PARTITION BY category ORDER BY revenue DESC) AS region_rank,
               round((revenue - avg(revenue) OVER (PARTITION BY category))::numeric, 2) AS vs_category_mean
        FROM cell
        ORDER BY category, region_rank
        LIMIT $2
    """,
    sql_sqlite="""
        WITH cell AS (
            SELECT r.name AS region, p.category AS category,
                   sum(o.amount) AS revenue,
                   avg(o.amount) AS avg_order,
                   count(*) AS orders
            FROM app_orders o
            JOIN app_regions r ON r.id = o.region_id
            JOIN app_products p ON p.id = o.product_id
            WHERE o.created_at >= datetime('now', ?)
            GROUP BY r.name, p.category
        )
        SELECT region, category,
               round(revenue, 2) AS revenue,
               round(avg_order, 2) AS avg_order,
               orders,
               round(revenue / NULLIF(sum(revenue) OVER (PARTITION BY category), 0), 5)
                   AS region_share_of_category,
               round(revenue / NULLIF(sum(revenue) OVER (PARTITION BY region), 0), 5)
                   AS category_share_of_region,
               dense_rank() OVER (PARTITION BY category ORDER BY revenue DESC) AS region_rank,
               round(revenue - avg(revenue) OVER (PARTITION BY category), 2) AS vs_category_mean
        FROM cell
        ORDER BY category, region_rank
        LIMIT ?
    """,
)


COHORT_RETENTION = Query(
    name="cohort_retention",
    object_type="cohort_report",
    ttl_ms=1_800_000,
    sla_class="normal",
    sql_postgres="""
        WITH per_product AS (
            SELECT p.id AS product_id, p.category, r.name AS region,
                   min(o.created_at) AS first_seen,
                   date_trunc('month', o.created_at)::date AS month,
                   sum(o.amount) AS revenue,
                   count(*) AS orders
            FROM app_orders o
            JOIN app_products p ON p.id = o.product_id
            JOIN app_regions r ON r.id = o.region_id
            GROUP BY p.id, p.category, r.name, 5
        ),
        cohort AS (
            SELECT category,
                   date_trunc('month', first_seen)::date AS cohort_month,
                   month,
                   sum(revenue) AS revenue,
                   sum(orders) AS orders
            FROM per_product
            GROUP BY 1, 2, 3
        )
        SELECT category, cohort_month, month,
               round(revenue::numeric, 2) AS revenue, orders,
               round((revenue / NULLIF(first_value(revenue) OVER (
                     PARTITION BY category, cohort_month ORDER BY month), 0))::numeric, 5) AS retention,
               round(avg(revenue) OVER (PARTITION BY category ORDER BY month
                                        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)::numeric, 2) AS revenue_ma3,
               ntile(4) OVER (PARTITION BY category ORDER BY revenue) AS revenue_quartile
        FROM cohort
        ORDER BY category, cohort_month, month
        LIMIT $1
    """,
    sql_sqlite="""
        WITH per_product AS (
            SELECT p.id AS product_id, p.category, r.name AS region,
                   min(o.created_at) AS first_seen,
                   strftime('%Y-%m-01', o.created_at) AS month,
                   sum(o.amount) AS revenue,
                   count(*) AS orders
            FROM app_orders o
            JOIN app_products p ON p.id = o.product_id
            JOIN app_regions r ON r.id = o.region_id
            GROUP BY p.id, p.category, r.name, 5
        ),
        cohort AS (
            SELECT category,
                   strftime('%Y-%m-01', first_seen) AS cohort_month,
                   month,
                   sum(revenue) AS revenue,
                   sum(orders) AS orders
            FROM per_product
            GROUP BY 1, 2, 3
        )
        SELECT category, cohort_month, month,
               round(revenue, 2) AS revenue, orders,
               round(revenue / NULLIF(first_value(revenue) OVER (
                     PARTITION BY category, cohort_month ORDER BY month), 0), 5) AS retention,
               round(avg(revenue) OVER (PARTITION BY category ORDER BY month
                                        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW), 2) AS revenue_ma3,
               ntile(4) OVER (PARTITION BY category ORDER BY revenue) AS revenue_quartile
        FROM cohort
        ORDER BY category, cohort_month, month
        LIMIT ?
    """,
)


QUERIES: dict[str, Query] = {
    q.name: q for q in (REVENUE_BY_REGION, TOP_PRODUCTS, DAILY_TREND, CATEGORY_MATRIX, COHORT_RETENTION)
}

# `cohort_retention` deliberately has no date predicate: it scans the whole order
# book and windows over it, which is what makes it the expensive query. It is the
# natural expensive tail for this application.
EXPENSIVE = "cohort_retention"

_ROTATION = ("revenue_by_region", "top_products", "daily_trend", "category_matrix")


@dataclass(frozen=True)
class QueryPlan:
    """A concrete query with its parameters and its cache key."""

    query: Query
    params_postgres: tuple[Any, ...]
    params_sqlite: tuple[Any, ...]
    cache_key: str
    label: str

    def params_for(self, dialect: str) -> Sequence[Any]:
        """Bound parameters for the active dialect."""
        return self.params_postgres if dialect == "postgres" else self.params_sqlite


def plan_for(key_id: int, *, expensive: bool, limit: int) -> QueryPlan:
    """Map a numeric key id onto a concrete dashboard query.

    Deterministic, so the same id always addresses the same object - which is
    what gives the analytics workload its strong temporal locality.
    """
    if expensive:
        return QueryPlan(
            query=QUERIES[EXPENSIVE],
            params_postgres=(limit,),
            params_sqlite=(limit,),
            cache_key=f"analytics:{EXPENSIVE}:all:limit{limit}",
            label=f"{EXPENSIVE}(all-time)",
        )

    name = _ROTATION[key_id % len(_ROTATION)]
    days = WINDOW_DAYS[(key_id // len(_ROTATION)) % len(WINDOW_DAYS)]
    region_id = (key_id // (len(_ROTATION) * len(WINDOW_DAYS))) % REGION_COUNT + 1
    sqlite_window = f"-{days} days"

    if name == "revenue_by_region":
        return QueryPlan(
            query=QUERIES[name],
            params_postgres=(days, limit),
            params_sqlite=(sqlite_window, limit),
            cache_key=f"analytics:{name}:all:{days}d:limit{limit}",
            label=f"{name}(all regions, {days}d)",
        )
    if name == "category_matrix":
        return QueryPlan(
            query=QUERIES[name],
            params_postgres=(days, limit),
            params_sqlite=(sqlite_window, limit),
            cache_key=f"analytics:{name}:all:{days}d:limit{limit}",
            label=f"{name}(all regions, {days}d)",
        )
    return QueryPlan(
        query=QUERIES[name],
        params_postgres=(region_id, days, limit),
        params_sqlite=(region_id, sqlite_window, limit),
        cache_key=f"analytics:{name}:region{region_id}:{days}d:limit{limit}",
        label=f"{name}(region {region_id}, {days}d)",
    )


async def run(backend: Backend, plan: QueryPlan) -> QueryResult:
    """Execute a plan against the active backend."""
    sql = plan.query.sql_for(backend.dialect)
    return await backend.fetch(sql, plan.params_for(backend.dialect))
