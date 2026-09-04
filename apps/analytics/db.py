"""Database access for the analytics application.

Two backends, one interface:

``PostgresBackend``
    asyncpg against Supabase Postgres, taken from
    `SUPABASE_DIRECT_CONNECTION_URL`. A connection pool, a server-side statement
    timeout, and no ORM - the SQL in `queries.py` is what runs.

``SqliteBackend``
    A local fixture with the identical schema, seeded on first use. It exists so
    the application always boots: a demo machine without Supabase credentials
    still gets real joins, real GROUP BY and real window functions, and real
    measured database time.

Both report the service time of each statement, which is what the application
hands to AURA as `db_ms`.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sqlite3
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from common.settings import Settings

log = logging.getLogger("aura.app.analytics.db")

REGIONS = (
    ("Mumbai", "IN"),
    ("Delhi", "IN"),
    ("Bengaluru", "IN"),
    ("Chennai", "IN"),
    ("Kolkata", "IN"),
    ("Pune", "IN"),
    ("Hyderabad", "IN"),
    ("Ahmedabad", "IN"),
    ("Singapore", "SG"),
    ("Dubai", "AE"),
    ("London", "GB"),
    ("Frankfurt", "DE"),
)
PRODUCT_CATEGORIES = (
    "electronics",
    "apparel",
    "grocery",
    "home",
    "beauty",
    "sports",
    "books",
    "toys",
    "automotive",
    "garden",
)
N_PRODUCTS = 800

SCHEMA_SQLITE = """
CREATE TABLE IF NOT EXISTS app_regions (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    country TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS app_products (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    category TEXT NOT NULL,
    unit_price REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS app_orders (
    id INTEGER PRIMARY KEY,
    region_id INTEGER NOT NULL,
    product_id INTEGER NOT NULL,
    qty INTEGER NOT NULL,
    amount REAL NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS app_orders_region_created ON app_orders (region_id, created_at);
CREATE INDEX IF NOT EXISTS app_orders_product_created ON app_orders (product_id, created_at);
"""


@dataclass
class QueryResult:
    """Rows plus the measured service time of the statement."""

    rows: list[dict[str, Any]]
    db_ms: float


class Backend(Protocol):
    """What the query layer needs from a database."""

    dialect: str

    async def start(self) -> None:
        """Open the pool or the fixture."""

    async def close(self) -> None:
        """Release resources."""

    async def fetch(self, sql: str, params: Sequence[Any]) -> QueryResult:
        """Run one statement and return its rows and service time."""


class PostgresBackend:
    """asyncpg pool against Supabase Postgres."""

    dialect = "postgres"

    def __init__(self, dsn: str, settings: Settings) -> None:
        self._dsn = dsn
        self._settings = settings
        self._pool: Any | None = None
        # The transaction pooler hands a different backend to every statement, so asyncpg's
        # prepared-statement cache addresses objects that are not there on the next call.
        # Turning it off is the documented requirement, not a tuning choice.
        self._pooled = ":6543" in dsn or "pooler.supabase.com" in dsn

    async def start(self) -> None:
        """Create the pool and pin a statement timeout on every connection."""
        import asyncpg  # imported lazily so the SQLite path needs no driver

        extra: dict[str, Any] = {}
        if self._pooled:
            # No prepared statements and no startup parameters through pgbouncer; the client
            # side command timeout still bounds a slow query.
            extra["statement_cache_size"] = 0
        else:
            extra["server_settings"] = {
                "statement_timeout": str(self._settings.db_statement_timeout_ms),
                "application_name": "aura-analytics",
            }
        self._pool = await asyncpg.create_pool(
            dsn=self._dsn,
            min_size=self._settings.db_pool_min_size,
            max_size=self._settings.db_pool_max_size,
            command_timeout=self._settings.db_statement_timeout_ms / 1000.0,
            **extra,
        )
        log.info(
            "postgres pool ready",
            extra={"event": "db_ready", "backend": "postgres", "pooled": self._pooled},
        )

    async def close(self) -> None:
        """Close the pool."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def fetch(self, sql: str, params: Sequence[Any]) -> QueryResult:
        """Execute one statement on a pooled connection."""
        if self._pool is None:
            raise RuntimeError("postgres pool is not started")
        started = time.perf_counter()
        async with self._pool.acquire() as connection:
            records = await connection.fetch(sql, *params)
        db_ms = (time.perf_counter() - started) * 1000.0
        return QueryResult([dict(record) for record in records], db_ms)


class SqliteBackend:
    """Local fixture with the same schema, seeded deterministically."""

    dialect = "sqlite"

    def __init__(self, path: str, settings: Settings) -> None:
        self._path = path
        self._settings = settings
        self._lock = asyncio.Lock()
        self._connection: sqlite3.Connection | None = None

    async def start(self) -> None:
        """Open (and seed on first use) the fixture database."""
        await asyncio.to_thread(self._open)
        log.info(
            "sqlite fixture ready",
            extra={"event": "db_ready", "backend": "sqlite", "path": self._path},
        )

    def _open(self) -> None:
        fresh = not os.path.exists(self._path) or os.path.getsize(self._path) == 0
        connection = sqlite3.connect(self._path, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        connection.executescript(SCHEMA_SQLITE)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        count = connection.execute("SELECT count(*) FROM app_orders").fetchone()[0]
        if fresh or count < self._settings.sqlite_orders // 2:
            _seed(connection, self._settings)
        self._connection = connection

    async def close(self) -> None:
        """Close the fixture."""
        if self._connection is not None:
            connection, self._connection = self._connection, None
            await asyncio.to_thread(connection.close)

    async def fetch(self, sql: str, params: Sequence[Any]) -> QueryResult:
        """Execute one statement. SQLite is serialised behind a lock."""
        if self._connection is None:
            raise RuntimeError("sqlite fixture is not started")
        async with self._lock:
            return await asyncio.to_thread(self._fetch_blocking, sql, tuple(params))

    def _fetch_blocking(self, sql: str, params: tuple[Any, ...]) -> QueryResult:
        assert self._connection is not None
        started = time.perf_counter()
        cursor = self._connection.execute(sql, params)
        rows = [dict(row) for row in cursor.fetchall()]
        cursor.close()
        db_ms = (time.perf_counter() - started) * 1000.0
        return QueryResult(rows, db_ms)


def _seed(connection: sqlite3.Connection, settings: Settings) -> None:
    """Populate the fixture with a deterministic order book."""
    rng = np.random.default_rng(settings.seed)
    connection.execute("DELETE FROM app_orders")
    connection.execute("DELETE FROM app_products")
    connection.execute("DELETE FROM app_regions")

    connection.executemany(
        "INSERT INTO app_regions (id, name, country) VALUES (?, ?, ?)",
        [(i + 1, name, country) for i, (name, country) in enumerate(REGIONS)],
    )

    prices = np.round(rng.gamma(shape=2.4, scale=22.0, size=N_PRODUCTS) + 3.0, 2)
    products = [
        (
            i + 1,
            f"{PRODUCT_CATEGORIES[i % len(PRODUCT_CATEGORIES)]}-sku-{i + 1:04d}",
            PRODUCT_CATEGORIES[i % len(PRODUCT_CATEGORIES)],
            float(prices[i]),
        )
        for i in range(N_PRODUCTS)
    ]
    connection.executemany(
        "INSERT INTO app_products (id, name, category, unit_price) VALUES (?, ?, ?, ?)",
        products,
    )

    n_orders = settings.sqlite_orders
    # Regions are unevenly sized, which is what makes the dashboard queries
    # interesting: a per-region query is cheap for Frankfurt and expensive for
    # Mumbai, at the same request rate.
    weights = 1.0 / np.power(np.arange(1, len(REGIONS) + 1), 0.75)
    weights /= weights.sum()
    region_ids = rng.choice(len(REGIONS), size=n_orders, p=weights) + 1
    product_ids = rng.integers(1, N_PRODUCTS + 1, size=n_orders)
    qty = rng.integers(1, 9, size=n_orders)
    now = time.time()
    ages = rng.power(0.6, size=n_orders) * 365 * 86_400.0
    timestamps = now - ages

    batch = []
    for i in range(n_orders):
        unit = prices[int(product_ids[i]) - 1]
        created = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(timestamps[i]))
        batch.append(
            (
                i + 1,
                int(region_ids[i]),
                int(product_ids[i]),
                int(qty[i]),
                round(float(unit) * int(qty[i]), 2),
                created,
            )
        )
    connection.executemany(
        "INSERT INTO app_orders (id, region_id, product_id, qty, amount, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        batch,
    )
    connection.commit()
    log.info("sqlite fixture seeded", extra={"event": "db_seed", "orders": n_orders})


async def open_backend(settings: Settings) -> Backend:
    """Pick Postgres when a DSN is configured, otherwise the SQLite fixture.

    A failure to reach Postgres is not fatal: the application logs it and falls
    back, because a demo that will not start is worse than a demo on fixtures.
    """
    for label, dsn in (
        ("direct", settings.supabase_direct_connection_url),
        ("pooler", settings.supabase_pooler_url),
    ):
        if not dsn:
            continue
        backend: Backend = PostgresBackend(dsn, settings)
        try:
            await backend.start()
            log.info("postgres ready", extra={"event": "db_route", "route": label})
            return backend
        except Exception as exc:
            log.warning(
                "postgres route unavailable",
                extra={"event": "db_route_failed", "route": label, "error": repr(exc)},
            )
    backend = SqliteBackend(settings.sqlite_path, settings)
    await backend.start()
    return backend
