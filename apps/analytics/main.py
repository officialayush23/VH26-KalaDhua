"""Analytics service - port 8102.

Economics of this application:

* objects are small (5-500 KB of aggregated rows);
* regeneration is dominated by database service time, not CPU: 30-1500 ms with
  a p95 far above the p50, because a query over Mumbai's order book costs many
  times one over Frankfurt's at exactly the same request rate;
* traffic has strong temporal locality - a dashboard is opened, refreshed a few
  times, and abandoned - with sharp bursts at the top of the hour;
* the objects are cheap to store and expensive to rebuild, which is the
  opposite of the content application.

A size-aware but cost-blind policy under-values these objects: they are tiny, so
keeping them looks free, yet they are exactly the ones worth keeping.
"""

from __future__ import annotations

import hashlib
import os
from typing import Any

import uvicorn
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from analytics import queries
from analytics.db import Backend, open_backend
from common.costing import CostMeter, CostVector
from common.service import AppService, build_app, configure_logging
from common.settings import get_settings

APPLICATION = "analytics"
# The platform picks the port in a deployed container (Railway, Render and Fly all
# set PORT and route only to it). The local default keeps the compose ports stable.
PORT = int(os.environ.get("PORT", 8102))
BASE_LIMIT = 400
KEY_SPACE = len(queries.WINDOW_DAYS) * queries.REGION_COUNT * 4


class AnalyticsService(AppService):
    """Dashboard query regeneration wired to AURA."""

    def __init__(self) -> None:
        super().__init__(
            application=APPLICATION,
            default_sla="high",
            work=self.produce,
            extra_metrics=self._extra,
        )
        self.backend: Backend | None = None
        self.db_ms_total = 0.0
        self.rows_total = 0

    async def ensure_backend(self) -> Backend:
        """Open the database on first use."""
        if self.backend is None:
            self.backend = await open_backend(self.settings)
        return self.backend

    async def aclose(self) -> None:
        """Close the pool alongside the usual shutdown."""
        await super().aclose()
        if self.backend is not None:
            await self.backend.close()
            self.backend = None

    def _extra(self) -> dict[str, float]:
        return {
            "aura_app_db_ms_total": round(self.db_ms_total, 3),
            "aura_app_db_rows_total": float(self.rows_total),
            "aura_app_db_postgres": 1.0 if (self.backend and self.backend.dialect == "postgres") else 0.0,
        }

    def cache_key(self, key_id: int | str) -> str:
        """Cache key for a dashboard query, including its bound parameters."""
        numeric = _numeric(key_id)
        plan = queries.plan_for(numeric, expensive=self.tail.contains(key_id), limit=BASE_LIMIT)
        return plan.cache_key

    async def produce(self, key_id: int | str, fresh: bool, options: dict[str, str] | None = None) -> dict[str, Any]:
        """Serve one dashboard query, through the cache."""
        backend = await self.ensure_backend()
        numeric = _numeric(key_id)
        expensive = self.tail.contains(key_id)
        limit = BASE_LIMIT
        plan = queries.plan_for(numeric, expensive=expensive, limit=limit)

        async def regen(meter: CostMeter) -> tuple[dict[str, Any], CostVector]:
            result = await queries.run(backend, plan)
            meter.add_db_ms(result.db_ms)
            self.db_ms_total += result.db_ms
            self.rows_total += len(result.rows)
            body = {
                "query": plan.query.name,
                "label": plan.label,
                "dialect": backend.dialect,
                "row_count": len(result.rows),
                "db_ms": round(result.db_ms, 3),
                "rows": result.rows,
            }
            return body, CostVector(db_ms=result.db_ms)

        outcome = await self.client.get_or_regen_detailed(
            plan.cache_key,
            object_type=plan.query.object_type,
            ttl_ms=plan.query.ttl_ms,
            regen=regen,
            sla_class=plan.query.sla_class,
            force_fresh=fresh,
        )
        self.account(object_type=plan.query.object_type, outcome=outcome, key_id=key_id)

        value = outcome.value if isinstance(outcome.value, dict) else {}
        body: dict[str, Any] = {
            "key": plan.cache_key,
            "application": APPLICATION,
            "object_type": plan.query.object_type,
            "query": plan.query.name,
            "label": plan.label,
            "served_from": outcome.served_from,
            "expensive_tail": expensive,
            "size_bytes": outcome.size_bytes,
            "serve_ms": round(outcome.serve_ms, 3),
            "regen": outcome.cost.model_dump(),
            "regen_cost_usd": round(outcome.cost_usd, 8),
            "admitted": outcome.admitted,
            "reason_code": outcome.reason_code,
            "row_count": value.get("row_count", 0),
        }
        if _truthy((options or {}).get("rows")):
            body["rows"] = value.get("rows", [])
        else:
            body["rows_preview"] = (value.get("rows") or [])[:5]
        return body


def _numeric(key_id: int | str) -> int:
    try:
        return abs(int(key_id))
    except (TypeError, ValueError):
        digest = hashlib.blake2b(str(key_id).encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "big") % (1 << 31)


def _truthy(value: str | None) -> bool:
    return str(value).lower() in {"1", "true", "yes", "on"}


def create_app():  # noqa: ANN201 - Starlette application factory
    """Build the ASGI application."""
    configure_logging(get_settings().log_level)
    service = AnalyticsService()

    async def profile(request: Request) -> JSONResponse:
        """Cost profile and the query catalogue."""
        backend = service.backend
        return JSONResponse(
            {
                "application": APPLICATION,
                "cost_profile": "db_heavy",
                "traffic_shape": "bursty_temporal_locality",
                "object_bytes": [400, 500_000],
                "regen_ms_range": [30, 1500],
                "backend": backend.dialect if backend else "not_opened",
                "key_space": KEY_SPACE,
                "queries": [
                    {
                        "name": query.name,
                        "object_type": query.object_type,
                        "ttl_ms": query.ttl_ms,
                        "sla_class": query.sla_class,
                    }
                    for query in queries.QUERIES.values()
                ],
                "expensive_tail_query": queries.EXPENSIVE,
            }
        )

    async def explain_sql(request: Request) -> JSONResponse:
        """The exact SQL a given key id will run - useful when demoing."""
        numeric = _numeric(request.path_params["key_id"])
        expensive = _truthy(request.query_params.get("expensive"))
        plan = queries.plan_for(numeric, expensive=expensive, limit=BASE_LIMIT)
        dialect = service.backend.dialect if service.backend else "sqlite"
        return JSONResponse(
            {
                "key": plan.cache_key,
                "query": plan.query.name,
                "dialect": dialect,
                "params": list(plan.params_for(dialect)),
                "sql": plan.query.sql_for(dialect).strip(),
            }
        )

    async def health(request: Request) -> JSONResponse:
        """Health, including a real round trip to the database."""
        try:
            backend = await service.ensure_backend()
            probe = await backend.fetch("SELECT count(*) AS n FROM app_orders", ())
            db_ok = True
            orders = int(next(iter(probe.rows[0].values())))
            probe_ms = round(probe.db_ms, 3)
        except Exception as exc:
            db_ok, orders, probe_ms = False, 0, 0.0
            service.log.warning("db probe failed", extra={"event": "db_probe_failed", "error": repr(exc)})
        return JSONResponse(
            {
                "ok": db_ok,
                "application": APPLICATION,
                "backend": service.backend.dialect if service.backend else "unavailable",
                "orders": orders,
                "probe_ms": probe_ms,
                "cache_breaker": service.client.breaker_state,
                "expensive_tail": service.tail.state(),
            },
            status_code=200 if db_ok else 503,
        )

    routes = [
        Route("/health", health, methods=["GET"]),
        Route("/profile", profile, methods=["GET"]),
        Route("/sql/{key_id}", explain_sql, methods=["GET"]),
    ]
    return build_app(service, routes)


app = create_app()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_config=None, timeout_graceful_shutdown=10)
